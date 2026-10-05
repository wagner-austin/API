"""The watch beside a serving node runner: close a run the moment it ends.

WHY A RUNNER WATCHES (MCPs board task c1d48330). Each node's runner ran its
collect pass and its fill pass once per 3-minute scheduled start, so a run
that ended was closed, and its verdict posted, only on the next start:
claimed-to-closed was a whole number of ticks and every session waiting on a
fleet verdict waited up to 3 minutes after its check had ended. Measured on
2026-10-04 (board task 74b13c20): serendipity's osm-mcp row ended 19:27:05Z
and closed 19:30:21Z, its search row ended 19:52:18Z and closed 19:54:17Z,
and lavender-wsl's store-hit rows read exactly 181 s claimed-to-closed
whatever their work.

WHY ON A THREAD, BESIDE THE PASSES. A launch takes about 15 s on diphtheria,
42 s on serendipity and 90 to 155 s on loki, which no poll between two
launches can cover, so the watch runs on a thread of its own for the whole
serve (:mod:`fleet.cli.node_serve`): the collect pass hands it the runs the
queue says this runner holds as soon as it has asked
(:func:`fleet.cli.node_collect.collect_pass`), the fill pass each run the
moment it is launched (:func:`fleet.cli.node_agent.fill_pass`), and every
``node_poll_seconds`` of the workspace it reads each held run's result off the node
(:func:`fleet.core.collect.poll_result`, one ssh read) and settles one that
has ended (:func:`collect_ended`), while the passes go on in the main
thread. A run is never settled twice: both paths settle through
:func:`fleet.cli.node_collect.collect_one_job`, which holds one lock and
finds a settled run no longer live.

WHY IT ENDS ONLY WHEN IT IS CLOSED (MCPs board task 8993c306). Until
2026-10-05 the watch ended at a fixed window, 100 s into each 180 s start,
so a run ending in the other 80 s, or while a long fill held the start past
the next fire, waited for a later start: loki's 05:51:03Z start launched
until about 05:55:01Z, the 05:54Z fire was skipped, and tools-fleet-wake,
which ended at 05:53:35Z, closed at 05:57:44Z. The watch now polls until
the serving loop hands over, which it does only through :meth:`RunWatch.close_if_idle`,
refused while a settle is under way, so a handover never cuts a settle
short and no settle starts once the handover is decided; a run ending after
that is settled by the next start's collect pass. Between polls it waits on
its condition rather than sleeping, so a handover wakes it at once instead
of after the rest of a poll.

WHAT IT COSTS, AND WHAT IT DOES NOT. While the runner holds no running run
the thread waits on a condition, with no sleep and no call, so a runner
holding nothing makes no call beyond its passes, and a poll that finds a run
still going writes nothing anywhere: absence of a result is the ordinary
answer (:mod:`fleet.core.collect`).
"""

from __future__ import annotations

import threading
from types import TracebackType
from typing import Final, Protocol

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.cli import collect as collect_cli
from fleet.cli.node_collect import collect_one_job
from fleet.contracts.dispatch import DispatchStatus
from fleet.contracts.node import NodeConfig
from fleet.core import collect, queue

_log = get_logger(__name__)

#: The prefix of the watch thread's name; the executor appends ``_0``.
THREAD_PREFIX: Final = "fleet-watch"


class Settle(Protocol):
    """Close out one held run whose node has written its result."""

    def __call__(self, run_id: str) -> str:
        """Settle the run.

        Args:
            run_id: The run that ended.

        Returns:
            One line saying what happened, for the log.
        """


def live_among(loaded: _config.LoadedWorkspace, run_ids: frozenset[str]) -> frozenset[str]:
    """The run ids this machine's ledger still calls running.

    Args:
        loaded: The workspace and its resolved record paths.
        run_ids: The runs to look for.

    Returns:
        Those of them that are still live, read from the local ledger, so
        asking costs no ssh and no queue call.
    """
    return frozenset(
        row["run_id"]
        for row in collect_cli.live_rows(loaded, run_id=None)
        if row["run_id"] in run_ids
    )


def collect_ended(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    board: McpCredentials,
    identity: JSONObject,
    *,
    agent: str,
    run_id: str,
) -> str:
    """Settle the queue job of one run this runner holds that has ended.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        board: The board's endpoint and headers, for the verdict.
        identity: This runner's identity arguments.
        agent: This runner's label.
        run_id: The run whose result the node has written.

    Returns:
        The settle's line, or a line saying the queue no longer has this
        runner holding the run as running: a cancel or a takeover, which the
        next fire's collect pass stops (:func:`fleet.cli.node_collect.stop_cancelled`).

    Raises:
        AppError: As :func:`fleet.cli.node_collect.collect_one_job` and
            :func:`fleet.core.queue.held_by` describe. Not caught.
    """
    for job in queue.held_by(credentials, agent=agent):
        if job["run_id"] == run_id and job["status"] is DispatchStatus.RUNNING:
            return collect_one_job(loaded, credentials, board, job, identity)
    return f"{run_id}: no running job of {agent} names it now; the next fire's collect reads it"


class RunWatch:
    """The runs a serving runner holds, watched on a thread of their own.

    The serving loop calls :meth:`hold` and :meth:`close_if_idle` from the
    main thread inside ``with watch``, whose exit closes the watch whatever
    happened; :meth:`watch` runs on the watch thread. Every field the two
    share while both run is read and written under one condition.

    Attributes:
        polls: How many times the watch read the held runs' results.
    """

    polls: int

    def __init__(
        self,
        loaded: _config.LoadedWorkspace,
        *,
        alias: str,
        node: NodeConfig,
        settle: Settle,
    ) -> None:
        """Bind the watch to its node; nothing is held and nothing polled.

        Args:
            loaded: The workspace and its resolved record paths, whose
                ``node_poll_seconds`` paces the polls.
            alias: This node's workspace name.
            node: Its declaration.
            settle: Closes out a held run that has ended.
        """
        self._loaded = loaded
        self._alias = alias
        self._node = node
        self._settle = settle
        self._poll_seconds = loaded.workspace["node_poll_seconds"]
        self._changed = threading.Condition()
        self._held: frozenset[str] = frozenset()
        self._closing = False
        self._settling = False
        self._closed = 0
        self.polls = 0

    def hold(self, run_ids: frozenset[str]) -> None:
        """Add runs to the watch, keeping those the ledger calls running.

        Args:
            run_ids: Runs the queue says this runner holds running, or one
                it has just launched.
        """
        live = live_among(self._loaded, run_ids)
        with self._changed:
            self._held = self._held | live
            self._changed.notify()

    def closed(self) -> int:
        """How many runs the watch has settled so far.

        Returns:
            Their count, which the serving loop compares to decide that a
            settle has freed room since its last fill.
        """
        with self._changed:
            return self._closed

    def close_if_idle(self) -> bool:
        """Close the watch unless a settle is under way.

        Returns:
            True when the watch is now closed, and will start no settle;
            False while it is settling a run, which the handover must not
            cut short.
        """
        with self._changed:
            if self._settling:
                return False
            self._closing = True
            self._changed.notify()
            return True

    def __enter__(self) -> RunWatch:
        """Open the span of the serving loop.

        Returns:
            This watch.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the watch, so it ends instead of holding the runner open,
        whether the loop handed over or raised; an exception goes on.

        Args:
            exc_type: The loop's exception type, or None.
            exc: Its exception, or None.
            traceback: Its traceback, or None.
        """
        with self._changed:
            self._closing = True
            self._changed.notify()

    def still_watched(self) -> int:
        """How many held runs the ledger still calls running.

        Returns:
            Their count.
        """
        with self._changed:
            held = self._held
        return len(live_among(self._loaded, held))

    def _woken(self) -> bool:
        """Whether there is a run to watch or the watch is closed.

        Returns:
            True once a run is held or the watch is closed.
        """
        return bool(self._held) or self._closing

    def _is_closing(self) -> bool:
        """Whether the watch has been closed.

        Returns:
            The flag, read under the condition by its waits.
        """
        return self._closing

    def _next(self) -> frozenset[str]:
        """Wait for a run to watch, then one poll, unless the watch is closed.

        Returns:
            The held runs to poll now, or empty once the watch is closed,
            before a run arrived or during the poll's wait.
        """
        with self._changed:
            self._changed.wait_for(self._woken)
            if not self._closing:
                self._changed.wait_for(self._is_closing, timeout=self._poll_seconds)
            return frozenset() if self._closing else self._held

    def _begin_settle(self) -> bool:
        """Mark a settle under way, unless the watch has been closed.

        Returns:
            True when the settle may start.
        """
        with self._changed:
            if self._closing:
                return False
            self._settling = True
            return True

    def _end_settle(self) -> None:
        """Mark the settle under way over, finished or raised."""
        with self._changed:
            self._settling = False

    def _settled(self, run_id: str) -> None:
        """Count a settled run and stop watching it.

        Args:
            run_id: The run.
        """
        with self._changed:
            self._closed += 1
            self._held = self._held - {run_id}

    def _drop(self, run_ids: frozenset[str]) -> None:
        """Stop watching runs no longer live.

        Args:
            run_ids: The runs to drop.
        """
        with self._changed:
            self._held = self._held - run_ids

    def watch(self) -> None:
        """Poll the held runs until the watch is closed.

        Raises:
            AppError: From a poll or a settle, which the serving loop sees
                as this thread's end and raises once it has stopped
                (:func:`fleet.cli.node_serve.serve`).
        """
        while watched := self._next():
            self.polls += 1
            for run_id in sorted(live_among(self._loaded, watched)):
                if collect.poll_result(self._node, run_id=run_id) is None:
                    continue
                if not self._begin_settle():
                    return
                _log.info("%s: %s has ended; collecting it now", self._alias, run_id)
                try:
                    line = self._settle(run_id)
                finally:
                    self._end_settle()
                _log.info("%s", line)
                self._settled(run_id)
            self._drop(watched - live_among(self._loaded, watched))


__all__ = [
    "THREAD_PREFIX",
    "RunWatch",
    "Settle",
    "collect_ended",
    "live_among",
]
