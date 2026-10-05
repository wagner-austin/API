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
thread. A run is never settled twice, and never read while it is retired:
the watch holds the run's lock in :data:`fleet.cli.run_locks.SETTLING`
across each read and the settle it starts, the collect pass holds it for
each job it settles and each run it stops, and a run the ledger no longer
calls running is not read.

WHY EACH RUN IS CHECKED ON A TASK OF ITS OWN (MCPs board task 8993c306).
The watch read its runs one after another and settled each inline, so a run
ending beside another waited for that one's settle, about 8 s of ssh and
queue calls: on 2026-10-05 libs/platform_core's settle on lavender-wsl ran
from 08:20:00Z to 08:20:13Z, and row ebb9009c, whose check ended at
08:20:01Z, closed at 08:20:23Z. Now every poll hands each held run that is
not already being checked to a task on the watch's pool
(:data:`CHECK_WORKERS`), which reads it and settles it if it has ended, so
one run's settle holds back no other run's read or settle.

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
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Final, Protocol

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.cli import collect as collect_cli
from fleet.cli.node_collect import collect_one_job
from fleet.cli.run_locks import SETTLING
from fleet.contracts.dispatch import DispatchStatus
from fleet.contracts.node import NodeConfig
from fleet.core import collect, queue

_log = get_logger(__name__)

#: The prefix of the watch thread's name; the executor appends ``_0``.
THREAD_PREFIX: Final = "fleet-watch"

#: How many held runs the watch checks at once: more than any node holds
#: live, so a check waits for a worker only past that.
CHECK_WORKERS: Final = 8


class Settle(Protocol):
    """Close out one held run whose node has written its result."""

    def __call__(self, *, run_id: str) -> str:
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

    The serving loop calls :meth:`hold` and :meth:`close_if_idle` from its
    own thread, and :func:`fleet.cli.node_serve.serve_on` calls
    :meth:`close_if_idle` from the main thread once that loop has failed,
    all inside ``with watch``, whose exit closes the watch whatever
    happened; :meth:`watch` runs on the watch thread. Every field they
    share while they run is read and written under one condition.

    Attributes:
        polls: How many polls the watch made, each handing every held run
            no check held to a check of its own.
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
        self._checking: frozenset[str] = frozenset()
        self._closing = False
        self._failed: Future[None] | None = None
        self._settling = 0
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
            False while it is settling a run, any run, which the handover
            must not cut short.
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
        """Whether a held run waits for a check, or the watch is stopping.

        Returns:
            True once a held run is not being checked, or :meth:`_stopping`.
        """
        return bool(self._held - self._checking) or self._stopping()

    def _stopping(self) -> bool:
        """Whether the watch has been closed or a check of it has raised.

        Returns:
            The answer, read under the condition by its waits.
        """
        return self._closing or self._failed is not None

    def _next(self) -> frozenset[str] | None:
        """Wait for a held run no check holds, then one poll, and take those due.

        Returns:
            The held runs to check now, marked as being checked, which may
            be none when every one settled or left during the poll's wait;
            None once the watch is closed or a check has raised, before a
            run arrived or during the wait.
        """
        with self._changed:
            self._changed.wait_for(self._woken)
            if not self._stopping():
                self._changed.wait_for(self._stopping, timeout=self._poll_seconds)
            if self._stopping():
                return None
            due = self._held - self._checking
            self._checking = self._checking | due
            return due

    def _begin_settle(self) -> bool:
        """Count a settle under way, unless the watch has been closed.

        Returns:
            True when the settle may start.
        """
        with self._changed:
            if self._closing:
                return False
            self._settling += 1
            return True

    def _end_settle(self) -> None:
        """Count one settle under way over, finished or raised."""
        with self._changed:
            self._settling -= 1

    def _settled(self, run_id: str) -> None:
        """Count a settled run and stop watching it.

        Args:
            run_id: The run.
        """
        with self._changed:
            self._closed += 1
            self._held = self._held - {run_id}

    def _unchecked(self, run_id: str) -> None:
        """End a run's check, and stop watching it if it is no longer live.

        Args:
            run_id: The run.
        """
        live = live_among(self._loaded, frozenset({run_id}))
        with self._changed:
            self._checking = self._checking - {run_id}
            self._held = self._held - ({run_id} - live)
            self._changed.notify()

    def _checked(self, check: Future[None]) -> None:
        """Keep the first check that raised, so the watch stops and raises it.

        Args:
            check: A check that has finished.
        """
        if check.exception() is None:
            return
        with self._changed:
            if self._failed is None:
                self._failed = check
            self._changed.notify()

    def _check(self, run_id: str) -> None:
        """Read one held run and settle it if it has ended, on a task of its own.

        Args:
            run_id: The run.

        Raises:
            AppError: From the read or the settle. Not caught: the watch
                stops and raises it (:meth:`watch`).
        """
        try:
            with SETTLING.holding(run_id):
                if not self.ended(run_id) or not self._begin_settle():
                    return
                _log.info("%s: %s has ended; collecting it now", self._alias, run_id)
                try:
                    line = self._settle(run_id=run_id)
                finally:
                    self._end_settle()
            _log.info("%s", line)
            self._settled(run_id)
        finally:
            self._unchecked(run_id)

    def watch(self) -> None:
        """Check the held runs every poll until the watch is closed or a check raises.

        Raises:
            AppError: From a check's read or settle, once every check under
                way has finished, which the serving loop sees as this
                thread's end and raises once it has stopped
                (:func:`fleet.cli.node_serve.serve`).
        """
        with ThreadPoolExecutor(
            max_workers=CHECK_WORKERS, thread_name_prefix=f"{THREAD_PREFIX}-{self._alias}-check"
        ) as pool:
            while (due := self._next()) is not None:
                self.polls += 1
                for run_id in sorted(due):
                    pool.submit(self._check, run_id).add_done_callback(self._checked)
        # The pool's exit waited for every check, so no failure is still to come.
        with self._changed:
            failed = self._failed
        if failed is not None:
            failed.result()

    def ended(self, run_id: str) -> bool:
        """Read one held run's result, if this machine still calls it running.

        The watch calls it under the run's lock in
        :data:`fleet.cli.run_locks.SETTLING`, so the ledger's answer cannot
        change before the read: a run the collect pass settled or stopped
        meanwhile is not read, since the read sends its script into the
        run's directory, the one its retire removes.

        Args:
            run_id: The run.

        Returns:
            True when it is live and the node has written its result; False
            too when the node did not answer, which is logged and read again
            at the next poll (:func:`fleet.core.collect.attempt_poll_result`).

        Raises:
            AppError: When the node answered and the read failed there, or
                its answer was unreadable.
        """
        if run_id not in live_among(self._loaded, frozenset({run_id})):
            return False
        polled = collect.attempt_poll_result(self._node, run_id=run_id)
        if polled["unreachable"] is not None:
            _log.info(
                "%s did not answer the read of %s; it is read again at the next poll: %s",
                self._alias,
                run_id,
                polled["unreachable"],
            )
            return False
        return polled["result"] is not None


__all__ = [
    "CHECK_WORKERS",
    "THREAD_PREFIX",
    "RunWatch",
    "Settle",
    "collect_ended",
    "live_among",
]
