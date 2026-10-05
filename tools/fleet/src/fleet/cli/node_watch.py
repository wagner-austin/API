"""The watch beside a node runner's tick: close a run the moment it ends.

WHY A TICK WATCHES (MCPs board task c1d48330). Each node's runner is a
scheduled task repeating every 3 minutes under IgnoreNew, and a tick ran its
collect pass and its fill pass once, so a run that ended was closed, and its
verdict posted, only on the next tick: claimed-to-closed was a whole number
of ticks and every session waiting on a fleet verdict waited up to 3 minutes
after its check had ended. Measured on 2026-10-04 (board task 74b13c20):
serendipity's osm-mcp row ended 19:27:05Z and closed 19:30:21Z, its search
row ended 19:52:18Z and closed 19:54:17Z, lavender-wsl's store-hit rows read
exactly 181 s claimed-to-closed whatever their work, and a tick spent about
170 of its 180 s doing nothing.

WHY ON A THREAD, BESIDE THE PASSES. The first watch (rolled 2026-10-04 at
fcf2aa507) began only after the tick's passes, so a run ending while they ran
waited for them: 33fdd720 ended 04:06:14Z during diphtheria's passes and
closed 34 s later, and a launch alone takes about 15 s on diphtheria, 42 s on
serendipity and 90 to 155 s on loki, which no poll between two launches can
cover. So the watch starts with the tick: the collect pass hands it the runs
the queue says this runner holds as soon as it has asked
(:func:`fleet.cli.node_collect.collect_pass`), the fill pass each run the
moment it is launched (:func:`fleet.cli.node_agent.fill_pass`), and every
:data:`POLL_SECONDS` it reads each held run's result off the node
(:func:`fleet.core.collect.poll_result`, one ssh read) and settles one that
has ended (:func:`collect_ended`), while the passes go on in the main
thread. A run is never settled twice: both paths settle through
:func:`fleet.cli.node_collect.collect_one_job`, which holds one lock and
finds a settled run no longer live.

WHY IT RERUNS NO PASS. That first watch ran both passes again when a run
ended, and a fill pass is as long as its launches: loki's 22:30:04Z tick
reran fill at 22:31:51Z, launched two jobs, exited at 401 s and skipped the
22:33Z and 22:36Z ticks under IgnoreNew. The watch now only settles; the
room a settled run frees is filled by the next tick, as before the watch.
It starts no poll whose wait would end past the ``node_watch_seconds``
window counted from the tick's start, so all that can outlast the window
is one settle already under way.

WHAT IT COSTS, AND WHAT IT DOES NOT. While the runner holds no running run
the thread waits on a condition, with no sleep and no call, so a runner
holding nothing makes no call beyond its passes, and a poll that finds a run
still going writes nothing anywhere: absence of a result is the ordinary
answer (:mod:`fleet.core.collect`).
"""

from __future__ import annotations

import datetime
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
from fleet.core import _test_hooks, collect, queue

_log = get_logger(__name__)

#: How long the watch waits between reads of the held runs' results.
POLL_SECONDS: Final[int] = 5

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
        next tick's collect pass stops (:func:`fleet.cli.node_collect.stop_cancelled`).

    Raises:
        AppError: As :func:`fleet.cli.node_collect.collect_one_job` and
            :func:`fleet.core.queue.held_by` describe. Not caught.
    """
    for job in queue.held_by(credentials, agent=agent):
        if job["run_id"] == run_id and job["status"] is DispatchStatus.RUNNING:
            return collect_one_job(loaded, credentials, board, job, identity)
    return f"{run_id}: no running job of {agent} names it now; the next tick's collect reads it"


class RunWatch:
    """The runs a runner holds this tick, watched on a thread of their own.

    The passes call :meth:`hold` from the main thread inside ``with watch``,
    whose exit says they are over; :meth:`watch` runs on the watch thread.
    Every field the two share while both run is read and written under one
    condition.

    Attributes:
        deadline: When the window closes, whole seconds since the epoch.
        polls: How many times the watch read the held runs' results.
        closed: How many runs it settled.
        held_any: Whether any run this machine's ledger calls running was
            ever held, which decides the line the tick ends with.
    """

    deadline: int
    polls: int
    closed: int
    held_any: bool

    def __init__(
        self,
        loaded: _config.LoadedWorkspace,
        *,
        alias: str,
        node: NodeConfig,
        settle: Settle,
    ) -> None:
        """Start the window at the current time.

        Args:
            loaded: The workspace and its resolved record paths, whose
                ``node_watch_seconds`` is the window.
            alias: This node's workspace name.
            node: Its declaration.
            settle: Closes out a held run that has ended.
        """
        self._loaded = loaded
        self._alias = alias
        self._node = node
        self._settle = settle
        self.deadline = _test_hooks.now() + loaded.workspace["node_watch_seconds"]
        self._changed = threading.Condition()
        self._held: frozenset[str] = frozenset()
        self._passes_running = True
        self.polls = 0
        self.closed = 0
        self.held_any = False

    def hold(self, run_ids: frozenset[str]) -> None:
        """Add runs to the watch, keeping those the ledger calls running.

        Args:
            run_ids: Runs the queue says this runner holds running, or one
                it has just launched.
        """
        live = live_among(self._loaded, run_ids)
        with self._changed:
            self._held = self._held | live
            self.held_any = self.held_any or bool(live)
            self._changed.notify()

    def __enter__(self) -> RunWatch:
        """Open the span of the tick's passes.

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
        """Say the tick's passes are over, raised or not, so an empty watch
        ends instead of holding the tick open; an exception goes on.

        Args:
            exc_type: The passes' exception type, or None.
            exc: Their exception, or None.
            traceback: Its traceback, or None.
        """
        with self._changed:
            self._passes_running = False
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
        """Whether there is a run to watch or nothing more will come.

        Returns:
            True once a run is held or the passes are over.
        """
        return bool(self._held) or not self._passes_running

    def _next(self) -> frozenset[str]:
        """Wait for runs to watch, and say whether another poll fits.

        Returns:
            The held runs when there are some and a wait of
            :data:`POLL_SECONDS` still ends inside the window; empty when the
            watch is over.
        """
        with self._changed:
            self._changed.wait_for(self._woken)
            if self._held and _test_hooks.now() + POLL_SECONDS <= self.deadline:
                return self._held
            return frozenset()

    def _drop(self, run_ids: frozenset[str]) -> None:
        """Stop watching runs settled or no longer live.

        Args:
            run_ids: The runs to drop.
        """
        with self._changed:
            self._held = self._held - run_ids

    def watch(self) -> None:
        """Poll the held runs until the window closes or none is left.

        Raises:
            AppError: From a poll or a settle, which the tick raises once its
                passes are over (:func:`run_tick`).
        """
        while watched := self._next():
            _test_hooks.sleep(POLL_SECONDS)
            self.polls += 1
            live = live_among(self._loaded, watched)
            for run_id in sorted(live):
                if collect.poll_result(self._node, run_id=run_id) is None:
                    continue
                _log.info("%s: %s has ended; collecting it now", self._alias, run_id)
                _log.info("%s", self._settle(run_id))
                self.closed += 1
                self._drop(frozenset({run_id}))
            self._drop(watched - live_among(self._loaded, watched))


def run_tick(watch: RunWatch, *, alias: str, passes: Callable[[], None]) -> None:
    """Run a tick's passes with the watch beside them.

    Args:
        watch: The watch the passes hand their runs to.
        alias: This node's workspace name, which names the thread.
        passes: The runner's collect pass followed by its fill pass.

    Raises:
        AppError: From the passes, once the watch has ended; or from the
            watch, once the passes have. Neither is caught: the next tick
            meets the same runs.
    """
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{THREAD_PREFIX}-{alias}") as pool:
        watching = pool.submit(watch.watch)
        with watch:
            passes()
        watching.result()
    if not watch.held_any:
        _log.info("%s holds no running job; no watch this tick", alias)
        return
    until = datetime.datetime.fromtimestamp(watch.deadline, tz=datetime.UTC).isoformat()
    _log.info(
        "%s watch until %s: %d poll(s), %d run(s) closed, %d run(s) still watched",
        alias,
        until,
        watch.polls,
        watch.closed,
        watch.still_watched(),
    )


__all__ = [
    "POLL_SECONDS",
    "THREAD_PREFIX",
    "RunWatch",
    "Settle",
    "collect_ended",
    "live_among",
    "run_tick",
]
