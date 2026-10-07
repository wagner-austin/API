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

ONLY A SETTLE IS A CLOSE, AND A NODE THAT MISSED A READ IS READ AGAIN (MCPs
board task c1d48330). The watch counted every answer of its settle as a
closed run and stopped watching it, so when lavender-wsl stopped answering
ssh at 02:53Z on 2026-10-07, row 24c4e934's settle answered "did not answer
the read", was counted closed, and the row waited for the next start, 192 s
after its check. A collect now says what it did
(:class:`fleet.cli.node_collected.Collected`): only a settle counts, a node
that did not answer keeps the run held for the next poll, and a run no
longer this runner's is let go. A run whose node did not answer the fire
boundary's read, whose lease that pass therefore did not renew, is OWED a
renewal (:meth:`RunWatch.owe`), which the watch makes at its first read that
reaches the node and finds it still going, so a node's few missed seconds do
not leave a renewal gap of two fires (374f0656 on lavender-wsl: renewed at
02:57:56Z, then not until 03:02:54Z).
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from enum import StrEnum
from types import TracebackType
from typing import Final, Protocol

from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.cli import collect as collect_cli
from fleet.cli.node_collect import collect_one_job
from fleet.cli.node_collected import Collected, CollectOutcome
from fleet.cli.run_locks import SETTLING
from fleet.contracts.dispatch import DispatchStatus
from fleet.contracts.node import NodeConfig
from fleet.core import collect, queue
from fleet.core.queue_transport import unanswered

_log = get_logger(__name__)

#: The prefix of the watch thread's name; the executor appends ``_0``.
THREAD_PREFIX: Final = "fleet-watch"

#: How many held runs the watch checks at once: more than any node holds
#: live, so a check waits for a worker only past that.
CHECK_WORKERS: Final = 8


class Settle(Protocol):
    """Collect one held run: settle it once ended, renew it when owed."""

    def __call__(self, *, run_id: str) -> Collected:
        """Collect the run.

        Args:
            run_id: The run that ended, or whose renewal is owed.

        Returns:
            What the collect did, and the line for the log.
        """


class RunRead(StrEnum):
    """What one read of a held run found."""

    #: The node has written its result.
    ENDED = "ended"
    #: The run is still going.
    RUNNING = "running"
    #: The node did not answer; it is read again at the next poll.
    UNREACHABLE = "unreachable"
    #: This machine's ledger no longer calls the run live, so it was not read.
    GONE = "gone"


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
) -> Collected:
    """Collect the queue job of one run this runner holds: settle it, or renew it.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        board: The board's endpoint and headers, for the verdict.
        identity: This runner's identity arguments.
        agent: This runner's label.
        run_id: The run whose result the node has written, or whose renewal
            is owed.

    Returns:
        What :func:`fleet.cli.node_collect.collect_one_job` did, or
        :attr:`~fleet.cli.node_collected.CollectOutcome.NOT_HELD` when the
        queue no longer has this runner holding the run as running: a cancel
        or a takeover, which the next fire's collect pass stops
        (:func:`fleet.cli.node_collect.stop_cancelled`).

    Raises:
        AppError: As :func:`fleet.cli.node_collect.collect_one_job` and
            :func:`fleet.core.queue.held_by` describe. Not caught.
    """
    for job in queue.held_by(credentials, agent=agent):
        if job["run_id"] == run_id and job["status"] is DispatchStatus.RUNNING:
            return collect_one_job(loaded, credentials, board, job, identity)
    return Collected(
        outcome=CollectOutcome.NOT_HELD,
        line=f"{run_id}: no running job of {agent} names it now; the next fire's collect reads it",
    )


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
        self._owed: frozenset[str] = frozenset()
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

    def owe(self, run_ids: frozenset[str]) -> None:
        """Hold runs whose renewal a collect pass could not make, and owe each one.

        The watch renews an owed run at its first read that reaches the
        node and finds the run still going, through the same collect
        (:data:`Settle`), which renews it then since its lease was last set
        a fire or more ago.

        Args:
            run_ids: Running runs whose node did not answer the pass's read.
        """
        live = live_among(self._loaded, run_ids)
        with self._changed:
            self._held = self._held | live
            self._owed = self._owed | live
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

    def _collected(self, run_id: str, outcome: CollectOutcome) -> None:
        """Account for what a collect of a held run did.

        A settle is counted closed and the run let go; a run the queue no
        longer has this runner holding is let go uncounted; a node that did
        not answer leaves the run held, and owed if it was; anything else
        pays what was owed.

        Args:
            run_id: The run.
            outcome: What the collect did.
        """
        with self._changed:
            if outcome is not CollectOutcome.UNREACHABLE:
                self._owed = self._owed - {run_id}
            if outcome is CollectOutcome.SETTLED:
                self._closed += 1
            if outcome in (CollectOutcome.SETTLED, CollectOutcome.NOT_HELD):
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

    def _due(self, run_id: str, found: RunRead) -> str | None:
        """Why a read calls for a collect of the run, if it does.

        Args:
            run_id: The run.
            found: What the read found.

        Returns:
            The reason, for the log, or None when there is nothing to collect.
        """
        if found is RunRead.ENDED:
            return "has ended"
        with self._changed:
            owed = run_id in self._owed
        if found is RunRead.RUNNING and owed:
            return "is still running and its renewal is owed"
        return None

    def _check(self, run_id: str) -> None:
        """Read one held run and collect it if it has ended or is owed a renewal.

        On a task of its own.

        Args:
            run_id: The run.

        A settle the queue did not answer (``QUEUE_UNANSWERED``) changed
        nothing (:mod:`fleet.cli.node_settle`), so the run stays held, its
        row still live, and is read and settled again at the next poll.

        Raises:
            AppError: From the read or the collect, but for
                ``QUEUE_UNANSWERED``. Not caught: the watch stops and raises
                it (:meth:`watch`).
        """
        try:
            with SETTLING.holding(run_id):
                reason = self._due(run_id, self.read(run_id))
                if reason is None or not self._begin_settle():
                    return
                _log.info("%s: %s %s; collecting it now", self._alias, run_id, reason)
                try:
                    collected = self._settle(run_id=run_id)
                except AppError as refusal:
                    if not unanswered(refusal):
                        raise
                    _log.info(
                        "%s: the queue did not answer the settle of %s; it is settled "
                        "again at the next poll: %s",
                        self._alias,
                        run_id,
                        refusal.message,
                    )
                    return
                finally:
                    self._end_settle()
            _log.info("%s", collected["line"])
            self._collected(run_id, collected["outcome"])
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

    def read(self, run_id: str) -> RunRead:
        """Read one held run's result, if this machine still calls it running.

        The watch calls it under the run's lock in
        :data:`fleet.cli.run_locks.SETTLING`, so the ledger's answer cannot
        change before the read: a run the collect pass settled or stopped
        meanwhile is not read, since the read sends its script into the
        run's directory, the one its retire removes.

        Args:
            run_id: The run.

        Returns:
            What the read found; a node that did not answer is logged and
            read again at the next poll (:func:`fleet.core.collect.attempt_poll_result`).

        Raises:
            AppError: When the node answered and the read failed there, or
                its answer was unreadable.
        """
        if run_id not in live_among(self._loaded, frozenset({run_id})):
            return RunRead.GONE
        polled = collect.attempt_poll_result(self._node, run_id=run_id)
        if polled["unreachable"] is not None:
            _log.info(
                "%s did not answer the read of %s; it is read again at the next poll: %s",
                self._alias,
                run_id,
                polled["unreachable"],
            )
            return RunRead.UNREACHABLE
        return RunRead.RUNNING if polled["result"] is None else RunRead.ENDED


__all__ = [
    "CHECK_WORKERS",
    "THREAD_PREFIX",
    "RunRead",
    "RunWatch",
    "Settle",
    "collect_ended",
    "live_among",
]
