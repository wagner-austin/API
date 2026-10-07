"""The collect half of a node runner's tick: close out, stop, or renew.

Split out of :mod:`fleet.cli.node_agent` when that module reached the file
ceiling; the claim half stays there. Each tick asks the queue what this
runner holds and settles every run it launched in one of four ways:

* FINISHED. The node wrote a result: read the transcript's tail, compose the
  verdict (:mod:`fleet.core.verdict`), post it to the submitter's feed when
  the job names no task, and close the job on both sides, the queue's side
  first (:mod:`fleet.cli.node_settle`); the queue's close posts a
  task-naming job's outcome to its thread (MCPs board task 2fecad69).
* STILL RUNNING, INSIDE ITS LEASE. Renew the queue claim, once it was last
  set a minute or more ago (:mod:`fleet.cli.node_collected`), and leave it.
* STILL RUNNING, PAST ITS LEASE (MCPs board task fd5cabfa). Stop it. Until
  this, a renewal had no deadline, so a suite that hung kept its claim and
  its node for as long as it stayed hung: measured 2026-09-22, slime job
  61bd1cbc on sedona logged only idle server metrics for 45 minutes of a
  70-minute run and held the only gpu+windows node with Python while the job
  behind it waited. The deadline is the one the project's lease already
  carries (:func:`fleet.core.collect.lease_deadline`), because past it the
  environment is unprotected and a second dispatch could be admitted into it;
  a run that finished there is refused on collection for exactly that reason,
  and one still going there is stopped for it.
* CANCELLED UNDER IT. Stop it. A cancel marks the queue row terminal and
  cannot reach a node, and :func:`fleet.core.queue.held_by` stops returning
  a cancelled job, so before this nothing ever looked at it again: measured,
  slime-1790104328 was cancelled on the queue at 2026-09-22 20:22Z and closed
  on this machine only at 00:49Z, after its processes had been killed by hand.

* CLAIMED, NEVER STARTED (MCPs board task 5a4f9b3e). A claim tick launches
  a run and only then reports its start, so a queue that refuses that one
  call ends the tick with the job claimed, no run id on it, and a live
  ledger row that nothing names. Measured 2026-09-29: serendipity's tick
  died on report_start's ConnectionRefused at 09:00:58Z while diphtheria's
  services were recreated, and the orphan row held its one run slot until
  10:18Z while every tick logged NODE_OWNER_RESERVED. Collect runs before
  claim in a tick, so a claimed job it sees was left by an earlier tick: it
  ADOPTS the run that claim launched, found by :func:`launched_by_claim`,
  by reporting its start, or REFUSES the claim when nothing was launched.
  A cancel of such a job reaches its run the same way.

A stop ends the build's whole process tree
(:meth:`fleet.core.dialect.Dialect.stop_script`) and then closes the row
(:func:`fleet.core.stop.stop_and_finish`), in that order, so a stop that
fails leaves the row live, matching the node.
"""

from __future__ import annotations

from typing import Final, Protocol, TypedDict

from platform_core.error_codes_fleet import FleetErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config, node_lost
from fleet.cli import collect as collect_cli
from fleet.cli.node_collected import RENEW_AFTER_SECONDS, Collected, CollectOutcome, lease_age
from fleet.cli.node_settle import settle
from fleet.cli.run_locks import SETTLING
from fleet.contracts.dispatch import ClosingStatus, DispatchJob, DispatchStatus, encode_job_line
from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, LedgerOutcome
from fleet.contracts.node import NodeConfig
from fleet.contracts.workspace import require_node, require_project
from fleet.core import _test_hooks, collect, queue, stop
from fleet.core.claim_window import CLAIM_LEASE_SECONDS, launched_within

_log = get_logger(__name__)

#: The exit status a run stopped past its lease is closed with. The build
#: wrote none, and the queue refuses ``failed`` without a non-zero status
#: (MCPs ``requireExitCodeMatchesStatus``), so it takes the status GNU
#: ``timeout`` gives a command it ended for running too long, which is what
#: happened, rather than a number a reader would have to look up.
TIMED_OUT_EXIT_CODE: Final = 124


class RunHolder(Protocol):
    """What the collect pass hands runs to: the serve's watch
    (:class:`fleet.cli.node_watch.RunWatch`)."""

    def hold(self, run_ids: frozenset[str]) -> None:
        """Watch these runs.

        Args:
            run_ids: Runs this runner holds running.
        """

    def owe(self, run_ids: frozenset[str]) -> None:
        """Watch these runs and renew each at its first read that reaches its node.

        Args:
            run_ids: Running runs whose node did not answer this pass's read.
        """


class Reconciled(TypedDict):
    """What reconciling a claim an earlier tick left without a start did.

    Attributes:
        line: One line saying so, for the log.
        adopted: The run adopted, which the collect pass hands to the serve's
            watch, or None when the claim was refused.
    """

    line: str
    adopted: str | None


def collect_one_job(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    board: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
) -> Collected:
    """Close one running job out, stop it past its lease, or renew it when due.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        board: The board's endpoint and headers, for the verdict.
        job: The queue job this runner holds.
        identity: This runner's identity arguments.

    Returns:
        What it did (:class:`fleet.cli.node_collected.CollectOutcome`) and
        the line for the log. A node that did not answer a read before
        anything changed is :attr:`~fleet.cli.node_collected.CollectOutcome.UNREACHABLE`,
        its lease left as it is (MCPs board task 8993c306: lavender-wsl,
        10:30Z); a run still going whose lease was set under
        :data:`fleet.cli.node_collected.RENEW_AFTER_SECONDS` ago writes
        nothing (MCPs board task c1d48330).

    Raises:
        AppError: With a node or workspace code when the node failed a read
            it answered or its declaration has gone, ``LEASE_NOT_HELD`` when
            the build finished after its lease lapsed, or
            ``QUEUE_ANSWER_MALFORMED`` when a node-lane job carries no sha.
            Not caught: those mean this machine's records and the fleet disagree.

    ONE AT A TIME PER RUN, under the run's lock in
    :data:`fleet.cli.run_locks.SETTLING`: the serve's watch settles a run
    that ends while the collect pass is still going
    (:mod:`fleet.cli.node_watch`), and the run each finds live is read
    under the same lock, so the second to reach one finds it closed.
    """
    with SETTLING.holding(job["run_id"]):
        row: LedgerEntry | None = None
        for candidate in collect_cli.live_rows(loaded, run_id=job["run_id"]):
            row = candidate
        if row is None:
            return Collected(
                outcome=CollectOutcome.NOT_LIVE,
                line=f"{encode_job_line(job)}: no live run on this machine, leaving it",
            )
        node = require_node(loaded.workspace, row["node"])
        plan = require_project(loaded.workspace, row["project"])
        polled = collect.attempt_poll_result(node, run_id=row["run_id"])
        if polled["unreachable"] is not None:
            return Collected(
                outcome=CollectOutcome.UNREACHABLE,
                line=f"{encode_job_line(job)}: did not answer the read: {polled['unreachable']}",
            )
        result = polled["result"]
        if result is None:
            deadline = collect.lease_deadline(row, plan)
            now = _test_hooks.now()
            if now > deadline:
                reason = (
                    f"{FleetErrorCode.LEASE_NOT_HELD.value}: still running {now - deadline}s "
                    f"past its lease deadline {deadline}, so the runner ended its process "
                    f"tree; raise {row['project']}'s expected_minutes if the suite needs longer"
                )
                stop.stop_on_node(node, run_id=row["run_id"])
                return settle(
                    loaded,
                    credentials,
                    board,
                    job,
                    identity,
                    row=row,
                    node=node,
                    exit_code=TIMED_OUT_EXIT_CODE,
                    ended_unix=now,
                    detail=reason,
                    stopped=reason,
                )
            age = lease_age(job, now=now)
            if age is not None and age < RENEW_AFTER_SECONDS:
                return Collected(
                    outcome=CollectOutcome.RUNNING,
                    line=f"{encode_job_line(job)}: still running, lease set {age} s ago",
                )
            queue.report_progress(
                credentials,
                job_id=job["job_id"],
                note=f"still running on {row['node']} as {row['run_id']}",
                lease_seconds=CLAIM_LEASE_SECONDS,
                identity=identity,
            )
            return Collected(
                outcome=CollectOutcome.RENEWED,
                line=f"{encode_job_line(job)}: still running, lease renewed",
            )

        finished = result["finished_unix"]
        if collect.outlived_its_lease(row, plan, finished_unix=finished):
            raise collect_cli.lapsed_lease_refusal(row, plan, finished_unix=finished)
        exit_code = result["exit_code"]
        return settle(
            loaded,
            credentials,
            board,
            job,
            identity,
            row=row,
            node=node,
            exit_code=exit_code,
            ended_unix=finished,
            detail=collect.describe(node, run_id=row["run_id"], exit_code=exit_code),
            stopped=None,
        )


def launched_by_claim(
    rows: tuple[LedgerEntry, ...], job: DispatchJob, *, alias: str
) -> LedgerEntry | None:
    """The live run on this node that a claim launched before its start was reported.

    The queue row carries no run id until the start lands, so the run is
    found from what the ledger row does carry: the submitter and session the
    lease was taken for (:func:`fleet.cli.node_launch.launch_claimed` takes
    both from the job), the project, and a start within the claim's lease of
    the claim, since the claiming tick launches seconds after it claims.

    Args:
        rows: Live ledger rows that no running job names.
        job: The claimed, or cancelled, job whose start never landed.
        alias: This node's workspace name.

    Returns:
        The run, or None when the claim launched nothing on this node.

    Raises:
        AppError: With ``DISPATCH_CLAIM_AMBIGUOUS`` when more than one row
            could be the run, which is refused rather than guessed at.
    """
    claimed = job["claimed_unix"]
    matches = [
        row
        for row in rows
        if claimed is not None
        and row["node"] == alias
        and row["project"] == job["project"]
        and row["agent"] == job["submitted_by"]
        and row["session_id"] == job["session_id"]
        and launched_within(claimed_unix=claimed, started_unix=row["started_unix"])
    ]
    if len(matches) > 1:
        raise AppError(
            FleetErrorCode.DISPATCH_CLAIM_AMBIGUOUS,
            f"{encode_job_line(job)} never reported a start, and {len(matches)} live runs on "
            f"{alias} could be the one it launched: {', '.join(row['run_id'] for row in matches)}",
        )
    return matches[0] if matches else None


def reconcile_claim(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    alias: str,
    running: frozenset[str],
) -> Reconciled:
    """Adopt the run a claim launched, or refuse a claim that launched nothing.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        job: A job this runner holds in status claimed, left by an earlier tick.
        identity: This runner's identity arguments.
        alias: This node's workspace name.
        running: The run ids of the running jobs this runner holds, whose
            rows are theirs and not this claim's.

    Returns:
        One line saying what happened, for the log, and the run adopted, if
        one was.

    Raises:
        AppError: ``DISPATCH_CLAIM_AMBIGUOUS`` from :func:`launched_by_claim`,
            or a queue failure. Not caught: the next tick meets the same job.
    """
    rows = tuple(
        row for row in collect_cli.live_rows(loaded, run_id=None) if row["run_id"] not in running
    )
    row = launched_by_claim(rows, job, alias=alias)
    if row is None:
        detail = (
            f"{FleetErrorCode.DISPATCH_NOT_LAUNCHED.value}: {job['job_id']} was claimed on "
            f"{alias}, its start report never reached the queue, and nothing was launched; "
            "refused so the submitter can resubmit it"
        )
        queue.report_close(
            credentials,
            job_id=job["job_id"],
            status=ClosingStatus.REFUSED,
            exit_code=None,
            detail=detail,
            identity=identity,
        )
        return Reconciled(line=f"{encode_job_line(job)}: refused, {detail}", adopted=None)
    queue.report_start(
        credentials,
        job_id=job["job_id"],
        node=alias,
        run_id=row["run_id"],
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    # The start line alone would read like an ordinary start; the trail says
    # why this one came a tick late (MCPs board task 55f2cb0b, A2).
    adopted = (
        f"adopted on {alias}: {row['run_id']} was launched by the claiming tick, whose start "
        "report never reached the queue"
    )
    queue.report_progress(
        credentials,
        job_id=job["job_id"],
        note=adopted,
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    return Reconciled(line=f"{encode_job_line(job)}: {adopted}", adopted=row["run_id"])


def stop_cancelled(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    *,
    agent: str,
    alias: str,
    held: frozenset[str],
) -> int:
    """Stop every run on this node whose queue job was cancelled under it.

    A candidate is a run this machine's ledger still calls running on this
    node that no job this runner holds names. A candidate the queue lists
    as cancelled while THIS runner held it is stopped, and so is one whose
    launching job has since left this runner (:mod:`fleet.cli.node_lost`);
    any other, a ``fleet-run`` dispatched by hand, say, is left alone, because a runner
    that stopped what it could not account for would be the sweep
    ``fleet-cancel``'s header refuses to be. With no candidate the queue is
    not asked at all, which is every ordinary tick.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        agent: This runner's label, which the cancelled jobs still carry.
        alias: This node's workspace name.
        held: The run ids of every job this runner holds, whatever its
            status.

    Returns:
        How many runs were stopped.

    Raises:
        AppError: A queue failure, or a node failure from the stop, which
            leaves that run's row live.
    """
    node = require_node(loaded.workspace, alias)
    candidates = {
        row["run_id"]: row
        for row in collect_cli.live_rows(loaded, run_id=None)
        if row["node"] == alias and row["run_id"] not in held
    }
    stopped = 0
    offset: int | None = 0
    while offset is not None and candidates:
        page = queue.cancelled_page(credentials, agent=agent, offset=offset)
        for job in page["jobs"]:
            # A job cancelled before its start landed names no run; its run
            # is found as a claim's is (:func:`launched_by_claim`).
            launched = (
                None
                if job["run_id"]
                else launched_by_claim(tuple(candidates.values()), job, alias=alias)
            )
            row = candidates.pop(job["run_id"] if launched is None else launched["run_id"], None)
            if row is None:
                continue
            if stop_cancelled_run(loaded, node=node, row=row, job=job, agent=agent):
                stopped += 1
        offset = page["next_offset"]
    # What no cancel accounts for may be a run whose job was taken over by
    # another runner (board task fd402617); what nothing accounts for stays.
    lost = node_lost.stop_lost(
        loaded, credentials, node=node, rows=tuple(candidates.values()), agent=agent
    )
    return stopped + lost


def stop_cancelled_run(
    loaded: _config.LoadedWorkspace,
    *,
    node: NodeConfig,
    row: LedgerEntry,
    job: DispatchJob,
    agent: str,
) -> bool:
    """Stop one run whose queue job was cancelled, and close its row cancelled.

    The one stop for a cancel, whichever tick finds it: a later tick's
    :func:`stop_cancelled`, or the claiming tick itself when the cancel landed
    between the launch and the start report
    (:func:`fleet.cli.node_start.report_started`).

    Args:
        loaded: The workspace and its resolved record paths.
        node: The node the run is on.
        row: The run's live ledger row.
        job: The cancelled queue job.
        agent: This runner's label.

    Returns:
        True when it stopped the run; False when the run was no longer live
        once its lock was held, settled meanwhile by the serve's watch.

    Raises:
        AppError: A node failure from the stop, which leaves the row live.
    """
    with SETTLING.holding(row["run_id"]):
        if not collect_cli.live_rows(loaded, run_id=row["run_id"]):
            return False
        stop.stop_and_finish(
            loaded.leases,
            loaded.ledger,
            loaded.feed,
            node=node,
            row=row,
            outcome=LedgerOutcome.CANCELLED,
            exit_code=NO_EXIT_CODE,
            detail=(
                f"queue job {job['job_id']} was cancelled while it ran; stopped by "
                f"{agent}; was dispatched by {row['agent']}"
            ),
        )
    _log.info("stopped %s: %s", row["run_id"], encode_job_line(job))
    return True


def collect_pass(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    board: McpCredentials,
    identity: JSONObject,
    *,
    agent: str,
    alias: str,
    holder: RunHolder,
    launching: frozenset[str],
) -> None:
    """Settle every running job this runner holds, reconcile every claim an
    earlier tick left without a start, then stop what was cancelled.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        board: The board's endpoint and headers.
        identity: This runner's identity arguments.
        agent: This runner's label.
        alias: This node's workspace name.
        holder: The serve's watch (:mod:`fleet.cli.node_watch`), given the
            run ids of the running jobs this runner holds as soon as the
            queue has said, before any is settled, so it reads them while
            this pass goes on; each run this pass adopts; and each run whose
            node did not answer this pass's read, whose renewal it then owes.
        launching: The jobs this serve's launches still carry
            (:mod:`fleet.cli.node_launch`), claimed with no start yet by
            design, which are left to them rather than reconciled.

    Raises:
        AppError: As :func:`collect_one_job`, :func:`reconcile_claim` and
            :func:`stop_cancelled` describe.
    """
    held = queue.held_by(credentials, agent=agent)
    running = frozenset(job["run_id"] for job in held if job["status"] is DispatchStatus.RUNNING)
    holder.hold(running)
    accounted = {job["run_id"] for job in held}
    for job in held:
        # Live is claimed or running; a claimed one no launch carries is an earlier tick's.
        if job["status"] is DispatchStatus.RUNNING:
            collected = collect_one_job(loaded, credentials, board, job, identity)
            _log.info("%s", collected["line"])
            if collected["outcome"] is CollectOutcome.UNREACHABLE:
                holder.owe(frozenset({job["run_id"]}))
        elif job["job_id"] in launching:
            _log.info("%s: its launch is under way", encode_job_line(job))
        else:
            reconciled = reconcile_claim(
                loaded, credentials, job, identity, alias=alias, running=running
            )
            _log.info("%s", reconciled["line"])
            adopted = reconciled["adopted"]
            if adopted is not None:
                # Watched from now, as a run launched by this serve is
                # (row 220c2a9e on serendipity, 2026-10-07, MCPs board
                # task c1d48330), and its own, not a lost one, below.
                holder.hold(frozenset({adopted}))
                accounted.add(adopted)
    stop_cancelled(loaded, credentials, agent=agent, alias=alias, held=frozenset(accounted))


__all__ = [
    "CLAIM_LEASE_SECONDS",
    "SETTLING",
    "TIMED_OUT_EXIT_CODE",
    "Reconciled",
    "RunHolder",
    "collect_one_job",
    "collect_pass",
    "stop_cancelled",
    "stop_cancelled_run",
]
