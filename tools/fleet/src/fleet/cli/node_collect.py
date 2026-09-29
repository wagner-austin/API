"""The collect half of a node runner's tick: close out, stop, or renew.

Split out of :mod:`fleet.cli.node_agent` when that module reached the file
ceiling; the claim half stays there. Each tick asks the queue what this
runner holds and settles every run it launched in one of four ways:

* FINISHED. The node wrote a result: read the transcript's tail, compose the
  verdict (:mod:`fleet.core.verdict`), post it to the job's task thread or the
  submitter's feed, and close the job on both sides.
* STILL RUNNING, INSIDE ITS LEASE. Renew the queue claim and leave it.
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

from typing import Final

from platform_core.error_codes_tooling import FleetErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.cli import collect as collect_cli
from fleet.contracts.dispatch import ClosingStatus, DispatchJob, DispatchStatus, encode_job_line
from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, LedgerOutcome
from fleet.contracts.node import NodeConfig
from fleet.contracts.workspace import require_node, require_project
from fleet.core import (
    _test_hooks,
    collect,
    dialect,
    dispatch,
    names,
    queue,
    remote,
    retire,
    stop,
    verdict,
)

_log = get_logger(__name__)

#: How long a claim survives without a report: the same hour the hub runner
#: takes, covering the fetch, the staging and the wait until the next tick's
#: collect renews it (:func:`collect_pass` renews every running job it holds,
#: so the lease is never sized for the slowest suite in advance).
CLAIM_LEASE_SECONDS: Final = 3600

#: The exit status a run stopped past its lease is closed with. The build
#: wrote none, and the queue refuses ``failed`` without a non-zero status
#: (MCPs ``requireExitCodeMatchesStatus``), so it takes the status GNU
#: ``timeout`` gives a command it ended for running too long, which is what
#: happened, rather than a number a reader would have to look up.
TIMED_OUT_EXIT_CODE: Final = 124

#: How far the queue's clock and this machine's may disagree when a claim is
#: matched to the run it launched: the claim is stamped by the database host
#: and the ledger row by the hub.
CLAIM_CLOCK_SLACK_SECONDS: Final = 60


def settle(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    board: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    row: LedgerEntry,
    node: NodeConfig,
    exit_code: int,
    detail: str,
    stopped: str | None,
) -> str:
    """Post a run's verdict, close its row, then close its queue job.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        board: The board's endpoint and headers, for the verdict.
        job: The queue job this runner holds.
        identity: This runner's identity arguments.
        row: The run's live ledger row.
        node: The node it ran on.
        exit_code: The status to record.
        detail: What the ledger and the feed say about it.
        stopped: Why the runner stopped the build, appended to the verdict
            line, or None when the build finished on its own.

    Returns:
        The verdict line, as posted and as the queue job's detail.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when the job carries no sha, or
            a node, board or queue failure. Not caught.
    """
    sha = require_sha(job)
    target = names.dispatch_directory(node["stage_root"], row["run_id"])
    spoken = dialect.for_platform(node["platform"])
    tail = remote.run_script(
        node["host"],
        spoken.script_path(target, names.LOG_TAIL_STEM),
        spoken.log_tail_script(target, verdict.LOG_TAIL_LINES),
        platform=node["platform"],
    )
    # Retired once its tail is read and before anything is posted, so a
    # retire that fails leaves the job held and the row live for the next
    # tick, and the verdict names the transcript where it now stays.
    retained = retire.retire_on_node(node, run_id=row["run_id"])
    judged = verdict.judge(
        job_id=job["job_id"],
        project=row["project"],
        sha=sha,
        node=row["node"],
        exit_code=exit_code,
        tail=tail,
        log_path=retained,
        run_id=row["run_id"],
    )
    rendered = verdict.render_verdict(judged)
    line = rendered if stopped is None else f"{rendered} stopped: {stopped}"
    queue.post_verdict(
        board,
        task_id=job["task_id"],
        submitted_by=job["submitted_by"],
        line=line,
        identity=identity,
    )
    dispatch.finish(
        loaded.leases,
        loaded.ledger,
        loaded.feed,
        row=row,
        outcome=collect.outcome_for(exit_code),
        exit_code=exit_code,
        detail=detail,
    )
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status=ClosingStatus.PASSED if exit_code == 0 else ClosingStatus.FAILED,
        exit_code=exit_code,
        detail=line,
        identity=identity,
    )
    return line


def collect_one_job(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    board: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
) -> str:
    """Close one running job out, stop it past its lease, or renew it.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        board: The board's endpoint and headers, for the verdict.
        job: The queue job this runner holds.
        identity: This runner's identity arguments.

    Returns:
        One line saying what happened, for the log.

    Raises:
        AppError: With a node or workspace code when the node cannot be
            reached or its declaration has gone, ``LEASE_NOT_HELD`` when the
            build finished after its lease lapsed, or
            ``QUEUE_ANSWER_MALFORMED`` when a node-lane job carries no sha,
            which the queue's pin makes impossible. Not caught: those mean
            this machine's own records and the fleet disagree.
    """
    row: LedgerEntry | None = None
    for candidate in collect_cli.live_rows(loaded, run_id=job["run_id"]):
        row = candidate
    if row is None:
        return f"{encode_job_line(job)}: no live run on this machine, leaving it"
    node = require_node(loaded.workspace, row["node"])
    plan = require_project(loaded.workspace, row["project"])
    result = collect.poll_result(node, run_id=row["run_id"])
    if result is None:
        deadline = collect.lease_deadline(row, plan)
        now = _test_hooks.now()
        if now > deadline:
            reason = (
                f"{FleetErrorCode.LEASE_NOT_HELD.value}: still running {now - deadline}s past "
                f"its lease deadline {deadline}, so the runner ended its process tree; raise "
                f"{row['project']}'s expected_minutes if the suite needs longer"
            )
            stop.stop_on_node(node, run_id=row["run_id"])
            line = settle(
                loaded,
                credentials,
                board,
                job,
                identity,
                row=row,
                node=node,
                exit_code=TIMED_OUT_EXIT_CODE,
                detail=reason,
                stopped=reason,
            )
            return f"{encode_job_line(job)}: {line}"
        queue.report_progress(
            credentials,
            job_id=job["job_id"],
            note=f"still running on {row['node']} as {row['run_id']}",
            lease_seconds=CLAIM_LEASE_SECONDS,
            identity=identity,
        )
        return f"{encode_job_line(job)}: still running, lease renewed"

    if collect.outlived_its_lease(row, plan, finished_unix=result["finished_unix"]):
        raise collect_cli.lapsed_lease_refusal(row, plan, finished_unix=result["finished_unix"])
    exit_code = result["exit_code"]
    line = settle(
        loaded,
        credentials,
        board,
        job,
        identity,
        row=row,
        node=node,
        exit_code=exit_code,
        detail=collect.describe(node, run_id=row["run_id"], exit_code=exit_code),
        stopped=None,
    )
    return f"{encode_job_line(job)}: {line}"


def launched_by_claim(
    rows: tuple[LedgerEntry, ...], job: DispatchJob, *, alias: str
) -> LedgerEntry | None:
    """The live run on this node that a claim launched before its start was reported.

    The queue row carries no run id until the start lands, so the run is
    found from what the ledger row does carry: the submitter and session the
    lease was taken for (:func:`fleet.cli.node_agent.launch_claimed` takes
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
        and claimed - CLAIM_CLOCK_SLACK_SECONDS
        <= row["started_unix"]
        <= claimed + CLAIM_LEASE_SECONDS
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
) -> str:
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
        One line saying what happened, for the log.

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
        return f"{encode_job_line(job)}: refused, {detail}"
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
    return f"{encode_job_line(job)}: {adopted}"


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
    node that no job this runner holds names. Only a candidate the queue
    lists as cancelled while THIS runner held it is stopped; any other, a
    ``fleet-run`` dispatched by hand, say, is left alone, because a runner
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
            stopped += 1
        offset = page["next_offset"]
    return stopped


def collect_pass(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    board: McpCredentials,
    identity: JSONObject,
    *,
    agent: str,
    alias: str,
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

    Raises:
        AppError: As :func:`collect_one_job`, :func:`reconcile_claim` and
            :func:`stop_cancelled` describe.
    """
    held = queue.held_by(credentials, agent=agent)
    running = frozenset(job["run_id"] for job in held if job["status"] is DispatchStatus.RUNNING)
    for job in held:
        # Live is claimed or running; a claimed one is an earlier tick's.
        if job["status"] is DispatchStatus.RUNNING:
            _log.info("%s", collect_one_job(loaded, credentials, board, job, identity))
        else:
            line = reconcile_claim(loaded, credentials, job, identity, alias=alias, running=running)
            _log.info("%s", line)
    stop_cancelled(
        loaded,
        credentials,
        agent=agent,
        alias=alias,
        held=frozenset(job["run_id"] for job in held),
    )


def require_sha(job: DispatchJob) -> str:
    """The commit a node-lane job names.

    Args:
        job: The claimed job.

    Returns:
        Its sha.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when the job carries none,
            which the queue's pin (MCPs mig 532) makes impossible for a
            node-lane row; a null here means the contract moved.
    """
    sha = job["sha"]
    if sha is None:
        raise AppError(
            code=FleetErrorCode.QUEUE_ANSWER_MALFORMED,
            message=f"node-lane job {job['job_id']} carries no sha; the queue's pin forbids it",
        )
    return sha


__all__ = [
    "CLAIM_LEASE_SECONDS",
    "TIMED_OUT_EXIT_CODE",
    "collect_one_job",
    "collect_pass",
    "require_sha",
    "settle",
    "stop_cancelled",
]
