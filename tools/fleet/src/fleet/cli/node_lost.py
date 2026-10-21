"""Stop a run whose queue job left this runner while it ran (board task fd402617).

A node runner renews each running job's claim on every tick. A tick that
cannot reach its node or the queue for longer than the claim's lease lets it
lapse, and the queue hands the job to another runner, which overwrites every
field of the job that named this one: ``claimedBy``, ``claimedAt``, ``node``
and ``runId``. From then on :func:`fleet.core.queue_listing.held_by` does not return
it and it is not cancelled, so neither half of the collect pass looks at the
run again, and its ledger row stays ``running`` for good. Measured
2026-09-30: MCPs/search's job baca2609 was claimed by lavender-wsl at 13:36Z,
last renewed at 13:51Z during that node's memory stall, and reclaimed by
diphtheria at 14:51Z, where it passed at 15:00Z. lavender-wsl's row
MCPs-search-lavender-wsl-1790775427 still read ``running`` at 20:36Z, and
every tick in between refused all work with NODE_OWNER_RESERVED against it.

THE JOB IS FOUND THROUGH ITS TRAIL, the only record of every runner that
ever held it. The row names its project, its submitter and that session, so
the submitter's jobs for the project are listed, and the one that launched
the run is the one whose trail carries a claim by THIS runner inside the
window the run began in (:func:`fleet.core.claim_window.launched_within`,
the rule :func:`fleet.cli.node_collect.launched_by_claim` uses). A run that
no job accounts for, one dispatched by hand with ``fleet-run`` say, is left
alone, as :func:`fleet.cli.node_collect.stop_cancelled` leaves it.

SEVERAL JOBS THAT ALL LEFT IT DO NOT WEDGE THE LANE (MCPs board task
c39ee056). A run that more than one job could have launched was refused
with DISPATCH_CLAIM_AMBIGUOUS rather than guessed at, and the refusal ended
the collect pass, so the runner claimed nothing until someone settled the
run. On 2026-10-18 two serve loops of serendipity's elevated runner ran at
once for 13 minutes after the hub runner's restart, refused each other's
launches, and left run MCPs-execution-elevated-serendipity-1792360026 that
jobs 981406ff, 4cce2d49 and d2571cc9 could each have launched; every pass
from 21:47Z ended there and the lane claimed nothing for two hours. Which
of them launched it decides nothing when none is still held here: whichever
it was has left this runner, so the run is lost, and it is stopped naming
every candidate. A candidate still held here may be the launcher whose start
has not landed, so the run is left alone, as before.
"""

from __future__ import annotations

from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.cli import collect as collect_cli
from fleet.cli.run_locks import SETTLING
from fleet.contracts.dispatch import DispatchJob, DispatchStatus, encode_job_line
from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, LedgerOutcome
from fleet.contracts.node import NodeConfig
from fleet.core import queue, queue_listing, stop
from fleet.core.claim_window import launched_within

_log = get_logger(__name__)


def still_held(job: DispatchJob, *, agent: str) -> bool:
    """Whether this runner holds a job now.

    Args:
        job: The job as the queue holds it.
        agent: This runner's label.

    Returns:
        True when the job is claimed or running under this runner.
    """
    live = job["status"] in (DispatchStatus.CLAIMED, DispatchStatus.RUNNING)
    return live and job["claimed_by"] == agent


def launching_jobs(
    credentials: McpCredentials, row: LedgerEntry, *, agent: str
) -> tuple[DispatchJob, ...]:
    """The queue jobs this runner claimed that could have launched a run.

    Args:
        credentials: The queue's endpoint and headers.
        row: The run's live ledger row.
        agent: This runner's label.

    Returns:
        The job that names the run when this runner holds it, which
        launched it; none when another runner holds the job that names it,
        whose run it is; otherwise every job of the row's submitter that
        carries a claim by this runner inside the window the run began in,
        or, among several such, the one this runner still holds whose start
        it has not reported yet, since the run may be that job's, so it is
        not shown lost. None when no job matches.

    Raises:
        AppError: A queue failure.

    THE QUEUE'S OWN RECORD COMES FIRST (MCPs board task c1d48330). A job
    names the run it launched once its start is reported, and the collect
    pass lists what it holds before it reads the ledger's candidates, so a
    start reported in between left a run nothing named: lavender-wsl's
    collect pass listed job 641605e5 claimed at 03:47:50Z on 2026-10-07, its
    launch reported the start at 03:47:53Z, and the pass took the run for a
    lost one that two of the same session's jobs could have launched, raised
    DISPATCH_CLAIM_AMBIGUOUS and ended the serve. A job of this runner that
    names another run launched that one, so it is no candidate here.

    AND ONLY THE RUNNER HOLDING IT. Two runners serve serendipity, its own
    and its elevated one, and each sees the other's live runs on the node as
    candidates. A naming job taken for the launcher whoever held it made
    the elevated runner stop the other's live run as lost at 04:27:09Z on
    2026-10-07 (job 2042e8ef, MCPs/packages/session-audit). The trail rule
    below already required a claim by this runner; the naming rule now
    requires that this runner holds the job, and leaves the run to the
    runner that does.
    """
    matches: list[DispatchJob] = []
    offset: int | None = 0
    while offset is not None:
        page = queue_listing.submitted_page(
            credentials, project=row["project"], submitted_by=row["agent"], offset=offset
        )
        for job in page["jobs"]:
            if job["session_id"] != row["session_id"]:
                continue
            if job["run_id"] == row["run_id"] and job["claimed_by"] is not None:
                return (job,) if job["claimed_by"] == agent else ()
            if still_held(job, agent=agent) and job["run_id"] != "":
                continue
            claims = queue.trail_claims(credentials, job_id=job["job_id"])
            if any(
                claim["actor"] == agent
                and launched_within(
                    claimed_unix=claim["claimed_unix"], started_unix=row["started_unix"]
                )
                for claim in claims
            ):
                matches.append(job)
        offset = page["next_offset"]
    unstarted = [job for job in matches if still_held(job, agent=agent)]
    if len(matches) > 1 and unstarted:
        return (unstarted[0],)
    return tuple(matches)


def lost_detail(jobs: tuple[DispatchJob, ...], *, agent: str, dispatched_by: str) -> str:
    """What a lost run's row says of the jobs that left this runner.

    Args:
        jobs: The one job that launched it, or every one that could have.
        agent: This runner's label.
        dispatched_by: The row's submitter.

    Returns:
        The detail, naming each job's status and holder.
    """
    left = [
        f"{job['status']} and held by "
        f"{job['claimed_by'] if job['claimed_by'] is not None else 'no runner'}"
        for job in jobs
    ]
    if len(jobs) == 1:
        named = (
            f"queue job {jobs[0]['job_id']} left {agent} while this run was live: it is {left[0]}"
        )
    else:
        named = (
            f"queue jobs {', '.join(job['job_id'] for job in jobs)} could each have launched "
            f"it and every one left {agent} while this run was live: they are "
            f"{', '.join(left)}"
        )
    return f"{named}; stopped by {agent}; was dispatched by {dispatched_by}"


def stop_lost(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    *,
    node: NodeConfig,
    rows: tuple[LedgerEntry, ...],
    agent: str,
) -> int:
    """Stop every run whose launching job has left this runner, and close it lost.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        node: This node's declaration.
        rows: Live rows on this node that no job this runner holds names and
            no job cancelled under it names.
        agent: This runner's label.

    Returns:
        How many runs were stopped.

    Raises:
        AppError: As :func:`launching_jobs` raises, or a node failure from the
            stop, which leaves that run's row live.
    """
    stopped = 0
    for row in rows:
        jobs = launching_jobs(credentials, row, agent=agent)
        if not jobs or any(still_held(job, agent=agent) for job in jobs):
            continue
        # Under the run's lock, and only while the ledger still calls it live,
        # since the serve's watch settles runs beside this pass.
        with SETTLING.holding(row["run_id"]):
            if not collect_cli.live_rows(loaded, run_id=row["run_id"]):
                continue
            stop.stop_and_finish(
                loaded.leases,
                loaded.ledger,
                loaded.feed,
                loaded.retires,
                node=node,
                row=row,
                outcome=LedgerOutcome.LOST,
                exit_code=NO_EXIT_CODE,
                detail=lost_detail(jobs, agent=agent, dispatched_by=row["agent"]),
            )
        for job in jobs:
            _log.info("stopped lost run %s: %s", row["run_id"], encode_job_line(job))
        stopped += 1
    return stopped


__all__ = ["launching_jobs", "lost_detail", "still_held", "stop_lost"]
