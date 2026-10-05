"""Stop a run whose queue job left this runner while it ran (board task fd402617).

A node runner renews each running job's claim on every tick. A tick that
cannot reach its node or the queue for longer than the claim's lease lets it
lapse, and the queue hands the job to another runner, which overwrites every
field of the job that named this one: ``claimedBy``, ``claimedAt``, ``node``
and ``runId``. From then on :func:`fleet.core.queue.held_by` does not return
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
"""

from __future__ import annotations

from platform_core.error_codes_fleet import FleetErrorCode
from platform_core.errors import AppError
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.cli import collect as collect_cli
from fleet.cli.run_locks import SETTLING
from fleet.contracts.dispatch import DispatchJob, DispatchStatus, encode_job_line
from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, LedgerOutcome
from fleet.contracts.node import NodeConfig
from fleet.core import queue, stop
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


def launching_job(
    credentials: McpCredentials, row: LedgerEntry, *, agent: str
) -> DispatchJob | None:
    """The queue job this runner claimed and launched a run for.

    Args:
        credentials: The queue's endpoint and headers.
        row: The run's live ledger row.
        agent: This runner's label.

    Returns:
        The job, or None when no job of the row's submitter carries a claim by
        this runner inside the window the run began in.

    Raises:
        AppError: ``DISPATCH_CLAIM_AMBIGUOUS`` when more than one job could
            have launched it, which is refused rather than guessed at; or a
            queue failure.
    """
    matches: list[DispatchJob] = []
    offset: int | None = 0
    while offset is not None:
        page = queue.submitted_page(
            credentials, project=row["project"], submitted_by=row["agent"], offset=offset
        )
        for job in page["jobs"]:
            if job["session_id"] != row["session_id"]:
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
    if len(matches) > 1:
        raise AppError(
            FleetErrorCode.DISPATCH_CLAIM_AMBIGUOUS,
            f"{row['run_id']} has no job held by {agent}, and {len(matches)} jobs could have "
            f"launched it: {', '.join(job['job_id'] for job in matches)}",
        )
    return matches[0] if matches else None


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
        AppError: As :func:`launching_job` raises, or a node failure from the
            stop, which leaves that run's row live.
    """
    stopped = 0
    for row in rows:
        job = launching_job(credentials, row, agent=agent)
        if job is None or still_held(job, agent=agent):
            continue
        holder = job["claimed_by"] if job["claimed_by"] is not None else "no runner"
        # Under the run's lock, and only while the ledger still calls it live,
        # since the serve's watch settles runs beside this pass.
        with SETTLING.holding(row["run_id"]):
            if not collect_cli.live_rows(loaded, run_id=row["run_id"]):
                continue
            stop.stop_and_finish(
                loaded.leases,
                loaded.ledger,
                loaded.feed,
                node=node,
                row=row,
                outcome=LedgerOutcome.LOST,
                exit_code=NO_EXIT_CODE,
                detail=(
                    f"queue job {job['job_id']} left {agent} while this run was live: it is "
                    f"{job['status']} and held by {holder}; stopped by {agent}; was dispatched "
                    f"by {row['agent']}"
                ),
            )
        _log.info("stopped lost run %s: %s", row["run_id"], encode_job_line(job))
        stopped += 1
    return stopped


__all__ = ["launching_job", "still_held", "stop_lost"]
