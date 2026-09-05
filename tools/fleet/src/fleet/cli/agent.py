"""CLI: one tick of the node runner that serves the corvis dispatch queue.

Usage:
    fleet-agent --config fleet.json --repo-root C:/Users/Test/PROJECTS/API \\
        --agent fleet-runner-austinpc --session <uuid>

THIS IS THE OTHER HALF OF QUEUE INVERSION. ``fleet-mcp``'s ``dispatch_*``
tools hold a queue that any session can enqueue onto -- from a phone, from
claude.ai, from another machine -- and the corvis server has no route to the
tailnet and no ssh key. So the work travels the other way: this command runs
on the hub, where the keys are, claims a job and executes it. No inbound route
to the fleet is ever opened, and no fleet credential ever lives on an
internet-facing multi-tenant server.

ONE TICK, NO LOOP. The interval belongs to whatever schedules this -- a shell
loop, Task Scheduler, a Monitor -- where it is visible and changeable without
editing code. Same decision ``tools/board-watch`` made next door.

A TICK IS TWO PASSES, IN THIS ORDER:

    1. COLLECT. For every job this runner holds that is already running, ask
       the node whether the suite has finished, and if it has, close the job
       out on both sides -- the local ledger AND the queue. Collection runs
       FIRST so a tick that also claims new work never leaves a finished
       result unreported for a whole interval.
    2. CLAIM. Take at most one new job, choose a node with capacity, stage,
       launch, and report it started.

At most ONE new job per tick, deliberately. Several can be in flight across
ticks; what a single tick must not do is claim a second job before the first
has been recorded, because the record is what the capacity check for the
second one reads.

WHY THIS COMMAND EXITS 0 FOR A REFUSED JOB, and it is the same argument
``fleet-collect`` makes for a failing suite: the status of this command is
whether THE AGENT worked. A job that no node had capacity for was handled
correctly -- the refusal is reported to the queue with its code and message
verbatim, where the submitter reads it. Exiting non-zero would stop the
scheduling loop on exactly the condition the loop exists to keep reporting.
That is transport, not recovery: nothing is softened, retried, or
best-efforted, and the one place an exception becomes a queue state is here,
at the boundary, rather than scattered through the engine.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger, setup_logging
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.cli import collect as collect_cli
from fleet.cli import run as run_cli
from fleet.contracts.dispatch import DispatchJob, encode_job_line
from fleet.contracts.ledger import LedgerEntry
from fleet.contracts.workspace import require_node, require_project
from fleet.core import collect, dispatch, queue

_log = get_logger(__name__)

AGENT_FLAG = "--agent"
SESSION_FLAG = "--session"
ROOT_FLAG = "--repo-root"
NODE_FLAG = "--node"

_FLAGS = (_config.CONFIG_FLAG, AGENT_FLAG, SESSION_FLAG, ROOT_FLAG, NODE_FLAG)

#: How long a claim survives without a report.
#:
#: An hour, not the project's expected minutes: a claim covers staging, the
#: whole suite, and the wait until the NEXT tick collects it, so a lease sized
#: to the build alone would hand the job to a second runner while the first
#: was still holding the node's lease for it.
CLAIM_LEASE_SECONDS = 3600


def ledger_row_for(loaded: _config.LoadedWorkspace, *, run_id: str) -> LedgerEntry | None:
    """Find this machine's live ledger row for a queue job's run.

    Args:
        loaded: The workspace and its resolved record paths.
        run_id: The run id the queue job carries.

    Returns:
        The row, or None when this machine has no live record of that run --
        which is what a queue job claimed by a runner on a DIFFERENT machine
        looks like from here, and is a reason to leave it alone rather than
        an error.
    """
    for row in collect_cli.live_rows(loaded, run_id=run_id):
        return row
    return None


def collect_one_job(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
) -> str:
    """Close one running queue job out, on both sides, if its node has finished.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        job: The queue job this runner holds.
        identity: This runner's identity arguments.

    Returns:
        One line saying what happened, for the log.

    Raises:
        AppError: With a node or workspace code when the node cannot be
            reached or its declaration has gone, or ``LEASE_NOT_HELD`` when
            the build was still writing after its lease lapsed. Not caught:
            those mean this machine's own records and the fleet disagree, and
            reporting a tidy outcome over that would destroy the evidence.
    """
    row = ledger_row_for(loaded, run_id=job["run_id"])
    if row is None:
        return f"{encode_job_line(job)}: no live run on this machine, leaving it"
    node = require_node(loaded.workspace, row["node"])
    result = collect.poll_result(node, run_id=row["run_id"])
    if result is None:
        return f"{encode_job_line(job)}: still running"

    plan = require_project(loaded.workspace, row["project"])
    if collect.outlived_its_lease(row, plan, finished_unix=result["finished_unix"]):
        raise collect_cli.lapsed_lease_refusal(row, plan, finished_unix=result["finished_unix"])

    exit_code = result["exit_code"]
    detail = collect.describe(node, run_id=row["run_id"], exit_code=exit_code)
    outcome = collect.outcome_for(exit_code)
    dispatch.finish(
        loaded.leases,
        loaded.ledger,
        loaded.feed,
        row=row,
        outcome=outcome,
        exit_code=exit_code,
        detail=detail,
    )
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status="passed" if exit_code == 0 else "failed",
        exit_code=exit_code,
        detail=detail,
        identity=identity,
    )
    return f"{encode_job_line(job)}: {outcome} -- {detail}"


def collect_pass(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    identity: JSONObject,
    *,
    agent: str,
) -> None:
    """Close out every finished job this runner is holding.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        agent: This runner's label.

    Raises:
        AppError: As :func:`collect_one_job` describes.
    """
    for job in queue.held_by(credentials, agent=agent):
        if job["status"] != "running":
            continue
        _log.info("%s", collect_one_job(loaded, credentials, job, identity))


def claim_pass(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    identity: JSONObject,
    *,
    node: str | None,
    project_root: pathlib.Path,
) -> DispatchJob | None:
    """Take one job and launch it, or report why it could not run.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        node: Restrict claims to this node, or None to serve the whole fleet.
        project_root: Absolute path to the monorepo root on this machine.

    Returns:
        The job that was claimed, whatever became of it, or None when the
        queue was empty -- the outcome of most ticks.

    Raises:
        AppError: Only from the queue calls themselves. A LOCAL refusal
            (no capacity, unknown project, a held resource, an unreachable
            node) is reported to the queue as ``refused`` with its code and
            message verbatim and does not propagate; see the module docstring
            for why that is transport rather than recovery.
    """
    job = queue.claim_next(
        credentials,
        node=node,
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    if job is None:
        return None
    _log.info("claimed %s", encode_job_line(job))
    try:
        chosen, declaration, workers = run_cli.choose(
            loaded, project=job["project"], named=job["requested_node"]
        )
    except AppError as refusal:
        queue.report_close(
            credentials,
            job_id=job["job_id"],
            status="refused",
            exit_code=None,
            detail=f"{refusal.code}: {refusal.message}",
            identity=identity,
        )
        _log.info("refused %s: %s", job["job_id"], refusal.message)
        return job

    row = dispatch.start(
        loaded.leases,
        loaded.ledger,
        loaded.feed,
        node_name=chosen,
        node=declaration,
        project=job["project"],
        plan=require_project(loaded.workspace, job["project"]),
        workers=workers,
        agent=job["submitted_by"],
        session_id=job["session_id"],
        project_root=project_root,
        archive_dir=loaded.archives,
    )
    queue.report_start(
        credentials,
        job_id=job["job_id"],
        node=chosen,
        run_id=row["run_id"],
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    _log.info("started %s on %s as %s", job["job_id"], chosen, row["run_id"])
    return job


def main(argv: Sequence[str] | None = None) -> int:
    """Run one tick: collect what has finished, then claim at most one job.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 whenever the agent itself worked, including when a job was refused
        or a suite failed. See the module docstring.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required one is absent.
        AppError: When the queue cannot be reached or answered a shape this
            runner cannot read, or when this machine's records and the fleet
            disagree about a run.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    loaded = _config.load_workspace(parsed)
    agent = cli_args.require_flag(parsed, AGENT_FLAG)
    session_id = cli_args.require_flag(parsed, SESSION_FLAG)
    project_root = pathlib.Path(cli_args.require_flag(parsed, ROOT_FLAG)).resolve()
    credentials = queue.load_credentials()
    identity = queue.identity_arguments(agent, session_id, str(project_root))

    collect_pass(loaded, credentials, identity, agent=agent)
    if (
        claim_pass(
            loaded,
            credentials,
            identity,
            node=parsed.get(NODE_FLAG),
            project_root=project_root,
        )
        is None
    ):
        _log.info("queue empty")
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="fleet-agent",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


# Without this, `python -m fleet.cli.agent` imports the module, runs nothing
# and exits 0 -- which reads as a tick that found an empty queue, on a runner
# that in fact never asked.
if __name__ == "__main__":
    entrypoint()


__all__ = [
    "AGENT_FLAG",
    "CLAIM_LEASE_SECONDS",
    "NODE_FLAG",
    "ROOT_FLAG",
    "SESSION_FLAG",
    "claim_pass",
    "collect_one_job",
    "collect_pass",
    "entrypoint",
    "ledger_row_for",
    "main",
]
