"""CLI: one tick of the hub runner that serves the corvis dispatch queue's
hub lane.

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

THE HUB LANE ONLY (MCPs board task fd5cabfa, A5). This runner claims from
the queue's ``hub`` lane: the ``build-bases`` rebuild and the four session
verbs, every one of them run on the hub itself and closed in the tick that
claimed it. The make targets, which run on a fleet node against a checked-out
commit, are the ``node`` lane, drained by one :mod:`fleet.cli.node_agent`
per enabled node. Until 2026-09-21 this one runner took every kind of job
oldest-first, so a revive queued behind 129 stale kills (board task
4199d1fb) and nine nodes behaved as one slow node; two lanes and per-node
runners are the fix, and nothing here collects a suite any more because
nothing here starts one.

A TICK IS TWO PASSES, IN THIS ORDER:

    1. CLAIM. Take at most one hub-lane job and run it to its close.
    2. OBSERVE. Read every enabled worker's Claude Code session records over
       ssh and hand them to the board's session ledger (MCPs board task
       5a3865bf; :mod:`fleet.core.observe` carries the why). Last, because
       it is the pass that touches every node, and a node asleep costs a
       ten-second ssh timeout that the queue work should not wait behind.
       Runs only when ``--registry`` names the identity registry: the pass
       needs to know which machines exist, and that file lives in the MCPs
       checkout, which a runner may legitimately not have.

At most ONE job per tick, deliberately: a bake or a session verb holds the
hub, and a second claimed before the first closed would run beside it.

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

from board_watch import config as board_config
from platform_core import cli_args
from platform_core.json_utils import JSONObject
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.contracts.dispatch import DispatchJob, encode_job_line
from fleet.core import _test_hooks, observe, published_tree, queue, rebuild, registry, restart

_log = get_logger(__name__)

AGENT_FLAG = "--agent"
SESSION_FLAG = "--session"
ROOT_FLAG = "--repo-root"
NODE_FLAG = "--node"
MCPS_ROOT_FLAG = "--mcps-root"
REGISTRY_FLAG = "--registry"

_FLAGS = (
    _config.CONFIG_FLAG,
    AGENT_FLAG,
    SESSION_FLAG,
    ROOT_FLAG,
    NODE_FLAG,
    MCPS_ROOT_FLAG,
    REGISTRY_FLAG,
)

#: How long a claim survives without a report.
#:
#: An hour: a hub verb closes in the tick that claimed it, so the lease only
#: has to outlive the longest of them (a bake, bounded at 1800 s by its own
#: deadline) and the tick's own ExecutionTimeLimit ends anything past that.
CLAIM_LEASE_SECONDS = 3600


def _refuse_hub_job(
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    detail: str,
) -> DispatchJob:
    """Close a hub-local job as refused, with its named reason.

    Args:
        credentials: The queue's endpoint and headers.
        job: The claimed job being refused.
        identity: This runner's identity arguments.
        detail: The ``CODE: message`` refusal for the queue.

    Returns:
        The job, for the caller to hand back.

    Raises:
        AppError: Only from the queue call itself.
    """
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status="refused",
        exit_code=None,
        detail=detail,
        identity=identity,
    )
    _log.info("refused %s: %s", job["job_id"], detail)
    return job


def rebuild_job(
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    mcps_root: pathlib.Path | None,
) -> DispatchJob:
    """Execute a claimed ``build-bases`` job on the hub, start to close.

    Synchronous, unlike a suite dispatch: the bake is a local make, minutes
    long, and closing it in the same tick leaves no cross-tick state to
    collect (module docstring of :mod:`fleet.core.rebuild`). A runner that
    dies mid-bake leaves the job ``running`` until its lease lapses, after
    which a later tick reclaims it and re-runs the idempotent make.

    Args:
        credentials: The queue's endpoint and headers.
        job: The claimed ``build-bases`` job.
        identity: This runner's identity arguments.
        mcps_root: The MCPs checkout, or None when the runner was started
            without ``--mcps-root``.

    Returns:
        The job, whatever became of it.

    Raises:
        AppError: Only from the queue calls themselves, as
            :func:`claim_pass` describes.
    """
    if mcps_root is None:
        return _refuse_hub_job(
            credentials,
            job,
            identity,
            detail=(
                f"{rebuild.ROOT_MISSING_CODE}: this runner was started "
                f"without {MCPS_ROOT_FLAG}, so it has no MCPs checkout to "
                "rebuild in; resubmit once a rebuild-capable runner is "
                "scheduled"
            ),
        )
    refusal = rebuild.refusal_for(mcps_root, job["submitted_by"])
    if refusal is not None:
        return _refuse_hub_job(credentials, job, identity, detail=refusal)
    run_id = f"bases-{job['job_id']}"
    queue.report_start(
        credentials,
        job_id=job["job_id"],
        node="austinpc",
        run_id=run_id,
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    _log.info("started %s on austinpc as %s (local bake)", job["job_id"], run_id)
    result = rebuild.run_build_bases(mcps_root, submitted_by=job["submitted_by"])
    detail = rebuild.describe_result(result)
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status="passed" if result["returncode"] == 0 else "failed",
        exit_code=result["returncode"],
        detail=detail,
        identity=identity,
    )
    _log.info("%s: %s", encode_job_line(job), detail)
    return job


def restart_job(
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    mcps_root: pathlib.Path | None,
) -> DispatchJob:
    """Execute a claimed session job (restart, revive, or either kill) on the
    hub, start to close.

    The same shape as :func:`rebuild_job` and for the same reasons: the
    work is local, well under a minute, and closing it in one tick leaves
    nothing to collect. The keystroke sequence and every rail around it
    belong to ``session_audit`` (``rollover``, ``revive`` and ``kill``);
    this runner composes one invocation per verb through
    :func:`fleet.core.restart.session_invocation` and reports what it said
    (module docstring of :mod:`fleet.core.restart`).

    Args:
        credentials: The queue's endpoint and headers.
        job: The claimed session job.
        identity: This runner's identity arguments.
        mcps_root: The MCPs checkout, or None when the runner was started
            without ``--mcps-root``.

    Returns:
        The job, whatever became of it.

    Raises:
        AppError: Only from the queue calls themselves, as
            :func:`claim_pass` describes.
    """
    if mcps_root is None:
        return _refuse_hub_job(
            credentials,
            job,
            identity,
            detail=(
                f"{restart.ROOT_MISSING_CODE}: this runner was started "
                f"without {MCPS_ROOT_FLAG}, so it has no session-audit to "
                "invoke; resubmit once a hub runner is scheduled"
            ),
        )
    refusal = restart.refusal_for(mcps_root, job["session_target"])
    if refusal is not None:
        return _refuse_hub_job(credentials, job, identity, detail=refusal)
    # Narrowed by the refusal above: a None target was refused there.
    target = job["session_target"] if job["session_target"] is not None else ""
    # The verb runs the session-audit published on origin/main, never the
    # checkout's working tree (MCPs board task f4cd489f); a tree that cannot
    # be extracted refuses the job with nothing run.
    tree = published_tree.extract_published_tree(mcps_root)
    if isinstance(tree, str):
        return _refuse_hub_job(credentials, job, identity, detail=tree)
    invocation = restart.session_invocation(
        mcps_root, tree, job["command"], target, job["submitted_by"]
    )
    # A revive types the submitter's label into its brief and a kill passes
    # it as an argument (MCPs migs 525 and 526), so the label is judged
    # before it can become an argv element, the same way the target is.
    if invocation["types_requester"]:
        requester_refusal = restart.requester_refusal(job["submitted_by"])
        if requester_refusal is not None:
            return _refuse_hub_job(credentials, job, identity, detail=requester_refusal)
    verb = invocation["verb"]
    run_id = f"{verb}-{job['job_id']}"
    queue.report_start(
        credentials,
        job_id=job["job_id"],
        node="austinpc",
        run_id=run_id,
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    _log.info("started %s on austinpc as %s (local %s)", job["job_id"], run_id, verb)
    result = restart.run_session_job(invocation)
    detail = restart.describe_result(result, invocation)
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status="passed" if result["returncode"] == 0 else "failed",
        exit_code=result["returncode"],
        detail=detail,
        identity=identity,
    )
    _log.info("%s: %s", encode_job_line(job), detail)
    return job


def claim_pass(
    credentials: McpCredentials,
    identity: JSONObject,
    *,
    node: str | None,
    mcps_root: pathlib.Path | None,
) -> DispatchJob | None:
    """Take one hub-lane job and run it to its close.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        node: Restrict claims to this node, or None to take any hub job;
            the hub verbs are pinned to ``austinpc``, so this is the hub's
            own alias or None.
        mcps_root: The MCPs checkout, or None when the runner was started
            without ``--mcps-root``.

    Returns:
        The job that was claimed, whatever became of it, or None when the
        hub lane was empty -- the outcome of most ticks.

    Raises:
        AppError: Only from the queue calls themselves. A LOCAL refusal (no
            MCPs checkout, a target that cannot be resolved) is reported to
            the queue as ``refused`` with its code and message verbatim and
            does not propagate; see the module docstring for why that is
            transport rather than recovery.
    """
    job = queue.claim_next(
        credentials,
        lane="hub",
        tags=(),
        node=node,
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    if job is None:
        return None
    _log.info("claimed %s", encode_job_line(job))
    if job["command"] == "build-bases":
        # The verbs that run on the hub itself, synchronously —
        # fleet.core.rebuild's module docstring carries the why.
        return rebuild_job(credentials, job, identity, mcps_root=mcps_root)
    return restart_job(credentials, job, identity, mcps_root=mcps_root)


def observe_pass(registry_path: pathlib.Path, identity: JSONObject) -> None:
    """Record every enabled worker's Claude Code sessions in the board's ledger.

    The board's credentials are read HERE, not at the top of the tick: they
    are the taskboard's (``TASKBOARD_MCP_API_KEY``), a different secret from
    the queue's, and a runner started without ``--registry`` has no use for
    them and must not be refused for lacking them.

    Args:
        registry_path: The identity registry, ``fleet-mcp/fleet-nodes.json``.
        identity: This runner's identity arguments.

    Raises:
        AppError: ``NODE_REGISTRY_UNREADABLE`` for a registry this cannot
            read, the board-watch credential codes when the taskboard's
            variables are unset, and the contract faults
            :func:`fleet.core.observe.observe_node` names. A node that does
            not answer is logged, not raised.
        OSError: When the registry file cannot be read.
    """
    nodes = registry.decode_registry_nodes(_test_hooks.read_text(registry_path))
    board = board_config.load_credentials()
    for outcome in observe.observe_pass(board, nodes, identity):
        _log.info("%s", observe.render_outcome(outcome))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one tick: claim at most one hub-lane job and run it, then observe.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 whenever the agent itself worked, including when a job was refused
        or a bake failed. See the module docstring.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required one is absent.
        AppError: When the queue cannot be reached or answered a shape this
            runner cannot read, or when the workspace does not decode.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    # The workspace is still read, and refused when malformed, although the
    # hub lane dispatches nothing to a node: a runner whose fleet.json does
    # not decode has no business claiming anything, and the node runners
    # beside it read the same file.
    _config.load_workspace(parsed)
    agent = cli_args.require_flag(parsed, AGENT_FLAG)
    session_id = cli_args.require_flag(parsed, SESSION_FLAG)
    project_root = pathlib.Path(cli_args.require_flag(parsed, ROOT_FLAG)).resolve()
    credentials = queue.load_credentials()
    identity = queue.identity_arguments(agent, session_id, str(project_root))

    raw_mcps_root = parsed.get(MCPS_ROOT_FLAG)
    mcps_root = pathlib.Path(raw_mcps_root).resolve() if raw_mcps_root is not None else None

    if claim_pass(credentials, identity, node=parsed.get(NODE_FLAG), mcps_root=mcps_root) is None:
        _log.info("hub lane empty")

    raw_registry = parsed.get(REGISTRY_FLAG)
    if raw_registry is None:
        # Said out loud, every tick: a ledger with no fleet rows must be
        # distinguishable from a runner that was never asked to write any.
        _log.info("session observation skipped: no %s given", REGISTRY_FLAG)
    else:
        observe_pass(pathlib.Path(raw_registry).resolve(), identity)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
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
    "MCPS_ROOT_FLAG",
    "NODE_FLAG",
    "REGISTRY_FLAG",
    "ROOT_FLAG",
    "SESSION_FLAG",
    "claim_pass",
    "entrypoint",
    "main",
    "observe_pass",
    "rebuild_job",
    "restart_job",
]
