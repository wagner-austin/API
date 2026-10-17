"""CLI: one tick of the runner that serves the corvis dispatch queue's hub
lane or its sessions lane.

Usage:
    fleet-agent --config registry --repo-root C:/Users/Test/PROJECTS/API \\
        --agent fleet-runner-austinpc --session <uuid> --lane hub [--announce]
    fleet-agent --config registry --repo-root C:/Users/Test/PROJECTS/API \\
        --agent fleet-sessions-austinpc --session <uuid> --lane sessions \\
        --node austinpc [--announce]

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

THE HUB AND SESSIONS LANES ONLY (MCPs board tasks fd5cabfa, A5, and
c39ee056). With ``--lane hub`` this runner claims the ``build-bases``
rebuild, the control duty's verb; with ``--lane sessions`` it claims the
session verbs (restart, revive, the two kills, the compact and the exit),
the sessions duty's, on the machine whose sessions they are. Every one runs
where it was claimed and closes in the tick that claimed it. The MCPs hub
runner places each lane's row by the duty the node registry gives its
machine, so the queue names no node for a session verb: holding the duty is
the placement. The make targets, which run on a fleet node against a
checked-out commit, are the ``node`` lane, drained by one
:mod:`fleet.cli.node_agent` per enabled node. Until 2026-09-21 one runner
took every kind of job oldest-first, so a revive queued behind 129 stale
kills (board task 4199d1fb) and nine nodes behaved as one slow node; lanes
and per-node runners are the fix, and nothing here collects a suite any more
because nothing here starts one.

A HUB TICK IS FOUR PASSES, IN THIS ORDER; a sessions tick is the first
alone, since publishing, observing and reaping are the control duty's:

    1. CLAIM. Take at most one job of the lane and run it to its close.
    2. PUBLISH. Read every project the registry declares at the commit
       ``refs/fleet/rolled`` names now, with the tags each requires, and
       hand them to the queue (MCPs board tasks 1915b042 and 0be88a36,
       :mod:`fleet.core.declared`). The queue keeps them, so a submitter
       with no API checkout can have it supply a project's tags, and in the
       same transaction refuses each queued make-target job queued before
       the read whose project is not among them: no node runner would ever
       claim such a job, since each claims only the registry projects it
       fits. Here, once a tick for the whole fleet, rather than in every
       node runner.
    3. OBSERVE. Read every enabled worker's Claude Code session records over
       ssh and hand them to the board's session ledger (MCPs board task
       5a3865bf; :mod:`fleet.core.observe` carries the why). Last, because
       it is the pass that touches every node, and a node asleep costs a
       ten-second ssh timeout that the queue work should not wait behind.
       Runs only when ``--registry`` names the identity registry: the pass
       needs to know which machines exist, and that file lives in the MCPs
       checkout, which a runner may legitimately not have.
    4. REAP. Read each WSL runner host's Workers against GitHub's runner
       list, read again two minutes later where one was named, and end each
       Worker both readings name as holding a job GitHub dropped, by pid,
       through the host's reaper (MCPs board task 715b9ec5,
       :mod:`fleet.core.runner_reap_dropped`). Last, because its second
       reading waits, and nothing above should wait behind it.

At most ONE job per tick, deliberately: a bake or a session verb holds its
machine, and a second claimed before the first closed would run beside it.

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

from fleet.cli import _config, hub_passes
from fleet.contracts.dispatch import (
    ClosingStatus,
    DispatchCommand,
    DispatchJob,
    DispatchLane,
    encode_job_line,
)
from fleet.core import (
    _test_hooks,
    queue,
    rebuild,
    restart,
    session_release,
)

_log = get_logger(__name__)

AGENT_FLAG = "--agent"
SESSION_FLAG = "--session"
ROOT_FLAG = "--repo-root"
NODE_FLAG = "--node"
MCPS_ROOT_FLAG = "--mcps-root"
REGISTRY_FLAG = "--registry"
LANE_FLAG = "--lane"

_FLAGS = (
    _config.CONFIG_FLAG,
    _config.RECORDS_FLAG,
    AGENT_FLAG,
    SESSION_FLAG,
    ROOT_FLAG,
    NODE_FLAG,
    MCPS_ROOT_FLAG,
    REGISTRY_FLAG,
    LANE_FLAG,
)

#: The flags only a hub tick's publish and observe passes read, which a
#: sessions tick refuses rather than ignores.
_HUB_ONLY_FLAGS = (MCPS_ROOT_FLAG, REGISTRY_FLAG)

#: The node a hub-lane job is reported on: ``build-bases`` is pinned there
#: (MCPs ``FLEET_DISPATCH_HUB_NODE``, migs 497 and 795).
HUB_NODE = "austinpc"

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
        status=ClosingStatus.REFUSED,
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
        node=HUB_NODE,
        run_id=run_id,
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    _log.info("started %s on %s as %s (local bake)", job["job_id"], HUB_NODE, run_id)
    result = rebuild.run_build_bases(mcps_root, submitted_by=job["submitted_by"])
    detail = rebuild.describe_result(result)
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status=ClosingStatus.PASSED if result["returncode"] == 0 else ClosingStatus.FAILED,
        exit_code=result["returncode"],
        detail=detail,
        identity=identity,
    )
    _log.info("%s: %s", encode_job_line(job), detail)
    return job


def session_job(
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    node: str,
) -> DispatchJob:
    """Execute a claimed session job (restart, revive, kill, compact or exit)
    on this machine, start to close.

    The same shape as :func:`rebuild_job` and for the same reasons: the
    work is local, well under a minute, and closing it in one tick leaves
    nothing to collect. The keystroke sequence and every rail around it
    belong to ``session_audit`` (``rollover``, ``revive``, ``kill`` and
    ``compact``); this runner composes one invocation per verb through
    :func:`fleet.core.restart.session_invocation` and reports what it said
    (module docstring of :mod:`fleet.core.restart`).

    Named for the job kind rather than one of its five verbs. It was
    ``restart_job``, which named one verb and claimed a state change this
    runner does not make: the change, and the check of each step before
    the next (the old process gone before a resume is typed, a new record
    on the installed build before a restart is reported), are the sealed
    ``session_audit`` release's, whose exit status this closes the job
    with (``state-change-verified``, board task cc7222ca).

    Args:
        credentials: The queue's endpoint and headers.
        job: The claimed session job.
        identity: This runner's identity arguments.
        node: This machine's alias, which the job is reported started on.

    Returns:
        The job, whatever became of it.

    Raises:
        AppError: Only from the queue calls themselves, as
            :func:`claim_pass` describes.
    """
    refusal = restart.refusal_for(job["session_target"])
    if refusal is not None:
        return _refuse_hub_job(credentials, job, identity, detail=refusal)
    # Narrowed by the refusal above: a None target was refused there.
    target = job["session_target"] if job["session_target"] is not None else ""
    # The verb runs the active sealed session-verbs release, verified for
    # this job, and nothing else (MCPs board task c7c2527d); no release, or
    # one that fails its seal, refuses the job with nothing run.
    release = session_release.active_session_release()
    if isinstance(release, str):
        return _refuse_hub_job(credentials, job, identity, detail=release)
    invocation = restart.session_invocation(
        release, job["command"], target, job["submitted_by"], node
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
        node=node,
        run_id=run_id,
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    _log.info("started %s on %s as %s (local %s)", job["job_id"], node, run_id, verb)
    result = restart.run_session_job(invocation)
    detail = restart.describe_result(result, invocation)
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status=ClosingStatus.PASSED if result["returncode"] == 0 else ClosingStatus.FAILED,
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
    lane: DispatchLane,
    node: str | None,
    host: str,
    mcps_root: pathlib.Path | None,
) -> DispatchJob | None:
    """Take one job of this runner's lane and run it to its close.

    A job is run by its command, whichever lane gave it: until the MCPs
    queue split the session verbs into their own lane (mig 795), the hub
    lane gave both kinds.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        lane: The hub lane or the sessions lane.
        node: Restrict claims to this node, plus jobs that name none, or
            None to take any job of the lane.
        host: The node a session job is reported started on: this machine.
        mcps_root: The MCPs checkout a base rebuild runs in, or None when
            the runner was started without ``--mcps-root``. Session verbs
            never use it: they run a sealed release
            (:mod:`fleet.core.session_release`).

    Returns:
        The job that was claimed, whatever became of it, or None when the
        lane was empty -- the outcome of most ticks.

    Raises:
        AppError: Only from the queue calls themselves. A LOCAL refusal (no
            MCPs checkout, a target that cannot be resolved, a session
            release that is missing or fails its seal) is reported to
            the queue as ``refused`` with its code and message verbatim and
            does not propagate; see the module docstring for why that is
            transport rather than recovery.
    """
    job = queue.claim_next(
        credentials,
        lane=lane,
        tags=(),
        node=node,
        projects=None,
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    if job is None:
        return None
    _log.info("claimed %s", encode_job_line(job))
    if job["command"] is DispatchCommand.BUILD_BASES:
        # The verbs that run on the hub itself, synchronously —
        # fleet.core.rebuild's module docstring carries the why.
        return rebuild_job(credentials, job, identity, mcps_root=mcps_root)
    return session_job(credentials, job, identity, node=host)


#: What each lane's check-in says the runner claims.
_LANE_CLAIMS = {
    DispatchLane.HUB: "the queue's hub lane (build-bases)",
    DispatchLane.SESSIONS: "the queue's sessions lane (the session verbs)",
}


def require_runner_lane(value: str) -> DispatchLane:
    """Name the lane this runner claims from.

    Args:
        value: The ``--lane`` value.

    Returns:
        The hub lane or the sessions lane.

    Raises:
        ValueError: ``FLEET_AGENT_USAGE`` for any other value, the node lane
            included, which is :mod:`fleet.cli.node_agent`'s.
    """
    for lane in _LANE_CLAIMS:
        if lane.value == value:
            return lane
    words = [lane.value for lane in _LANE_CLAIMS]
    raise ValueError(f"FLEET_AGENT_USAGE: {LANE_FLAG} is one of {words}, not {value!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one tick: claim and run at most one job of the lane, then, on the
    hub lane, publish the registry and observe.

    With ``--announce`` the tick instead posts the check-in that registers
    this runner's session on the board's ledger and claims nothing. The
    queue's close writer posts a task-naming job's outcome under the
    closer's label and the board admits a post only from a ledgered
    session, so a runner that never announced could not close a
    session-exit job, which names the finished task (MCPs board task
    2fecad69).

    A sessions tick names its machine with ``--node``: the claim takes jobs
    for it and jobs naming none, and its session jobs are reported started
    there. It refuses ``--mcps-root`` and ``--registry``, which only the hub
    tick's passes read.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 whenever the agent itself worked, including when a job was refused
        or a bake failed. See the module docstring.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required one is absent; ``FLEET_AGENT_USAGE`` for a lane this
            runner does not claim, or a sessions tick given a hub-only flag.
        AppError: When the queue or the board cannot be reached or answered
            a shape this runner cannot read, or when the workspace does not
            decode.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    announce = queue.ANNOUNCE_FLAG in tokens
    parsed = cli_args.parse_single_flags(
        [token for token in tokens if token != queue.ANNOUNCE_FLAG], _FLAGS
    )
    # The workspace is still read, and refused when malformed, although
    # neither lane dispatches to a node: a runner whose registry does not
    # decode has no business claiming anything, and the node runners beside
    # it read the same file.
    _config.load_workspace(parsed)
    lane = require_runner_lane(cli_args.require_flag(parsed, LANE_FLAG))
    if lane is DispatchLane.SESSIONS:
        given = [flag for flag in _HUB_ONLY_FLAGS if flag in parsed]
        if given:
            raise ValueError(
                f"FLEET_AGENT_USAGE: the sessions lane takes no {' or '.join(given)}; "
                "only a hub tick publishes and observes"
            )
        host = cli_args.require_flag(parsed, NODE_FLAG)
    else:
        host = HUB_NODE
    agent = cli_args.require_flag(parsed, AGENT_FLAG)
    session_id = cli_args.require_flag(parsed, SESSION_FLAG)
    project_root = pathlib.Path(cli_args.require_flag(parsed, ROOT_FLAG)).resolve()
    identity = queue.identity_arguments(agent, session_id, str(project_root))
    if announce:
        _log.info(
            "%s",
            queue.announce(
                board_config.load_credentials(),
                machine=f"{sys.platform}:{_test_hooks.hostname()}",
                body=(
                    f"{agent}: claims {_LANE_CLAIMS[lane]} and closes each job it claims, "
                    "which posts the outcome to the job's task thread under this label"
                ),
                identity=identity,
            ),
        )
        return 0
    credentials = queue.load_credentials()

    raw_mcps_root = parsed.get(MCPS_ROOT_FLAG)
    mcps_root = pathlib.Path(raw_mcps_root).resolve() if raw_mcps_root is not None else None

    claimed = claim_pass(
        credentials,
        identity,
        lane=lane,
        node=parsed.get(NODE_FLAG),
        host=host,
        mcps_root=mcps_root,
    )
    if claimed is None:
        _log.info("%s lane empty", lane.value)
    if lane is DispatchLane.SESSIONS:
        return 0
    hub_passes.publish_pass(credentials, identity, repo_root=project_root)

    raw_registry = parsed.get(REGISTRY_FLAG)
    if raw_registry is None:
        # Said out loud, every tick: a ledger with no fleet rows must be
        # distinguishable from a runner that was never asked to write any.
        _log.info("session observation skipped: no %s given", REGISTRY_FLAG)
    else:
        hub_passes.observe_pass(pathlib.Path(raw_registry).resolve(), identity)
    hub_passes.reap_pass(project_root)
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
    "HUB_NODE",
    "LANE_FLAG",
    "MCPS_ROOT_FLAG",
    "NODE_FLAG",
    "REGISTRY_FLAG",
    "ROOT_FLAG",
    "SESSION_FLAG",
    "claim_pass",
    "entrypoint",
    "main",
    "rebuild_job",
    "require_runner_lane",
    "session_job",
]
