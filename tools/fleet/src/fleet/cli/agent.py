"""CLI: one tick of the runner that serves the corvis dispatch queue's hub
lane, its sessions lane or its netboot lane.

Usage:
    fleet-agent --config registry --repo-root /home/corvis/PROJECTS/API \\
        --agent fleet-hub-diphtheria --session <uuid> --lane hub [--announce]
    fleet-agent --config registry --repo-root /home/corvis/PROJECTS/API \\
        --agent fleet-sessions-diphtheria --session <uuid> --lane sessions \\
        --node diphtheria [--announce]
    fleet-agent --config registry --repo-root /home/corvis/PROJECTS/API \\
        --agent fleet-netboot-diphtheria --session <uuid> --lane netboot \\
        --netboot-release <release>/MCPs --nodejs /usr/bin/node \\
        --served-root /srv/corvis-netboot --env-file <workspace>/.env [--announce]

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

THE HUB, SESSIONS AND NETBOOT LANES ONLY (MCPs board tasks fd5cabfa, A5,
c39ee056 and 83102d9d). With ``--lane hub`` this runner claims the
``build-bases`` rebuild, the control duty's verb; with ``--lane sessions``
it claims the session verbs (restart, revive, the two kills, the compact
and the exit), the sessions duty's, on the machine whose sessions they are;
with ``--lane netboot`` it claims the arms, the netboot duty's, whatever
machine each names, since the machine an arm names is its target and never
where it runs. Every one runs where it was claimed and closes in the tick
that claimed it (:mod:`fleet.cli.hub_jobs`). The MCPs hub runner places
each lane's row by the duty the node registry gives its machine, so the
queue names no node for a session verb: holding the duty is the placement.
The make targets, which run on a fleet node against a
checked-out commit, are the ``node`` lane, drained by one
:mod:`fleet.cli.node_agent` per enabled node. Until 2026-09-21 one runner
took every kind of job oldest-first, so a revive queued behind 129 stale
kills (board task 4199d1fb) and nine nodes behaved as one slow node; lanes
and per-node runners are the fix, and nothing here collects a suite any more
because nothing here starts one.

A HUB TICK IS FOUR PASSES, IN THIS ORDER; a sessions tick is the first
alone, since publishing, observing and reaping are the control duty's, and
a netboot tick is the first and then the netboot duty's own adopt pass,
which gives each machine that checked in after a reinstall its roster name
and address (MCPs board task 49d366ac, :mod:`fleet.core.netboot_adopt`):

    1. CLAIM. Give back each job held under this label whose run a restart
       ended (:mod:`fleet.cli.orphans`), then take at most one job of the
       lane and run it to its close, reported started as this tick's run.
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
    3. OBSERVE. Read the Claude Code session records of every enabled
       machine whose roster file declares ``ledger_observer`` fleet-arm over
       ssh and hand them to the board's session ledger (MCPs board task
       5a3865bf; :mod:`fleet.core.observe` carries the why). Last, because
       it is the pass that touches every node, and a node asleep costs a
       ten-second ssh timeout that the queue work should not wait behind.
       The machines are the registry's own roster (``registry/machines``,
       MCPs board task e447de72), so every hub tick observes; until then
       they were read from MCPs' ``fleet-nodes.json``, named by a
       ``--registry`` flag, and a tick without it observed nothing.
    4. REAP. Read each runner host's Workers against GitHub's runner
       list, read again two minutes later where one was named, and end each
       Worker both readings name as holding a job GitHub dropped, by pid,
       through the host's reaper (MCPs board task 715b9ec5,
       :mod:`fleet.core.runner_reap_dropped`). Last, because its second
       reading waits, and nothing above should wait behind it.

At most ONE job per tick, deliberately: a bake, a session verb or an arm
holds its machine, and a second claimed before the first closed would run
beside it.

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
from collections.abc import Mapping, Sequence

from board_watch import config as board_config
from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import JSONObject
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.mcp_client import McpCredentials
from platform_core.report_line import emit_error_line, emit_line

from fleet.cli import _config, hub_jobs, hub_passes, job_close, orphans
from fleet.contracts.dispatch import (
    DispatchCommand,
    DispatchJob,
    DispatchLane,
    encode_job_line,
)
from fleet.core import _test_hooks, netboot_adopt, netboot_arm, queue, rebuild, tick_run

_log = get_logger(__name__)

AGENT_FLAG = "--agent"
SESSION_FLAG = "--session"
ROOT_FLAG = "--repo-root"
NODE_FLAG = "--node"
MCPS_ROOT_FLAG = "--mcps-root"
LANE_FLAG = "--lane"
NETBOOT_RELEASE_FLAG = "--netboot-release"
NODEJS_FLAG = "--nodejs"
SERVED_ROOT_FLAG = "--served-root"
ENV_FILE_FLAG = "--env-file"

#: The flag only a hub tick's base rebuilds read, which a sessions or
#: netboot tick refuses rather than ignores.
_HUB_ONLY_FLAGS = (MCPS_ROOT_FLAG,)

#: The flags a netboot tick places its arms with, which every other lane
#: refuses rather than ignores.
NETBOOT_FLAGS = (NETBOOT_RELEASE_FLAG, NODEJS_FLAG, SERVED_ROOT_FLAG, ENV_FILE_FLAG)

_FLAGS = (
    _config.CONFIG_FLAG,
    _config.RECORDS_FLAG,
    AGENT_FLAG,
    SESSION_FLAG,
    ROOT_FLAG,
    NODE_FLAG,
    MCPS_ROOT_FLAG,
    LANE_FLAG,
    *NETBOOT_FLAGS,
)


def claim_pass(
    credentials: McpCredentials,
    identity: JSONObject,
    *,
    agent: str,
    run: tick_run.TickRun,
    lane: DispatchLane,
    node: str | None,
    host: str,
    mcps_root: pathlib.Path | None,
    netboot: netboot_arm.NetbootPlace | None,
) -> DispatchJob | None:
    """Give back every held job whose run is gone, then take one job of this
    runner's lane and run it to its close.

    A job is run by its command, whichever lane gave it: until the MCPs
    queue split the session verbs into their own lane (mig 795), the hub
    lane gave both kinds. So a job whose kind needs what this runner was not
    started with is refused naming the flag, rather than run without it.

    The give-back comes first (:mod:`fleet.cli.orphans`, MCPs board task
    05dd6588): a job whose run a restart of the hub runner ended is back in
    the queue before this claim asks for one, so this tick takes it again.
    The claim holds a short lease until the start, which reports the job
    started as this tick's run and renews it to the full lease.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        agent: This runner's label, which the queue holds its jobs under.
        run: This tick's run, which a job is reported started as.
        lane: The hub, sessions or netboot lane.
        node: Restrict claims to this node, plus jobs that name none, or
            None to take any job of the lane.
        host: The node a job is reported started on: this machine.
        mcps_root: The MCPs checkout a base rebuild runs in, or None when
            the runner was started without ``--mcps-root``. Session verbs
            and arms never use it: they run a sealed release.
        netboot: Where an arm runs from and writes to, or None when the
            runner is not a netboot tick.

    Returns:
        The job that was claimed, whatever became of it, or None when the
        lane was empty -- the outcome of most ticks.

    Raises:
        AppError: Only from the queue calls themselves, and from asking this
            machine's user manager about a held job's run. A LOCAL refusal (no
            MCPs checkout, a target that cannot be resolved, a release that
            is missing or fails its seal, no key to mint with) is reported
            to the queue as ``refused`` with its code and message verbatim
            and does not propagate; see the module docstring for why that is
            transport rather than recovery.
    """
    orphans.orphan_pass(credentials, identity, agent=agent, own=run)
    job = queue.claim_next(
        credentials,
        lane=lane,
        tags=(),
        node=node,
        projects=None,
        lease_seconds=hub_jobs.PRESTART_LEASE_SECONDS,
        identity=identity,
    )
    if job is None:
        return None
    _log.info("claimed %s", encode_job_line(job))
    run_id = tick_run.encode_run_id(run)
    if job["command"] is DispatchCommand.BUILD_BASES:
        # The verbs that run on the hub itself, synchronously —
        # fleet.core.rebuild's module docstring carries the why.
        if mcps_root is None:
            return job_close.refuse(
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
        return hub_jobs.rebuild_job(
            credentials, job, identity, mcps_root=mcps_root, node=host, run_id=run_id
        )
    if job["command"] in (DispatchCommand.NETBOOT_ARM, DispatchCommand.NETBOOT_RETIRE):
        if netboot is None:
            return job_close.refuse(
                credentials,
                job,
                identity,
                detail=(
                    f"{netboot_arm.NO_RELEASE_CODE}: this runner was started without "
                    f"{', '.join(NETBOOT_FLAGS)}, so it has no netboot release to run "
                    f"{job['command']} from; the netboot duty's fleet-netboot runner claims "
                    "the netboot lane"
                ),
            )
        return hub_jobs.netboot_job(
            credentials, job, identity, place=netboot, node=host, run_id=run_id
        )
    return hub_jobs.session_job(credentials, job, identity, node=host, run_id=run_id)


#: What each lane's check-in says the runner claims.
_LANE_CLAIMS = {
    DispatchLane.HUB: "the queue's hub lane (build-bases)",
    DispatchLane.SESSIONS: "the queue's sessions lane (the session verbs)",
    DispatchLane.NETBOOT: "the queue's netboot lane (netboot-arm, netboot-retire)",
}


def require_runner_lane(value: str) -> DispatchLane:
    """Name the lane this runner claims from.

    Args:
        value: The ``--lane`` value.

    Returns:
        The hub, sessions or netboot lane.

    Raises:
        ValueError: ``FLEET_AGENT_USAGE`` for any other value, the node lane
            included, which is :mod:`fleet.cli.node_agent`'s.
    """
    for lane in _LANE_CLAIMS:
        if lane.value == value:
            return lane
    words = [lane.value for lane in _LANE_CLAIMS]
    raise ValueError(f"FLEET_AGENT_USAGE: {LANE_FLAG} is one of {words}, not {value!r}")


def _refuse_flags(
    parsed: Mapping[str, str], lane: DispatchLane, flags: Sequence[str], why: str
) -> None:
    """Refuse a tick given flags its lane does not read.

    Args:
        parsed: The parsed flags.
        lane: The lane this tick claims from.
        flags: The flags the lane takes none of.
        why: Why it takes none of them.

    Raises:
        ValueError: ``FLEET_AGENT_USAGE`` naming each given flag.
    """
    given = [flag for flag in flags if flag in parsed]
    if given:
        raise ValueError(
            f"FLEET_AGENT_USAGE: the {lane.value} lane takes no {' or '.join(given)}; {why}"
        )


def lane_host(parsed: Mapping[str, str], lane: DispatchLane) -> str:
    """Name the machine this tick's jobs are reported started on.

    A sessions tick names it with ``--node``, since the sessions duty's
    machine can reach in to a session host and act as it (MCPs board task
    78a1eca7). The hub and netboot lanes run on whichever machine holds
    their duty, so they report on that machine by its own name (MCPs board
    task c39ee056).

    Args:
        parsed: The parsed flags.
        lane: The lane this tick claims from.

    Returns:
        The machine's alias.

    Raises:
        ValueError: ``FLEET_AGENT_USAGE`` for a flag the lane does not read,
            or a sessions tick that names no ``--node``.
    """
    if lane is not DispatchLane.HUB:
        _refuse_flags(parsed, lane, _HUB_ONLY_FLAGS, "only a hub tick rebuilds the bases")
    if lane is not DispatchLane.NETBOOT:
        _refuse_flags(parsed, lane, NETBOOT_FLAGS, "only a netboot tick arms")
    if lane is DispatchLane.SESSIONS:
        return cli_args.require_flag(parsed, NODE_FLAG)
    if lane is DispatchLane.NETBOOT:
        # The machine an arm names is its target, never where it runs, so
        # the lane claims every arm, whatever machine it names.
        _refuse_flags(
            parsed,
            lane,
            (NODE_FLAG,),
            "it claims every arm, and the machine an arm names is its target",
        )
    return _test_hooks.hostname()


def netboot_place(parsed: Mapping[str, str], registry: pathlib.Path) -> netboot_arm.NetbootPlace:
    """Read where a netboot tick's arms run from and write to.

    Args:
        parsed: The parsed flags.
        registry: The registry directory this tick loaded.

    Returns:
        The installed release, the node binary, the served root, the
        workspace env file and the registry.

    Raises:
        ValueError: When any of :data:`NETBOOT_FLAGS` is absent.
    """
    return netboot_arm.NetbootPlace(
        release=pathlib.Path(cli_args.require_flag(parsed, NETBOOT_RELEASE_FLAG)),
        nodejs=cli_args.require_flag(parsed, NODEJS_FLAG),
        served_root=pathlib.Path(cli_args.require_flag(parsed, SERVED_ROOT_FLAG)),
        env_file=pathlib.Path(cli_args.require_flag(parsed, ENV_FILE_FLAG)),
        registry=registry,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one tick: claim and run at most one job of the lane, then, on the
    netboot lane, adopt each reinstalled machine that checked in, and on the
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
    there. A netboot tick names where its arms run from and write to with
    :data:`NETBOOT_FLAGS` and claims with no node. Each refuses the flags
    only another lane reads (:func:`lane_host`).

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 whenever the agent itself worked, including when a job was refused
        or a bake failed. See the module docstring.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required one is absent; ``FLEET_AGENT_USAGE`` for a lane this
            runner does not claim, or a tick given a flag only another lane
            reads.
        AppError: When the queue or the board cannot be reached or answered
            a shape this runner cannot read, when the workspace does not
            decode, when a claiming tick runs in no unit of a user manager
            (:data:`fleet.core.tick_run.UNBOUND_CODE`), or when that manager
            cannot be asked about a held job's run.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    announce = queue.ANNOUNCE_FLAG in tokens
    parsed = cli_args.parse_single_flags(
        [token for token in tokens if token != queue.ANNOUNCE_FLAG], _FLAGS
    )
    # The workspace is read, and refused when malformed, although no lane
    # here dispatches to a node: a runner whose registry does not decode has
    # no business claiming anything, the node runners beside it read the same
    # files, and a hub tick's observe pass walks its machines.
    loaded = _config.load_workspace(parsed)
    lane = require_runner_lane(cli_args.require_flag(parsed, LANE_FLAG))
    host = lane_host(parsed, lane)
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

    netboot = netboot_place(parsed, loaded.registry) if lane is DispatchLane.NETBOOT else None
    claimed = claim_pass(
        credentials,
        identity,
        agent=agent,
        run=tick_run.read_own_run(),
        lane=lane,
        node=parsed.get(NODE_FLAG),
        host=host,
        mcps_root=mcps_root,
        netboot=netboot,
    )
    if claimed is None:
        _log.info("%s lane empty", lane.value)
    if netboot is not None:
        for line in netboot_adopt.adopt_pass(netboot):
            _log.info("%s", line)
    if lane is not DispatchLane.HUB:
        return 0
    hub_passes.publish_pass(credentials, identity, repo_root=project_root)
    hub_passes.observe_pass(
        loaded.workspace["machines"], identity, loaded.workspace["native_evidence_seconds"]
    )
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
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


# Without this, `python -m fleet.cli.agent` imports the module, runs nothing
# and exits 0 -- which reads as a tick that found an empty queue, on a runner
# that in fact never asked.
if __name__ == "__main__":
    entrypoint()


__all__ = [
    "AGENT_FLAG",
    "ENV_FILE_FLAG",
    "LANE_FLAG",
    "MCPS_ROOT_FLAG",
    "NETBOOT_FLAGS",
    "NETBOOT_RELEASE_FLAG",
    "NODEJS_FLAG",
    "NODE_FLAG",
    "ROOT_FLAG",
    "SERVED_ROOT_FLAG",
    "SESSION_FLAG",
    "claim_pass",
    "entrypoint",
    "lane_host",
    "main",
    "netboot_place",
    "require_runner_lane",
]
