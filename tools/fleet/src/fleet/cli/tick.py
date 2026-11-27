"""CLI: one fleet tick, run by the MCPs hub runner as a native process.

Usage:
    poetry run -- python -m fleet.cli.tick --api-root /home/corvis/PROJECTS/API \\
        --log-directory /home/corvis/.local/state/corvis/logs --lane hub
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane node --node sedona
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane announce --node sedona
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane hub-announce
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane sessions --node diphtheria
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane netboot --netboot-release <release>/MCPs --nodejs /usr/bin/node \\
        --served-root /srv/corvis-netboot --env-file <workspace>/.env
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane inspect --inspect-release <release>/MCPs --nodejs /usr/bin/node \\
        --workspace <workspace>
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane host-target --project osrs-lab/manual-desktop

The command behind every fleet job of the MCPs hub runner (MCPs board task
b27bd34a, ``scripts/fleet/hub-jobs.json``): its ``fleet-hub`` job runs the
hub lane and each ``fleet-node-<alias>`` one node's lane, every 3 minutes,
one run at a time. ``announce`` is the node lane's one-time check-in, each
node job's boot command, and ``hub-announce`` the hub's, which the board
needs since the queue's close posts a task-naming job's outcome under the
closing runner's label (MCPs board task 2fecad69). The hub lane runs on the
machine holding the control duty, as ``fleet-hub-<that machine's name>``
(MCPs board task c39ee056), so moving the duty gives the new holder its own
identity rather than lending it another machine's. The ``fleet-sessions``
job, placed on the machine holding the sessions duty, runs the ``sessions``
lane for that machine as ``fleet-sessions-<alias>``, with
``sessions-announce`` its check-in (MCPs board task c39ee056). The
``fleet-netboot`` job, placed on the machine holding the netboot duty, runs
the ``netboot`` lane as ``fleet-netboot-<that machine's name>``, with
``netboot-announce`` its check-in (MCPs board task 83102d9d): the row names
the installed netboot release it arms from, the node binary, the served root
and the workspace env file, and the tick hands them to the agent. The
``fleet-inspect`` job, placed on the machine holding the control duty, runs
the ``inspect`` lane as ``fleet-inspect-<that machine's name>``, with
``inspect-announce`` its check-in (MCPs board task 3e9c99af): the row names
the installed fleet-audit release it inspects from, the node binary and the
workspace whose registry names the roster, each lane's flags read from
:data:`fleet.cli.agent_lanes.LANE_PLACES`. The ``osrs-desktop-update`` job,
placed on the machine holding the control duty, runs the ``host-target``
lane every hour for the project its ``--project`` names (MCPs board task
7cd6036a): rolled's ``fleet-host-target`` (:mod:`fleet.cli.host_target`)
runs that project's declared target on its node, acting as
``fleet-host-target-<that machine's name>``, the identity a lease names.
Each row runs this module with ``poetry run`` in the fleet package's own
environment.

WHAT A TICK DOES:

* reads the machine's credentials from the hpc-wake pump's untracked
  ``runs/credentials.env``, ``KEY=VALUE`` lines read by
  :mod:`platform_core.env_file` as that pump reads them, and hands them to
  the agent's environment and to nothing else. A missing key surfaces as the
  agent's own named refusal;
* removes this lane's logs older than :data:`RETENTION_DAYS`, so the log
  directory bounds itself;
* runs :mod:`fleet.cli.rolled`, which runs the agent from the rolled commit,
  with its streams captured through the package's deadline-carrying run
  hook;
* appends one record to the day's log: ``TICK START``, the agent's output,
  ``TICK EXIT`` with its status. Until 2026-09-20 a tick wrote nothing, and a
  tick hung for three days was found only by its absence from another log.

THE EXIT CODE IS THE AGENT'S. The agents exit 0 whenever they worked, refused
jobs and failed suites included, so a non-zero means the tick itself broke
and the hub runner's record of the run shows something real. The host
target exits with make's status, so a host-target tick's non-zero is the
target failing, which is what its row's record is read for.

THE DEADLINES NEST. The agent's wall is one minute inside the row's
40-minute deadline (:data:`fleet.cli.rolled.AGENT_WALL_SECONDS`); this
tick's is half a minute past the agent's, so a stuck agent is ended and
logged by the launcher, a stuck launcher is ended and logged here, and the
runner's deadline ends only a tick that escaped both.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import sys
import uuid
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Final, TypedDict

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.env_file import parse_env_file
from platform_core.members import find_member
from platform_core.report_line import emit_error_line, emit_line

from fleet.cli import agent_lanes
from fleet.cli import host_target as host_target_cli
from fleet.cli import rolled as rolled_cli
from fleet.contracts.dispatch import DispatchLane
from fleet.core import _test_hooks, names

API_ROOT_FLAG: Final = "--api-root"
LOG_DIRECTORY_FLAG: Final = "--log-directory"
LANE_FLAG: Final = "--lane"
NODE_FLAG: Final = "--node"
#: The project whose host target a host-target tick runs, the host target's
#: own flag, which the tick hands it unchanged.
PROJECT_FLAG: Final = host_target_cli.PROJECT_FLAG


class Lane(StrEnum):
    """The tick an MCPs hub runner job runs; each value is its ``--lane`` word."""

    HUB = "hub"
    HUB_ANNOUNCE = "hub-announce"
    SESSIONS = "sessions"
    SESSIONS_ANNOUNCE = "sessions-announce"
    NETBOOT = "netboot"
    NETBOOT_ANNOUNCE = "netboot-announce"
    INSPECT = "inspect"
    INSPECT_ANNOUNCE = "inspect-announce"
    NODE = "node"
    ANNOUNCE = "announce"
    RESEARCH = "research"
    RESEARCH_ANNOUNCE = "research-announce"
    HOST_TARGET = "host-target"


#: The ticks ``fleet-agent`` runs, each a duty's tick or its one-time
#: check-in, by the queue lane that duty claims: the control duty's hub and
#: inspections (MCPs board task 3e9c99af), the sessions duty's and the
#: netboot duty's. One table, so the runners are planned by one function and
#: named by one (MCPs board task af8b5662).
_AGENT_LANES: Final[Mapping[Lane, DispatchLane]] = {
    Lane.HUB: DispatchLane.HUB,
    Lane.HUB_ANNOUNCE: DispatchLane.HUB,
    Lane.SESSIONS: DispatchLane.SESSIONS,
    Lane.SESSIONS_ANNOUNCE: DispatchLane.SESSIONS,
    Lane.NETBOOT: DispatchLane.NETBOOT,
    Lane.NETBOOT_ANNOUNCE: DispatchLane.NETBOOT,
    Lane.INSPECT: DispatchLane.INSPECT,
    Lane.INSPECT_ANNOUNCE: DispatchLane.INSPECT,
    Lane.RESEARCH: DispatchLane.RESEARCH,
    Lane.RESEARCH_ANNOUNCE: DispatchLane.RESEARCH,
}

#: The lanes that post the runner's check-in and claim nothing.
_ANNOUNCE_LANES: Final[frozenset[Lane]] = frozenset(
    {
        Lane.HUB_ANNOUNCE,
        Lane.SESSIONS_ANNOUNCE,
        Lane.NETBOOT_ANNOUNCE,
        Lane.INSPECT_ANNOUNCE,
        Lane.ANNOUNCE,
        Lane.RESEARCH_ANNOUNCE,
    }
)

#: The namespace a hub, sessions or netboot runner's session id is derived in.
SESSIONS_NAMESPACE: Final = uuid.NAMESPACE_URL


def runner_identity(lane: DispatchLane | Lane, alias: str) -> tuple[str, str]:
    """The label and session id a machine's duty runner acts as.

    Derived from the queue lane its duty claims and the machine's alias, as
    a node runner's are from the alias
    (:func:`fleet.core.names.node_identity`): one stable session per
    duty and machine, so a check-in binds one label to one session for
    good, and a duty moving to another machine gives that machine its own.
    Until MCPs board task c39ee056 the one hub was austinpc, whose fixed
    pair, ``fleet-runner-austinpc`` and session a850f688, is bound for good
    and so is not reused. The host-target tick claims no queue lane and is
    named by its own lane alike (MCPs board task 7cd6036a).

    Args:
        lane: The queue lane the duty claims, one of :data:`_AGENT_LANES`'
            values, or :attr:`Lane.HOST_TARGET`. The node lane's runner is
            ``fleet-node-agent``, named by ``node_identity``.
        alias: The machine's registry name.

    Returns:
        ``fleet-<lane>-<alias>`` and the version-5 UUID of
        ``fleet-agent/<lane>/<alias>`` in :data:`SESSIONS_NAMESPACE`.
    """
    return (
        f"fleet-{lane.value}-{alias}",
        str(uuid.uuid5(SESSIONS_NAMESPACE, f"fleet-agent/{lane.value}/{alias}")),
    )


#: The credentials file, relative to the API checkout.
ENV_FILE: Final = pathlib.PurePosixPath("tools/hpc-wake/runs/credentials.env")

#: How many days a lane's log is kept.
RETENTION_DAYS: Final[int] = 14

#: This tick's deadline for the launcher: thirty seconds past the agent's.
TICK_WALL_SECONDS: Final[int] = rolled_cli.AGENT_WALL_SECONDS + 30


def require_lane(value: str) -> Lane:
    """Name the lane a tick runs.

    Args:
        value: The ``--lane`` value.

    Returns:
        The :class:`Lane` whose word is ``value``.

    Raises:
        ValueError: ``FLEET_TICK_USAGE`` for any other value.
    """
    lane = find_member(value, Lane)
    if lane is not None:
        return lane
    words = [member.value for member in Lane]
    raise ValueError(f"FLEET_TICK_USAGE: {LANE_FLAG} is one of {words}, not {value!r}")


class TickPlan(TypedDict):
    """What one lane's tick runs and where it writes.

    Attributes:
        arguments: Everything after ``python -m fleet.cli.rolled``.
        stem: The log name before its date: ``fleet-agent`` for the hub,
            ``fleet-sessions-<alias>`` for the sessions runner,
            ``fleet-netboot-<alias>`` for the netboot runner,
            ``fleet-host-target-<alias>`` for the host-target runner and
            ``fleet-node-<alias>`` for a node, each shared by its announce.
        header: What the ``TICK START`` line names after its time.
    """

    arguments: tuple[str, ...]
    stem: str
    header: str


#: The ticks that place their commands, by the queue lane whose flags they
#: hand the agent (:data:`fleet.cli.agent_lanes.LANE_PLACES`); their
#: check-ins place nothing.
_PLACED_LANES: Final[Mapping[Lane, DispatchLane]] = {
    Lane.NETBOOT: DispatchLane.NETBOOT,
    Lane.INSPECT: DispatchLane.INSPECT,
}

#: The flags each tick hands its agent, which it requires, and why every
#: other tick refuses those of them it does not read itself: each placed
#: lane's places and the host-target lane's project. Every other tick hands
#: none.
_HANDED_FLAGS: Final[Mapping[Lane, tuple[tuple[str, ...], str]]] = {
    **{
        lane: (agent_lanes.LANE_PLACES[placed][0], agent_lanes.LANE_PLACES[placed][1])
        for lane, placed in _PLACED_LANES.items()
    },
    Lane.HOST_TARGET: ((PROJECT_FLAG,), "only a host-target tick runs a project's target"),
}

#: Every flag a tick hands its agent, each once, in the parser's order.
HANDED_FLAGS: Final[tuple[str, ...]] = tuple(
    sorted({flag for flags, _ in _HANDED_FLAGS.values() for flag in flags})
)

_FLAGS: Final = (API_ROOT_FLAG, LOG_DIRECTORY_FLAG, LANE_FLAG, NODE_FLAG, *HANDED_FLAGS)


def _handed_arguments(lane: Lane, given: Mapping[str, str]) -> tuple[str, ...]:
    """The flags a lane's tick hands the agent, and nothing for any other.

    Args:
        lane: Which tick this is.
        given: The tick's own values of :data:`HANDED_FLAGS`.

    Returns:
        Each of the lane's own flags and its value, in that order, for a
        placed or host-target lane; an empty tuple for every other lane.

    Raises:
        ValueError: ``FLEET_TICK_USAGE`` when a lane lacks one of its flags,
            or any lane is given a flag only another reads.
    """
    handed = _HANDED_FLAGS.get(lane)
    own = () if handed is None else handed[0]
    # The lane's own entry leaves nothing once its own flags are taken out,
    # so every entry is read alike.
    for flags, why in _HANDED_FLAGS.values():
        foreign = [flag for flag in flags if flag in given and flag not in own]
        if foreign:
            raise ValueError(
                f"FLEET_TICK_USAGE: the {lane.value} lane takes no {', '.join(foreign)}; {why}"
            )
    missing = [flag for flag in own if flag not in given]
    if missing:
        raise ValueError(f"FLEET_TICK_USAGE: the {lane.value} lane needs {', '.join(missing)}")
    return tuple(token for flag in own for token in (flag, given[flag]))


def _agent_plan(
    api_root: pathlib.Path,
    lane: Lane,
    agent_lane: DispatchLane,
    node: str | None,
    trailing: tuple[str, ...],
) -> TickPlan:
    """Plan a ``fleet-agent`` tick: the hub, sessions or netboot runner's.

    The sessions duty is placed by the tick's ``--node``; the control and
    netboot duties run on the machine the tick runs on, which is named by
    its hostname. The hub's tick alone hands the agent the MCPs checkout it
    rebuilds in, and keeps its historic log stem, ``fleet-agent``.

    Args:
        api_root: The API checkout.
        lane: Which tick this is.
        agent_lane: The queue lane its duty claims (:data:`_AGENT_LANES`).
        node: The machine's alias for the sessions lane; None otherwise.
        trailing: The netboot flags and the announce flag, as the lane has
            them.

    Returns:
        The launcher's command line, the log stem and the record's header.

    Raises:
        ValueError: ``FLEET_TICK_USAGE`` when the sessions lane names no
            node, or the hub or netboot lane names one.
    """
    placement: tuple[str, ...]
    if agent_lane is DispatchLane.SESSIONS:
        if node is None:
            raise ValueError(f"FLEET_TICK_USAGE: the {lane.value} lane needs {NODE_FLAG}")
        alias, placement, header = node, ("--node", node), f"{lane.value} {node}"
    else:
        if node is not None:
            raise ValueError(f"FLEET_TICK_USAGE: the {lane.value} lane takes no {NODE_FLAG}")
        alias, placement, header = _test_hooks.hostname(), (), lane.value
    label, session = runner_identity(agent_lane, alias)
    rebuild = ("--mcps-root", str(api_root.parent / "MCPs")) if lane is Lane.HUB else ()
    repo = str(api_root)
    module = "fleet-research" if agent_lane is DispatchLane.RESEARCH else "fleet-agent"
    lane_arguments = () if agent_lane is DispatchLane.RESEARCH else ("--lane", agent_lane.value)
    return TickPlan(
        arguments=(
            rolled_cli.REPO_ROOT_FLAG,
            repo,
            rolled_cli.AGENT_FLAG,
            module,
            rolled_cli.SEPARATOR,
            "--agent",
            label,
            "--session",
            session,
            "--repo-root",
            repo,
            *lane_arguments,
            *placement,
            *rebuild,
            *trailing,
        ),
        stem="fleet-agent" if agent_lane is DispatchLane.HUB else label,
        header=header,
    )


def _host_target_plan(api_root: pathlib.Path, node: str | None, project: str) -> TickPlan:
    """Plan a host-target tick: one run of a project's declared host target.

    The tick runs on the machine holding the control duty, named by its
    hostname, which is where the hub's mirrors and records are; the node the
    target runs on is the project's registry entry's, never the tick's.

    Args:
        api_root: The API checkout.
        node: The tick's ``--node``, which this lane refuses.
        project: The project key its ``--project`` names.

    Returns:
        The launcher's command line, the runner's label as the log stem and
        the lane and project as the record's header.

    Raises:
        ValueError: ``FLEET_TICK_USAGE`` when the tick names a node.
    """
    if node is not None:
        raise ValueError(
            f"FLEET_TICK_USAGE: the {Lane.HOST_TARGET.value} lane takes no {NODE_FLAG}; the "
            "node a host target runs on is its project's registry entry's"
        )
    label, session = runner_identity(Lane.HOST_TARGET, _test_hooks.hostname())
    return TickPlan(
        arguments=(
            rolled_cli.REPO_ROOT_FLAG,
            str(api_root),
            rolled_cli.AGENT_FLAG,
            rolled_cli.HOST_TARGET_AGENT,
            rolled_cli.SEPARATOR,
            host_target_cli.AGENT_FLAG,
            label,
            host_target_cli.SESSION_FLAG,
            session,
            PROJECT_FLAG,
            project,
        ),
        stem=label,
        header=f"{Lane.HOST_TARGET.value} {project}",
    )


def plan_tick(
    api_root: pathlib.Path, lane: Lane, node: str | None, handed: Mapping[str, str]
) -> TickPlan:
    """Plan one lane's tick.

    Args:
        api_root: The API checkout.
        lane: Which tick this is.
        node: The machine's alias for the node and sessions lanes; None for
            the hub, netboot, inspect and host-target lanes.
        handed: The tick's own values of :data:`HANDED_FLAGS`, which a
            placed or host-target lane hands the agent its own of and every
            lane refuses another's.

    Returns:
        The launcher's command line, the log stem and the record's header.

    Raises:
        ValueError: ``FLEET_TICK_USAGE`` when a hub, netboot, inspect or
            host-target lane names a node, a node or sessions lane names
            none, or a lane's handed flags are missing from it or given to
            another.
    """
    repo = str(api_root)
    announce = ("--announce",) if lane in _ANNOUNCE_LANES else ()
    arms = _handed_arguments(lane, handed)
    if lane is Lane.HOST_TARGET:
        return _host_target_plan(api_root, node, handed[PROJECT_FLAG])
    agent_lane = _AGENT_LANES.get(lane)
    if agent_lane is not None:
        return _agent_plan(api_root, lane, agent_lane, node, (*arms, *announce))
    if node is None:
        raise ValueError(f"FLEET_TICK_USAGE: the {lane.value} lane needs {NODE_FLAG}")
    return TickPlan(
        arguments=(
            rolled_cli.REPO_ROOT_FLAG,
            repo,
            rolled_cli.AGENT_FLAG,
            rolled_cli.NODE_AGENT,
            rolled_cli.SEPARATOR,
            rolled_cli.NODE_FLAG,
            node,
            *announce,
        ),
        stem=names.node_runner_label(node),
        header=f"{lane.value} {node}",
    )


def remove_stale_logs(directory: pathlib.Path, stem: str, *, now_unix: int) -> int:
    """Remove this lane's logs last written more than the retention ago.

    Args:
        directory: Where the logs are.
        stem: The lane's log name before its date.
        now_unix: The time the retention is measured from.

    Returns:
        How many logs were removed.
    """
    cutoff = now_unix - RETENTION_DAYS * 86_400
    stale = [path for path in directory.glob(f"{stem}-*.log") if path.stat().st_mtime < cutoff]
    for path in stale:
        path.unlink()
    return len(stale)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one tick and append its record to the day's log.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        The agent's exit status, as the launcher reports it.

    Raises:
        ValueError: ``FLEET_TICK_USAGE`` for a malformed command line.
        AppError: ``CONFIG_ERROR`` when the credentials file holds a line
            with no ``=``.
        OSError: When the credentials file or the log directory cannot be
            read or written.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    api_root = pathlib.Path(cli_args.require_flag(parsed, API_ROOT_FLAG))
    log_directory = pathlib.Path(cli_args.require_flag(parsed, LOG_DIRECTORY_FLAG))
    lane = require_lane(cli_args.require_flag(parsed, LANE_FLAG))
    handed = {flag: parsed[flag] for flag in HANDED_FLAGS if flag in parsed}
    plan = plan_tick(api_root, lane, parsed.get(NODE_FLAG), handed)
    env_file = api_root / ENV_FILE
    credentials = tuple(
        parse_env_file(_test_hooks.read_text(env_file), source=str(env_file)).items()
    )

    started = _test_hooks.now()
    remove_stale_logs(log_directory, plan["stem"], now_unix=started)
    result = _test_hooks.run(
        (sys.executable, "-m", "fleet.cli.rolled", *plan["arguments"]),
        timeout_seconds=TICK_WALL_SECONDS,
        set_env=credentials,
    )
    started_at = datetime.datetime.fromtimestamp(started, tz=datetime.UTC)
    lines = [f"TICK START {started_at.isoformat()} {plan['header']} task-pid {os.getpid()}"]
    lines.extend(stream.rstrip("\n") for stream in (result["stdout"], result["stderr"]) if stream)
    finished_at = datetime.datetime.fromtimestamp(_test_hooks.now(), tz=datetime.UTC)
    lines.append(f"TICK EXIT {result['returncode']} {finished_at.isoformat()}")
    log = log_directory / f"{plan['stem']}-{started_at.date().isoformat()}.log"
    _test_hooks.append_text(log, "\n".join(lines))
    return result["returncode"]


def entrypoint() -> None:
    """Module entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = [
    "API_ROOT_FLAG",
    "ENV_FILE",
    "HANDED_FLAGS",
    "LANE_FLAG",
    "LOG_DIRECTORY_FLAG",
    "NODE_FLAG",
    "PROJECT_FLAG",
    "RETENTION_DAYS",
    "SESSIONS_NAMESPACE",
    "TICK_WALL_SECONDS",
    "Lane",
    "TickPlan",
    "entrypoint",
    "main",
    "plan_tick",
    "remove_stale_logs",
    "require_lane",
    "runner_identity",
]


# Without this, `python -m fleet.cli.tick` imports the module, runs nothing
# and exits 0, and a scheduled tick that ran no agent would read as a clean
# one.
if __name__ == "__main__":
    entrypoint()
