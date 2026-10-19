"""CLI: one fleet tick, run by the MCPs hub runner as a native process.

Usage:
    poetry run -- python -m fleet.cli.tick --api-root C:/Users/Test/PROJECTS/API \\
        --log-directory C:/Users/Test/AppData/Local/Temp/claude --lane hub
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane node --node sedona
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane announce --node sedona
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane elevated --node serendipity
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane hub-announce
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane sessions --node austinpc
    poetry run -- python -m fleet.cli.tick --api-root ... --log-directory ... \\
        --lane netboot --netboot-release <release>/MCPs --nodejs /usr/bin/node \\
        --served-root /srv/corvis-netboot --env-file <workspace>/.env

The command behind every fleet job of the MCPs hub runner (MCPs board task
b27bd34a, ``scripts/fleet/hub-jobs.json``): its ``fleet-hub`` job runs the
hub lane and each ``fleet-node-<alias>`` one node's lane, every 3 minutes,
one run at a time. ``announce`` is the node lane's one-time check-in, each
node job's boot command, and ``hub-announce`` the hub's, which the board
needs since the queue's close posts a task-naming job's outcome under the
closing runner's label (MCPs board task 2fecad69). The hub lane runs on the
machine holding the control duty, as ``fleet-hub-<that machine's name>``
(MCPs board task c39ee056), so moving the duty gives the new holder its own
identity rather than lending it austinpc's. The ``fleet-sessions``
job, placed on the machine holding the sessions duty, runs the ``sessions``
lane for that machine as ``fleet-sessions-<alias>``, with
``sessions-announce`` its check-in (MCPs board task c39ee056). The
``fleet-netboot`` job, placed on the machine holding the netboot duty, runs
the ``netboot`` lane as ``fleet-netboot-<that machine's name>``, with
``netboot-announce`` its check-in (MCPs board task 83102d9d): the row names
the installed netboot release it arms from, the node binary, the served root
and the workspace env file, and the tick hands them to the agent. A node declaring
``elevated`` also has ``fleet-node-<alias>-elevated``, whose ``elevated``
lane (and its ``elevated-announce``) runs that node's second, elevated
runner under its own identity and log (MCPs board task a98d7083). Until the
hub runner, each of these was a scheduled task of its own,
``API-FleetAgent-3min`` and ``API-FleetNode-<alias>[-elevated]-3min``.

WHY PYTHON AND NOT POWERSHELL (MCPs board task 94ac1c4f). The ticks ran
``powershell.exe`` over a script until 2026-09-29. On the hub,
``powershell.exe`` run as a task action under ``LogonType=S4U`` can stall
before its engine loads, and it never exits: on 2026-09-28 diphtheria's tick
hung 16 minutes and sedona's 7, and under IgnoreNew each hang refused every
later tick of its node (0x800710E0) while the queue grew. A native binary
under the same principal does not stall; the hpc-wake pump moved to Python
for this reason on 2026-09-09. The action is ``poetry.exe``, itself a native
launcher, running this module in the fleet package's own environment.

WHAT A TICK DOES, which is what ``FleetTick.ps1`` did:

* reads the machine's credentials from the hpc-wake pump's untracked
  ``runs/env.ps1`` with :mod:`platform_core.env_assignments`, the parser
  that pump shares, and hands them to the agent's environment and to
  nothing else. A missing key surfaces as the agent's own named refusal;
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
and the scheduler's last result shows something real.

THE DEADLINES NEST. The agent's wall is one minute inside the tasks'
40-minute ExecutionTimeLimit (:data:`fleet.cli.rolled.AGENT_WALL_SECONDS`);
this tick's is half a minute past the agent's, so a stuck agent is ended and
logged by the launcher, a stuck launcher is ended and logged here, and the
scheduler's limit ends only a tick that escaped both.
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
from platform_core.env_assignments import parse_env_assignments
from platform_core.members import find_member

from fleet.cli import agent as agent_cli
from fleet.cli import rolled as rolled_cli
from fleet.core import _test_hooks, names

API_ROOT_FLAG: Final = "--api-root"
LOG_DIRECTORY_FLAG: Final = "--log-directory"
LANE_FLAG: Final = "--lane"
NODE_FLAG: Final = "--node"
_FLAGS: Final = (
    API_ROOT_FLAG,
    LOG_DIRECTORY_FLAG,
    LANE_FLAG,
    NODE_FLAG,
    *agent_cli.NETBOOT_FLAGS,
)


class Lane(StrEnum):
    """The tick an MCPs hub runner job runs; each value is its ``--lane`` word."""

    HUB = "hub"
    HUB_ANNOUNCE = "hub-announce"
    SESSIONS = "sessions"
    SESSIONS_ANNOUNCE = "sessions-announce"
    NETBOOT = "netboot"
    NETBOOT_ANNOUNCE = "netboot-announce"
    NODE = "node"
    ANNOUNCE = "announce"
    ELEVATED = "elevated"
    ELEVATED_ANNOUNCE = "elevated-announce"


#: The lanes the hub runner serves: its tick and its one-time check-in.
_HUB_LANES: Final[frozenset[Lane]] = frozenset({Lane.HUB, Lane.HUB_ANNOUNCE})

#: The lanes the sessions runner serves, on the machine holding the
#: sessions duty: its tick and its one-time check-in.
_SESSIONS_LANES: Final[frozenset[Lane]] = frozenset({Lane.SESSIONS, Lane.SESSIONS_ANNOUNCE})

#: The lanes the netboot runner serves, on the machine holding the netboot
#: duty: its tick and its one-time check-in.
_NETBOOT_LANES: Final[frozenset[Lane]] = frozenset({Lane.NETBOOT, Lane.NETBOOT_ANNOUNCE})

#: The node lanes that run a node's ELEVATED runner (MCPs board task
#: a98d7083): its own identity and log, claiming only the jobs requiring the
#: ``elevated`` tag.
_ELEVATED_LANES: Final[frozenset[Lane]] = frozenset({Lane.ELEVATED, Lane.ELEVATED_ANNOUNCE})

#: The lanes that post the runner's check-in and claim nothing.
_ANNOUNCE_LANES: Final[frozenset[Lane]] = frozenset(
    {
        Lane.HUB_ANNOUNCE,
        Lane.SESSIONS_ANNOUNCE,
        Lane.NETBOOT_ANNOUNCE,
        Lane.ANNOUNCE,
        Lane.ELEVATED_ANNOUNCE,
    }
)

#: The namespace a hub, sessions or netboot runner's session id is derived in.
SESSIONS_NAMESPACE: Final = uuid.NAMESPACE_URL


def hub_identity(alias: str) -> tuple[str, str]:
    """The label and session id a machine's hub runner acts as.

    Derived from the alias as a sessions runner's are: one stable session
    per machine, so the control duty moving to another machine gives that
    machine its own label for good, ``fleet-hub-<alias>``. Until MCPs board
    task c39ee056 the one hub was austinpc, whose fixed pair,
    ``fleet-runner-austinpc`` and session a850f688, is bound for good and so
    is not reused.

    Args:
        alias: The machine's registry name.

    Returns:
        ``fleet-hub-<alias>`` and the version-5 UUID of
        ``fleet-agent/hub/<alias>`` in :data:`SESSIONS_NAMESPACE`.
    """
    return (
        f"fleet-hub-{alias}",
        str(uuid.uuid5(SESSIONS_NAMESPACE, f"fleet-agent/hub/{alias}")),
    )


def sessions_identity(alias: str) -> tuple[str, str]:
    """The label and session id a machine's sessions runner acts as.

    Derived from the alias as a node runner's are
    (:func:`fleet.cli.node_agent.node_identity`): one stable session per
    machine, so its check-in binds one label to one session for good, and
    the duty moving to another machine gives that machine its own.

    Args:
        alias: The machine's registry name.

    Returns:
        ``fleet-sessions-<alias>`` and the version-5 UUID of
        ``fleet-agent/sessions/<alias>`` in :data:`SESSIONS_NAMESPACE`.
    """
    return (
        f"fleet-sessions-{alias}",
        str(uuid.uuid5(SESSIONS_NAMESPACE, f"fleet-agent/sessions/{alias}")),
    )


def netboot_identity(alias: str) -> tuple[str, str]:
    """The label and session id a machine's netboot runner acts as.

    Derived from the alias as a hub runner's are, so the netboot duty moving
    to another machine gives that machine its own label for good.

    Args:
        alias: The machine's registry name.

    Returns:
        ``fleet-netboot-<alias>`` and the version-5 UUID of
        ``fleet-agent/netboot/<alias>`` in :data:`SESSIONS_NAMESPACE`.
    """
    return (
        f"fleet-netboot-{alias}",
        str(uuid.uuid5(SESSIONS_NAMESPACE, f"fleet-agent/netboot/{alias}")),
    )


#: The credentials file, relative to the API checkout.
ENV_FILE: Final = pathlib.PurePosixPath("tools/hpc-wake/runs/env.ps1")

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
            ``fleet-netboot-<alias>`` for the netboot runner and
            ``fleet-node-<alias>`` for a node, each shared by its announce.
        header: What the ``TICK START`` line names after its time.
    """

    arguments: tuple[str, ...]
    stem: str
    header: str


def _netboot_arguments(lane: Lane, netboot: Mapping[str, str]) -> tuple[str, ...]:
    """The flags a netboot tick hands the agent, and nothing for any other.

    Args:
        lane: Which tick this is.
        netboot: The tick's own values of :data:`fleet.cli.agent.NETBOOT_FLAGS`.

    Returns:
        Each flag and its value, in that order, for the netboot lane; an
        empty tuple for every other lane.

    Raises:
        ValueError: ``FLEET_TICK_USAGE`` when the netboot lane lacks one of
            them or any other lane is given one.
    """
    if lane is Lane.NETBOOT:
        missing = [flag for flag in agent_cli.NETBOOT_FLAGS if flag not in netboot]
        if missing:
            raise ValueError(f"FLEET_TICK_USAGE: the {lane.value} lane needs {', '.join(missing)}")
        return tuple(token for flag in agent_cli.NETBOOT_FLAGS for token in (flag, netboot[flag]))
    if netboot:
        raise ValueError(
            f"FLEET_TICK_USAGE: the {lane.value} lane takes no {', '.join(netboot)}; "
            f"only the {Lane.NETBOOT.value} lane arms"
        )
    return ()


def plan_tick(
    api_root: pathlib.Path, lane: Lane, node: str | None, netboot: Mapping[str, str]
) -> TickPlan:
    """Plan one lane's tick.

    Args:
        api_root: The API checkout.
        lane: Which tick this is.
        node: The machine's alias for the node and sessions lanes; None for
            the hub and netboot lanes.
        netboot: The tick's own values of :data:`fleet.cli.agent.NETBOOT_FLAGS`,
            which the netboot lane hands the agent and every other refuses.

    Returns:
        The launcher's command line, the log stem and the record's header.

    Raises:
        ValueError: ``FLEET_TICK_USAGE`` when a hub or netboot lane names a
            node, a node or sessions lane names none, or the netboot flags
            are missing from the netboot lane or given to another.
    """
    repo = str(api_root)
    announce = ("--announce",) if lane in _ANNOUNCE_LANES else ()
    arms = _netboot_arguments(lane, netboot)
    if lane in _NETBOOT_LANES:
        if node is not None:
            raise ValueError(f"FLEET_TICK_USAGE: the {lane.value} lane takes no {NODE_FLAG}")
        label, session = netboot_identity(_test_hooks.hostname())
        return TickPlan(
            arguments=(
                rolled_cli.REPO_ROOT_FLAG,
                repo,
                rolled_cli.AGENT_FLAG,
                "fleet-agent",
                rolled_cli.SEPARATOR,
                "--agent",
                label,
                "--session",
                session,
                "--repo-root",
                repo,
                "--lane",
                "netboot",
                *arms,
                *announce,
            ),
            stem=label,
            header=lane.value,
        )
    if lane in _HUB_LANES:
        if node is not None:
            raise ValueError(f"FLEET_TICK_USAGE: the {lane.value} lane takes no {NODE_FLAG}")
        label, session = hub_identity(_test_hooks.hostname())
        hub = (
            rolled_cli.REPO_ROOT_FLAG,
            repo,
            rolled_cli.AGENT_FLAG,
            "fleet-agent",
            rolled_cli.SEPARATOR,
            "--agent",
            label,
            "--session",
            session,
            "--repo-root",
            repo,
            "--lane",
            "hub",
        )
        if lane is Lane.HUB_ANNOUNCE:
            return TickPlan(arguments=(*hub, *announce), stem="fleet-agent", header=lane.value)
        mcps_root = api_root.parent / "MCPs"
        return TickPlan(
            arguments=(
                *hub,
                "--mcps-root",
                str(mcps_root),
                "--registry",
                str(mcps_root / "fleet-mcp" / "fleet-nodes.json"),
            ),
            stem="fleet-agent",
            header=lane.value,
        )
    if node is None:
        raise ValueError(f"FLEET_TICK_USAGE: the {lane.value} lane needs {NODE_FLAG}")
    if lane in _SESSIONS_LANES:
        label, session = sessions_identity(node)
        return TickPlan(
            arguments=(
                rolled_cli.REPO_ROOT_FLAG,
                repo,
                rolled_cli.AGENT_FLAG,
                "fleet-agent",
                rolled_cli.SEPARATOR,
                "--agent",
                label,
                "--session",
                session,
                "--repo-root",
                repo,
                "--lane",
                "sessions",
                "--node",
                node,
                *announce,
            ),
            stem=label,
            header=f"{lane.value} {node}",
        )
    elevated = lane in _ELEVATED_LANES
    runner = names.runner_name(node, elevated=elevated)
    return TickPlan(
        arguments=(
            rolled_cli.REPO_ROOT_FLAG,
            repo,
            rolled_cli.AGENT_FLAG,
            "fleet-node-agent",
            rolled_cli.SEPARATOR,
            "--node",
            node,
            *(("--elevated",) if elevated else ()),
            *announce,
        ),
        stem=f"fleet-node-{runner}",
        header=f"{lane.value} {runner}",
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


def main(argv: Sequence[str]) -> int:
    """Run one tick and append its record to the day's log.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        The agent's exit status, as the launcher reports it.

    Raises:
        ValueError: ``FLEET_TICK_USAGE`` for a malformed command line.
        AppError: ``CONFIG_ERROR`` when the credentials file holds a line
            that is not a plain assignment.
        OSError: When the credentials file or the log directory cannot be
            read or written.
    """
    parsed = cli_args.parse_single_flags(argv, _FLAGS)
    api_root = pathlib.Path(cli_args.require_flag(parsed, API_ROOT_FLAG))
    log_directory = pathlib.Path(cli_args.require_flag(parsed, LOG_DIRECTORY_FLAG))
    lane = require_lane(cli_args.require_flag(parsed, LANE_FLAG))
    netboot = {flag: parsed[flag] for flag in agent_cli.NETBOOT_FLAGS if flag in parsed}
    plan = plan_tick(api_root, lane, parsed.get(NODE_FLAG), netboot)
    env_file = api_root / ENV_FILE
    credentials = parse_env_assignments(_test_hooks.read_text(env_file), source=str(env_file))

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
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "API_ROOT_FLAG",
    "ENV_FILE",
    "LANE_FLAG",
    "LOG_DIRECTORY_FLAG",
    "NODE_FLAG",
    "RETENTION_DAYS",
    "SESSIONS_NAMESPACE",
    "TICK_WALL_SECONDS",
    "Lane",
    "TickPlan",
    "entrypoint",
    "hub_identity",
    "main",
    "netboot_identity",
    "plan_tick",
    "remove_stale_logs",
    "require_lane",
    "sessions_identity",
]


# Without this, `python -m fleet.cli.tick` imports the module, runs nothing
# and exits 0, and a scheduled tick that ran no agent would read as a clean
# one.
if __name__ == "__main__":
    entrypoint()
