"""CLI: one scheduled fleet tick, run by Task Scheduler as a native process.

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

The action behind every fleet task on the hub: ``API-FleetAgent-3min`` runs
the hub lane and each ``API-FleetNode-<alias>-3min`` one node's lane
(``tools/fleet/scripts/register-*.ps1``). ``announce`` is the node lane's
one-time check-in, run by the registration, and ``hub-announce`` the hub
runner's, which the board needs since the queue's close posts a task-naming
job's outcome under the closing runner's label (MCPs board task 2fecad69).
A node declaring ``elevated`` also has
``API-FleetNode-<alias>-elevated-3min``, whose ``elevated`` lane (and its
``elevated-announce``) runs that node's second, elevated runner under its
own identity and log (MCPs board task a98d7083).

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
from collections.abc import Sequence
from enum import StrEnum
from typing import Final, TypedDict

from platform_core import cli_args
from platform_core.env_assignments import parse_env_assignments
from platform_core.members import find_member

from fleet.cli import rolled as rolled_cli
from fleet.core import _test_hooks, names

API_ROOT_FLAG: Final = "--api-root"
LOG_DIRECTORY_FLAG: Final = "--log-directory"
LANE_FLAG: Final = "--lane"
NODE_FLAG: Final = "--node"
_FLAGS: Final = (API_ROOT_FLAG, LOG_DIRECTORY_FLAG, LANE_FLAG, NODE_FLAG)


class Lane(StrEnum):
    """The tick a scheduled task runs; each value is its ``--lane`` word."""

    HUB = "hub"
    HUB_ANNOUNCE = "hub-announce"
    NODE = "node"
    ANNOUNCE = "announce"
    ELEVATED = "elevated"
    ELEVATED_ANNOUNCE = "elevated-announce"


#: The lanes the hub runner serves: its tick and its one-time check-in.
_HUB_LANES: Final[frozenset[Lane]] = frozenset({Lane.HUB, Lane.HUB_ANNOUNCE})

#: The node lanes that run a node's ELEVATED runner (MCPs board task
#: a98d7083): its own identity and log, claiming only the jobs requiring the
#: ``elevated`` tag.
_ELEVATED_LANES: Final[frozenset[Lane]] = frozenset({Lane.ELEVATED, Lane.ELEVATED_ANNOUNCE})

#: The node lanes that post the runner's check-in and claim nothing.
_ANNOUNCE_LANES: Final[frozenset[Lane]] = frozenset({Lane.ANNOUNCE, Lane.ELEVATED_ANNOUNCE})

#: The hub runner's board identity: its label and the session it posts as.
HUB_AGENT: Final = "fleet-runner-austinpc"
HUB_SESSION: Final = "a850f688-f98d-415c-a244-e993226ca2fc"

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
            ``fleet-node-<alias>`` for a node, whose announce shares it.
        header: What the ``TICK START`` line names after its time.
    """

    arguments: tuple[str, ...]
    stem: str
    header: str


def plan_tick(api_root: pathlib.Path, lane: Lane, node: str | None) -> TickPlan:
    """Plan one lane's tick.

    Args:
        api_root: The API checkout.
        lane: Which tick this is.
        node: The node's alias for the node lanes; None for the hub.

    Returns:
        The launcher's command line, the log stem and the record's header.

    Raises:
        ValueError: ``FLEET_TICK_USAGE`` when a hub lane names a node or a
            node lane names none.
    """
    repo = str(api_root)
    if lane in _HUB_LANES:
        if node is not None:
            raise ValueError(f"FLEET_TICK_USAGE: the {lane.value} lane takes no {NODE_FLAG}")
        hub = (
            rolled_cli.REPO_ROOT_FLAG,
            repo,
            rolled_cli.AGENT_FLAG,
            "fleet-agent",
            rolled_cli.SEPARATOR,
            "--agent",
            HUB_AGENT,
            "--session",
            HUB_SESSION,
            "--repo-root",
            repo,
        )
        if lane is Lane.HUB_ANNOUNCE:
            return TickPlan(arguments=(*hub, "--announce"), stem="fleet-agent", header=lane.value)
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
    announce = ("--announce",) if lane in _ANNOUNCE_LANES else ()
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
    plan = plan_tick(api_root, lane, parsed.get(NODE_FLAG))
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
    "HUB_AGENT",
    "HUB_SESSION",
    "LANE_FLAG",
    "LOG_DIRECTORY_FLAG",
    "NODE_FLAG",
    "RETENTION_DAYS",
    "TICK_WALL_SECONDS",
    "Lane",
    "TickPlan",
    "entrypoint",
    "main",
    "plan_tick",
    "remove_stale_logs",
    "require_lane",
]


# Without this, `python -m fleet.cli.tick` imports the module, runs nothing
# and exits 0, and a scheduled tick that ran no agent would read as a clean
# one.
if __name__ == "__main__":
    entrypoint()
