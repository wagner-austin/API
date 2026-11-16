"""Reach the game host: its address, the remote command line and the checked runner.

The game host is demure, the stack server (MCPs board task 99194467). It was
sedona until 2026-10-18 and colossus until colossus was reinstalled as a
fleet worker on 2026-10-30, which left TankPit with no host and this module
naming an account that no longer existed. So the host is named here and
nothing else: its login user and tailnet address are read from API's fleet
roster, ``tools/fleet/registry/machines/<host>.json``, the one place a
machine's address is written, and its home directory is that user's.
:mod:`scripts.fleet_host`, :mod:`scripts.fleet_release`,
:mod:`scripts.fleet_gate` and :mod:`scripts.sim_host` send their commands
through here, so a failed command is refused the same way, naming its code,
the command, its exit code and its standard error.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Final

from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_str

from scripts import _test_hooks as script_hooks

#: The game host, by its name in API's fleet roster.
GAME_HOST: Final[str] = "demure"

#: API's fleet roster: one file per machine.
ROSTER_DIR: Final[Path] = (
    Path(__file__).resolve().parents[3] / "tools" / "fleet" / "registry" / "machines"
)

_MACHINE = narrow_json_to_dict(
    load_json_str((ROSTER_DIR / f"{GAME_HOST}.json").read_text(encoding="utf-8"))
)

#: The game host's login user, from its roster file.
GAME_HOST_USER: Final[str] = require_str(_MACHINE, "user")

#: The game host's ssh destination: its user at its tailnet address.
GAME_HOST_SSH: Final[str] = f"{GAME_HOST_USER}@{require_str(_MACHINE, 'tailnetIp')}"

#: The game host's home directory, under which every TankPit path lives.
GAME_HOST_HOME: Final[str] = f"/home/{GAME_HOST_USER}"


class FleetHostError(Exception):
    """A step of operating the game host failed; the message leads with its code."""


def run_checked(argv: list[str], cwd: Path, code: str) -> str:
    """Run a command and insist it succeeded.

    Args:
        argv: The command.
        cwd: Where to run it.
        code: The error code a failure is raised with.

    Returns:
        Its standard output.

    Raises:
        FleetHostError: ``code``, naming the command, its exit code and
            its standard error.
    """
    result = script_hooks.run_command(argv, cwd)
    if result["returncode"] != 0:
        raise FleetHostError(
            f"{code}: {' '.join(argv)} exited {result['returncode']}: {result['stderr'].strip()}"
        )
    return result["stdout"]


def on_host(argv: list[str]) -> list[str]:
    """An ssh command line that runs one command on the game host.

    ssh hands the remote shell one string, so the words are quoted for it
    here and arrive as the same words, whatever spaces or quotes they hold.

    Args:
        argv: The command to run there.

    Returns:
        The local argv.
    """
    return ["ssh", GAME_HOST_SSH, shlex.join(argv)]


__all__ = [
    "GAME_HOST",
    "GAME_HOST_HOME",
    "GAME_HOST_SSH",
    "GAME_HOST_USER",
    "ROSTER_DIR",
    "FleetHostError",
    "on_host",
    "run_checked",
]
