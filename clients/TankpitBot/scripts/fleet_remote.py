"""Reach the game host: its address, the remote command line and the checked runner.

The game host is colossus, a Linux machine on the home network the TV is on
(MCPs board task c481c259; the operator, 2026-10-18: "we dont rwally.plan on
ising windows ever again tbh"). Until then it was sedona, where every command
was a PowerShell line read first by cmd.exe and the fleet ran inside Docker
Desktop's VM. :mod:`scripts.fleet_host`, :mod:`scripts.fleet_release`,
:mod:`scripts.fleet_gate` and :mod:`scripts.sim_host` send their commands
through here, so a failed command is refused the same way, naming its code,
the command, its exit code and its standard error.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Final

from scripts import _test_hooks as script_hooks

#: The game host's ssh destination, on the tailnet.
GAME_HOST_SSH: Final[str] = "user@100.111.149.33"

#: The game host's home directory, under which every TankPit path lives.
GAME_HOST_HOME: Final[str] = "/home/user"


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
    "GAME_HOST_HOME",
    "GAME_HOST_SSH",
    "FleetHostError",
    "on_host",
    "run_checked",
]
