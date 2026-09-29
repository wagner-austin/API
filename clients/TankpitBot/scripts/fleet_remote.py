"""Reach sedona, the game host, from the hub: its address and the checked command runner.

:mod:`scripts.fleet_host` stages and runs the fleet there and
:mod:`scripts.fleet_gate` proves a bot plays once it is up. Both send their
commands through here, so a failed command is refused the same way,
naming its code, the command, its exit code and its standard error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from scripts import _test_hooks as script_hooks

#: sedona's ssh destination, on the tailnet.
SEDONA_SSH: Final[str] = "austi@100.95.76.122"


class FleetHostError(Exception):
    """A step of operating the fleet on sedona failed; the message leads with its code."""


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


def on_sedona(command: str) -> list[str]:
    """An ssh command line that runs one PowerShell command on sedona.

    Args:
        command: The PowerShell command.

    Returns:
        The argv.
    """
    return ["ssh", SEDONA_SSH, "powershell", "-NoProfile", "-Command", command]


__all__ = [
    "SEDONA_SSH",
    "FleetHostError",
    "on_sedona",
    "run_checked",
]
