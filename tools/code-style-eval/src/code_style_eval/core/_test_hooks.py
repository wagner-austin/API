"""Internal hooks for the evaluation core.

Production code binds these to real implementations at import. Tests assign a
fake and call reset() afterwards. No conditionals: call the hook directly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class CompletedProcessProto(Protocol):
    """Protocol for the part of a completed process this package reads."""

    returncode: int
    stdout: str
    stderr: str


class RunCheckerProto(Protocol):
    """Protocol for running one checker over one directory."""

    def __call__(self, command: tuple[str, ...], cwd: Path) -> CompletedProcessProto:
        """Run a checker and capture its output.

        Args:
            command: Argv of the checker, already composed.
            cwd: Directory to run it in.

        Returns:
            The finished process.
        """
        ...


def _default_run_checker(command: tuple[str, ...], cwd: Path) -> CompletedProcessProto:
    """Production implementation for running a checker.

    ``check=False`` because a non-zero exit IS the measurement here. A
    checker that reports findings is doing its job, and raising on it would
    turn every failing generation into a crashed sweep.

    Args:
        command: Argv of the checker.
        cwd: Directory to run it in.

    Returns:
        The finished process, whatever its exit status.
    """
    return subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


class Hooks:
    """Container for evaluation hooks, each bound to its real implementation."""

    run_checker: RunCheckerProto = _default_run_checker

    @classmethod
    def reset(cls) -> None:
        """Restore every hook to its real implementation."""
        reset_hooks()


def reset_hooks() -> None:
    """Restore every hook to the production implementation it is bound to."""
    Hooks.run_checker = _default_run_checker


__all__ = [
    "CompletedProcessProto",
    "Hooks",
    "RunCheckerProto",
    "reset_hooks",
]
