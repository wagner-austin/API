"""Internal hooks for the evaluation core.

Production code binds these to real implementations at import. Tests assign a
fake and call reset() afterwards. No conditionals: call the hook directly.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol


class CompletedProcessProto(Protocol):
    """Protocol for the part of a completed process this package reads."""

    returncode: int
    stdout: str
    stderr: str


class RunCheckerProto(Protocol):
    """Protocol for running one checker over one directory."""

    def __call__(
        self, command: tuple[str, ...], cwd: Path, env: Mapping[str, str]
    ) -> CompletedProcessProto:
        """Run a checker and capture its output.

        Args:
            command: Argv of the checker, already composed.
            cwd: Directory to run it in.
            env: Environment for the checker process.

        Returns:
            The finished process.
        """
        ...


def _default_run_checker(
    command: tuple[str, ...], cwd: Path, env: Mapping[str, str]
) -> CompletedProcessProto:
    """Production implementation for running a checker.

    ``check=False`` because a non-zero exit IS the measurement here. A
    checker that reports findings is doing its job, and raising on it would
    turn every failing generation into a crashed sweep.

    The encoding is named explicitly, and this is not a detail. ``text=True``
    alone decodes with the LOCALE codec, which on this platform is cp1252,
    while the checkers and the generated files are UTF-8. Worse than being
    wrong, it is wrong SILENTLY: the decode runs on subprocess's internal
    reader thread, so a byte cp1252 cannot map raises there, the exception
    dies with the thread, and ``communicate`` returns None for that stream.
    The caller then holds a ``stdout`` that this module's own protocol
    promises is a ``str``. Observed on a real sweep, where a generated file
    made a checker emit byte 0x81.

    ``errors="replace"`` for the same reason it is not a softening: the
    result is a diagnostic string that goes on to be JSON-encoded, so it has
    to be representable. Replacing one undecodable byte in a message costs a
    character; letting the stream come back None costs the whole run, and
    costs it quietly.

    Args:
        command: Argv of the checker.
        cwd: Directory to run it in.
        env: Environment for the checker process, carrying MYPYPATH.

    Returns:
        The finished process, whatever its exit status. Both streams are
        captured strings, never None.
    """
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
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
