"""Subprocess runner for Kohya_ss training.

This module handles subprocess execution for the Kohya_ss training scripts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import _test_hooks


class SubprocessResultImpl:
    """Implementation of SubprocessResult protocol."""

    _returncode: int
    _stdout: str
    _stderr: str

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        """Initialize subprocess result.

        Args:
            returncode: Return code from subprocess.
            stdout: Standard output.
            stderr: Standard error.
        """
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    @property
    def returncode(self) -> int:
        """Return code from subprocess."""
        return self._returncode

    @property
    def stdout(self) -> str:
        """Standard output from subprocess."""
        return self._stdout

    @property
    def stderr(self) -> str:
        """Standard error from subprocess."""
        return self._stderr


def run_subprocess(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
) -> _test_hooks.SubprocessResult:
    """Run a subprocess.

    Uses the subprocess_runner hook if set, otherwise runs real subprocess.

    Args:
        args: Command and arguments.
        cwd: Working directory.
        timeout: Timeout in seconds.

    Returns:
        Subprocess result with returncode, stdout, stderr.
    """
    if _test_hooks.Hooks.subprocess_runner is not None:
        return _test_hooks.Hooks.subprocess_runner(args, cwd=cwd, timeout=timeout)

    cwd_str = str(cwd) if cwd is not None else None
    result = subprocess.run(
        args,
        cwd=cwd_str,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )
    return SubprocessResultImpl(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


__all__ = [
    "SubprocessResultImpl",
    "run_subprocess",
]
