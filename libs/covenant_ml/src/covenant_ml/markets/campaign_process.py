"""Real subprocess boundary for durable prospective acquisition attempts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class CommandRunner(Protocol):
    """Launch one CLI process and return its exit status with a durable log."""

    def __call__(self, command: tuple[str, ...], directory: Path, log: Path) -> int: ...


def run_command(command: tuple[str, ...], directory: Path, log: Path) -> int:
    """Execute without a shell, preserving combined output and a five-minute limit.

    Args:
        command: Exact argument vector, including executable.
        directory: Package working directory.
        log: New log destination.

    Returns:
        Actual process return code, including nonzero failure.

    Raises:
        OSError: Process launch or log persistence fails.
        TimeoutExpired: Child exceeds five minutes; its attempt remains started.
    """
    with log.open("xb") as output:
        completed = subprocess.run(
            command,
            cwd=directory,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    return completed.returncode


__all__ = ["CommandRunner", "run_command"]
