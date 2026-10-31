"""Real process boundary for the database client, bounded in time and file size."""

from __future__ import annotations

import resource
import subprocess
from pathlib import Path
from typing import Protocol


class BoundedRunner(Protocol):
    """Run one command without a shell and return its exit status."""

    def __call__(
        self, command: tuple[str, ...], directory: Path, *, seconds: int, max_file_bytes: int
    ) -> int: ...


def run_bounded(
    command: tuple[str, ...], directory: Path, *, seconds: int, max_file_bytes: int
) -> int:
    """Execute with a wall deadline and a kernel limit on every file the child writes.

    Standard input and output are closed; diagnostics stay on this process's
    standard error, which the native launcher keeps private to the attempt.

    Args:
        command: Exact argument vector, including the executable.
        directory: Working directory for every relative path the child writes.
        seconds: Positive wall-clock limit.
        max_file_bytes: ``RLIMIT_FSIZE`` applied in the child before exec; a
            larger write terminates it with ``SIGXFSZ``.

    Returns:
        The child's return code: negative when a signal terminated it.

    Raises:
        OSError: The process cannot be started.
        TimeoutExpired: The deadline passed; the child has been killed.
    """

    def limit() -> None:
        """Apply the file-size limit inside the child only."""
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))

    completed = subprocess.run(
        command,
        cwd=directory,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        timeout=seconds,
        preexec_fn=limit,
        check=False,
    )
    return completed.returncode


__all__ = ["BoundedRunner", "run_bounded"]
