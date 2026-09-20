"""The shapes of a child process's run and result.

Held apart from :mod:`maketools._test_hooks` because the process-table
reader takes its command runner as a parameter (so its parsing is testable
on the platform that cannot run the command) and the hooks module imports
the reader; the two would otherwise import each other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict


class CommandResult(TypedDict):
    """What running a captured command produced.

    Attributes:
        returncode: Process exit status.
        stdout: Standard output, decoded as UTF-8.
        stderr: Standard error, decoded as UTF-8.
    """

    returncode: int
    stdout: str
    stderr: str


class RunInheritingProtocol(Protocol):
    """Runs a child that shares this process's terminal."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        new_session: bool,
    ) -> int:
        """Run it to completion.

        Args:
            argv: Executable and arguments, never a shell string.
            cwd: The working directory.
            env: The child's complete environment.
            new_session: Start the child in its own session (POSIX
                ``setsid``), so a signal aimed at this process's group does
                not reach the suite and the suite's group can be named for
                the reaper. Accepted and ignored by the standard library on
                Windows, where the job object does the same work.

        Returns:
            The exit status; a failure is the answer, not an exception.
        """
        ...


class RunCapturingProtocol(Protocol):
    """Runs a short command and collects its output."""

    def __call__(self, argv: Sequence[str], *, cwd: Path) -> CommandResult:
        """Run it to completion.

        Args:
            argv: Executable and arguments, never a shell string.
            cwd: The working directory.

        Returns:
            Exit status and captured streams.
        """
        ...


__all__ = ["CommandResult", "RunCapturingProtocol", "RunInheritingProtocol"]
