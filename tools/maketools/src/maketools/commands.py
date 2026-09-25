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
        timeout_seconds: int,
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
            timeout_seconds: Wall-clock bound on the child.

                REQUIRED, AND NAMED BY THE CALLER RATHER THAN DEFAULTED,
                because the children this runs are not alike: a ``git
                config`` answers instantly and a fanned-out ``make check``
                is hours, and one number covering both would be sized for
                the longer and would stop bounding the shorter. A default
                is the value a new call site reaches for without deciding,
                which is how the unbounded ones got here.

                SHARING A TERMINAL IS NOT A BOUND. It is tempting to think
                an inherited child is bounded by the person watching it, and
                that is false wherever it matters: these commands run
                unattended under the fleet's scheduled tasks, where a wedged
                child holds its lease to expiry and reports nothing (board
                tasks 41ac6ed2, 35940277 and 0d891468).

        Returns:
            The exit status; a failure is the answer, not an exception.

        Raises:
            subprocess.TimeoutExpired: When the child outlives the bound.
                Raised rather than folded into the exit status: an
                uncaptured child has no output to return alongside a
                number, and a caller reading one would not know the run
                had been cut short.
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
