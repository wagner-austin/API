"""Dependency-injection seam for everything impure this package does.

Every impure act -- running a command, reading the clock, sleeping, reading
the process table, killing a process, writing a line -- is reached through a
symbol here, bound to its real implementation at import time. Production
calls the hook directly, so there is no conditional dispatch and no second
code path; a test rebinds the symbol and exercises the same lines.

STANDARD LIBRARY ONLY. The Makefiles launch this package with the SYSTEM
interpreter before poetry has synced anything, so nothing here may import
``platform_core``: the seam is spelled out in full rather than lifted from
``fleet.core._test_hooks``, which is otherwise its model.

THE PLATFORM IS A HOOK, and that is what lets one suite cover both arms of
the launcher on either machine. ``platform`` answers ``sys.platform``; the
launcher branches on it once, and a test rebinds it to walk the arm the
machine it runs on would not take. The two real process-table readers are
bound by a conditional EXPRESSION rather than an ``if`` statement so that
the binding line carries no branch the other platform cannot execute, and
each reader takes its input through a seam (the command runner, a ``/proc``
root) so its parsing is exercised everywhere.
"""

from __future__ import annotations

import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol

from maketools.commands import CommandResult, RunCapturingProtocol, RunInheritingProtocol
from maketools.job import JobApi, kernel32_job_api
from maketools.processes import (
    LINUX_PROC_ROOT,
    ProcessRow,
    linux_process_table,
    windows_process_table,
)


class NowProtocol(Protocol):
    """Reads the wall clock."""

    def __call__(self) -> float:
        """Read it.

        Returns:
            Seconds since the epoch.
        """
        ...


class SleepProtocol(Protocol):
    """Blocks the process."""

    def __call__(self, seconds: float) -> None:
        """Block.

        Args:
            seconds: How long.
        """
        ...


class WriteLineProtocol(Protocol):
    """Writes one line to a stream."""

    def __call__(self, line: str) -> None:
        """Write it, with a newline.

        Args:
            line: The text.
        """
        ...


class EnvironProtocol(Protocol):
    """Reads the process environment."""

    def __call__(self) -> dict[str, str]:
        """Read it.

        Returns:
            A copy; mutating it changes nothing.
        """
        ...


class PlatformProtocol(Protocol):
    """Names the platform."""

    def __call__(self) -> str:
        """Name it.

        Returns:
            ``sys.platform``: ``win32``, ``linux``, ``darwin``.
        """
        ...


class ProcessTableProtocol(Protocol):
    """Snapshots every process this machine will show."""

    def __call__(self) -> Sequence[ProcessRow]:
        """Snapshot it.

        Returns:
            One row per process, in no particular order.
        """
        ...


class KillProtocol(Protocol):
    """Terminates one process."""

    def __call__(self, pid: int) -> None:
        """Terminate it.

        Args:
            pid: The process.

        Raises:
            OSError: When the process cannot be signalled, including when it
                is already gone; the caller decides what that means.
        """
        ...


class ProcessAliveProtocol(Protocol):
    """Reports whether a pid still names a process."""

    def __call__(self, pid: int) -> bool:
        """Report it.

        Args:
            pid: The process.

        Returns:
            True when it is still in the process table.
        """
        ...


class RemoveTreeProtocol(Protocol):
    """Deletes a directory and everything under it."""

    def __call__(self, path: Path) -> None:
        """Delete it.

        Args:
            path: The directory.
        """
        ...


class RemoveFileProtocol(Protocol):
    """Deletes one file."""

    def __call__(self, path: Path) -> None:
        """Delete it.

        Args:
            path: The file.
        """
        ...


class ProcessIdProtocol(Protocol):
    """Reads this process's pid."""

    def __call__(self) -> int:
        """Read it.

        Returns:
            The pid.
        """
        ...


class TokenProtocol(Protocol):
    """Mints a short random token."""

    def __call__(self) -> str:
        """Mint it.

        Returns:
            Eight lowercase hex characters.
        """
        ...


class DrawProtocol(Protocol):
    """Draws one integer from an inclusive range."""

    def __call__(self, low: int, high: int) -> int:
        """Draw it.

        Args:
            low: The smallest value the draw may return.
            high: The largest value the draw may return.

        Returns:
            An integer in ``[low, high]``.
        """
        ...


class JobApiFactoryProtocol(Protocol):
    """Binds the Windows job-object calls."""

    def __call__(self) -> JobApi:
        """Bind them.

        Returns:
            The API.
        """
        ...


class TrackedFilesProtocol(Protocol):
    """Lists the files git tracks matching a pathspec."""

    def __call__(self, repo_root: Path, pathspec: str) -> Sequence[Path]:
        """List them.

        Args:
            repo_root: The repository.
            pathspec: A git pathspec such as ``*Makefile``.

        Returns:
            Absolute paths, in git's order.
        """
        ...


#: Hard ceiling on a CAPTURED child. Only short questions are captured
#: (``poetry run mypy --version``, ``git ls-files``, one process snapshot);
#: a hang there is a broken toolchain, and an operator should see it fail
#: rather than stare at a blank line.
CAPTURE_TIMEOUT_SECONDS: Final[int] = 120

#: The signal :func:`_default_kill` sends. ``SIGKILL`` (9) off Windows,
#: because a wedged xdist worker blocked in ``sys.stdin.readline()`` is the
#: thing being reaped and a catchable signal is a request it will not
#: answer; ``SIGTERM`` on Windows, where ``os.kill`` maps every signal to
#: ``TerminateProcess`` and ``SIGKILL`` is not defined. Spelled as a number
#: rather than ``signal.SIGKILL`` because that name does not exist on
#: Windows and the type checker there would refuse it.
POSIX_KILL_SIGNAL: Final[int] = 9


def _default_run_inheriting(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    new_session: bool,
    timeout_seconds: int,
) -> int:
    """Run a child on this process's terminal.

    Args:
        argv: Executable and arguments.
        cwd: The working directory.
        env: The child's environment.
        new_session: Whether to ``setsid`` the child on POSIX.
        timeout_seconds: Wall-clock bound, named by the caller for the
            reason :class:`~maketools.commands.RunInheritingProtocol` gives.

    Returns:
        The exit status.

    Raises:
        subprocess.TimeoutExpired: When the child outlives the bound.
    """
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(env),
        check=False,
        start_new_session=new_session,
        timeout=timeout_seconds,
    )
    return completed.returncode


def _default_run_capturing(argv: Sequence[str], *, cwd: Path) -> CommandResult:
    """Run a short child and collect its output.

    Args:
        argv: Executable and arguments.
        cwd: The working directory.

    Returns:
        Exit status and streams.
    """
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        check=False,
        capture_output=True,
        timeout=CAPTURE_TIMEOUT_SECONDS,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def _default_now() -> float:
    """Read the real wall clock.

    Returns:
        Seconds since the epoch.
    """
    return time.time()


def _default_sleep(seconds: float) -> None:
    """Block the real process.

    Args:
        seconds: How long.
    """
    time.sleep(seconds)


def _default_write_line(line: str) -> None:
    """Write to standard output.

    Args:
        line: The text.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _default_write_error(line: str) -> None:
    """Write to standard error.

    Args:
        line: The text.
    """
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _default_environ() -> dict[str, str]:
    """Copy the real environment.

    Returns:
        The copy.
    """
    return dict(os.environ)


def _default_platform() -> str:
    """Name the real platform.

    Returns:
        ``sys.platform``.
    """
    return sys.platform


def _windows_table() -> Sequence[ProcessRow]:
    """Snapshot the process table through PowerShell's CIM query.

    Returns:
        The rows.
    """
    return windows_process_table(run_capturing)


def _linux_table() -> Sequence[ProcessRow]:
    """Snapshot the process table from ``/proc``.

    Returns:
        The rows.
    """
    return linux_process_table(LINUX_PROC_ROOT)


def _default_kill(pid: int) -> None:
    """Terminate a real process.

    Args:
        pid: The process.
    """
    os.kill(pid, POSIX_KILL_SIGNAL if sys.platform != "win32" else signal.SIGTERM)


def _default_process_alive(pid: int) -> bool:
    """Report whether a real pid is still in the table.

    Args:
        pid: The process.

    Returns:
        True when a row with that pid exists.
    """
    return any(row["pid"] == pid for row in process_table())


def _default_remove_tree(path: Path) -> None:
    """Delete a real directory tree.

    Args:
        path: The directory.
    """
    shutil.rmtree(path)


def _default_remove_file(path: Path) -> None:
    """Delete a real file.

    Args:
        path: The file.
    """
    path.unlink()


def _default_process_id() -> int:
    """Read the real pid.

    Returns:
        The pid.
    """
    return os.getpid()


def _default_token() -> str:
    """Mint a real random token.

    Returns:
        Eight hex characters.
    """
    return secrets.token_hex(4)


def _default_draw(low: int, high: int) -> int:
    """Draw a real random integer from an inclusive range.

    Args:
        low: The smallest value the draw may return.
        high: The largest value the draw may return.

    Returns:
        An integer in ``[low, high]``.
    """
    return low + secrets.randbelow(high - low + 1)


def _default_tracked_files(repo_root: Path, pathspec: str) -> Sequence[Path]:
    """Ask git for the tracked files matching a pathspec.

    Args:
        repo_root: The repository.
        pathspec: The pathspec.

    Returns:
        Absolute paths, in git's order.

    Raises:
        RuntimeError: When git refuses; a lint that silently examined zero
            files would report a clean tree.
    """
    result = run_capturing(["git", "ls-files", "-z", "--", pathspec], cwd=repo_root)
    if result["returncode"] != 0:
        raise RuntimeError(f"git ls-files failed in {repo_root}: {result['stderr'].strip()}")
    return [repo_root / name for name in result["stdout"].split("\0") if name != ""]


run_inheriting: RunInheritingProtocol = _default_run_inheriting
run_capturing: RunCapturingProtocol = _default_run_capturing
now: NowProtocol = _default_now
sleep: SleepProtocol = _default_sleep
write_line: WriteLineProtocol = _default_write_line
write_error: WriteLineProtocol = _default_write_error
environ: EnvironProtocol = _default_environ
platform: PlatformProtocol = _default_platform
# A conditional expression, not an ``if``: the line has no branch a suite on
# the other platform would leave uncovered, and both readers are exercised
# through their own seams.
_default_process_table: ProcessTableProtocol = (
    _windows_table if sys.platform == "win32" else _linux_table
)
_default_job_api: JobApiFactoryProtocol = kernel32_job_api
process_table: ProcessTableProtocol = _default_process_table
kill: KillProtocol = _default_kill
process_alive: ProcessAliveProtocol = _default_process_alive
remove_tree: RemoveTreeProtocol = _default_remove_tree
remove_file: RemoveFileProtocol = _default_remove_file
process_id: ProcessIdProtocol = _default_process_id
token: TokenProtocol = _default_token
draw: DrawProtocol = _default_draw
tracked_files: TrackedFilesProtocol = _default_tracked_files
job_api: JobApiFactoryProtocol = _default_job_api


__all__ = [
    "CAPTURE_TIMEOUT_SECONDS",
    "POSIX_KILL_SIGNAL",
    "CommandResult",
    "DrawProtocol",
    "EnvironProtocol",
    "JobApiFactoryProtocol",
    "KillProtocol",
    "NowProtocol",
    "PlatformProtocol",
    "ProcessAliveProtocol",
    "ProcessIdProtocol",
    "ProcessTableProtocol",
    "RemoveFileProtocol",
    "RemoveTreeProtocol",
    "RunCapturingProtocol",
    "RunInheritingProtocol",
    "SleepProtocol",
    "TokenProtocol",
    "TrackedFilesProtocol",
    "WriteLineProtocol",
    "draw",
    "environ",
    "job_api",
    "kill",
    "now",
    "platform",
    "process_alive",
    "process_id",
    "process_table",
    "remove_file",
    "remove_tree",
    "run_capturing",
    "run_inheriting",
    "sleep",
    "token",
    "tracked_files",
    "write_error",
    "write_line",
]
