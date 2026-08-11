"""Dependency-injection hooks for the harness package.

Every non-pure operation the harness performs is reached through a
module-level symbol here, bound to its real implementation at import time.
Callers always invoke the hook directly — there is no ``if testing`` branch
anywhere, so the production and test code paths are byte-identical in shape.

Tests rebind a symbol to a fake and restore it afterwards. This module is
private to the package; consumers outside it must not import from here.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from socketserver import BaseServer
from typing import Protocol


class ReadTextLinesProto(Protocol):
    """Read a UTF-8 text file and return its lines."""

    def __call__(self, path: Path) -> tuple[str, ...]:
        """Read every line of a text file.

        Args:
            path: File to read.

        Returns:
            The file's lines with trailing newlines removed, in file order.

        Raises:
            OSError: When the file cannot be read.
            UnicodeDecodeError: When the file is not valid UTF-8.
        """
        ...


class WriteLineProto(Protocol):
    """Emit one line to the process's standard output."""

    def __call__(self, text: str) -> None:
        """Write one line.

        Args:
            text: Line content, without a trailing newline.
        """
        ...


class PathExistsProto(Protocol):
    """Report whether a filesystem path exists."""

    def __call__(self, path: Path) -> bool:
        """Test one path for existence.

        Args:
            path: Path to test.

        Returns:
            ``True`` when the path exists, ``False`` otherwise.
        """
        ...


class ReadArgvProto(Protocol):
    """Read this process's command-line arguments, excluding the program name."""

    def __call__(self) -> list[str]:
        """Return the argument list.

        Returns:
            Arguments after the program name, in order.
        """
        ...


class RunCaptureProto(Protocol):
    """Run a child process to completion and capture everything it printed."""

    def __call__(self, argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
        """Run one command.

        Args:
            argv: Argument vector, program first.

        Returns:
            The child's exit status and its combined output lines, in order.

        Raises:
            OSError: When the program cannot be started.
        """
        ...


class ListNamesProto(Protocol):
    """List the immediate entry names of a directory."""

    def __call__(self, path: Path) -> tuple[str, ...]:
        """List one directory.

        Args:
            path: Directory to list.

        Returns:
            Entry names, sorted, without their parent path.

        Raises:
            OSError: When the directory cannot be read.
        """
        ...


class CopyEntryProto(Protocol):
    """Copy one file or directory tree into a destination directory."""

    def __call__(self, source: Path, destination: Path) -> None:
        """Copy one entry.

        Args:
            source: File or directory to copy.
            destination: Directory to copy it into, which must exist.

        Raises:
            OSError: When the copy fails.
        """
        ...


class MakeDirsProto(Protocol):
    """Create a directory and every missing parent."""

    def __call__(self, path: Path) -> None:
        """Create one directory.

        Args:
            path: Directory to create. Existing directories are left alone.

        Raises:
            OSError: When the directory cannot be created.
        """
        ...


class WriteTextLinesProto(Protocol):
    """Write lines to a UTF-8 text file, replacing any existing content."""

    def __call__(self, path: Path, lines: Sequence[str]) -> None:
        """Write one text file.

        Args:
            path: File to write.
            lines: Line contents, without trailing newlines.

        Raises:
            OSError: When the file cannot be written.
        """
        ...


class SpawnedMatchProto(Protocol):
    """The child-process surface the fleet manager consumes."""

    @property
    def pid(self) -> int:
        """The child's process id."""
        ...

    def poll(self) -> int | None:
        """Return the exit code, or ``None`` while the child runs."""
        ...


class SpawnMatchProto(Protocol):
    """Start one detached match process with its transcript on disk."""

    def __call__(self, argv: Sequence[str], transcript: Path) -> SpawnedMatchProto:
        """Spawn one match.

        Args:
            argv: Argument vector, program first.
            transcript: File that receives the child's combined output.

        Returns:
            The spawned process handle.

        Raises:
            OSError: When the program cannot be started or the
                transcript cannot be opened.
        """
        ...


class ServeForeverProto(Protocol):
    """Run an HTTP server's accept loop until the process is interrupted."""

    def __call__(self, server: BaseServer) -> None:
        """Serve until interrupted.

        Args:
            server: The bound server to run.
        """
        ...


class KillTreeProto(Protocol):
    """Terminate a process and every child it spawned."""

    def __call__(self, pid: int) -> None:
        """Kill one process tree.

        Args:
            pid: Root process id of the tree.
        """
        ...


def _run_capture_impl(argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
    """Production implementation of :class:`RunCaptureProto`.

    Streams are merged because the caller wants one transcript of the match in
    the order it happened, and the launcher writes progress to one stream and
    the planner writes its scorecard to the other.

    Undecodable bytes are replaced rather than raising. This decodes a console
    stream produced by a third-party game process and its launcher, not a data
    format this package defines; the alternative is discarding a match that ran
    for seven minutes because one cosmetic byte was not UTF-8.

    Args:
        argv: Argument vector, program first.

    Returns:
        The child's exit status and its combined output lines, in order.

    Raises:
        OSError: When the program cannot be started.
    """
    finished = subprocess.run(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        check=False,
        # The planner and the engine it spawns are latency-sensitive
        # tenants on a machine that also runs Docker's VM and virus
        # scans; the child inherits this class, so a co-tenant spike
        # deschedules the batch work instead of the sample stream
        # (log 2026-08-10).
        creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS,
    )
    return finished.returncode, tuple(finished.stdout.splitlines())


def _list_names_impl(path: Path) -> tuple[str, ...]:
    """Production implementation of :class:`ListNamesProto`.

    Args:
        path: Directory to list.

    Returns:
        Entry names, sorted, without their parent path.

    Raises:
        OSError: When the directory cannot be read.
    """
    return tuple(sorted(entry.name for entry in path.iterdir()))


def _copy_entry_impl(source: Path, destination: Path) -> None:
    """Production implementation of :class:`CopyEntryProto`.

    Args:
        source: File or directory to copy.
        destination: Directory to copy it into, which must exist.

    Raises:
        OSError: When the copy fails.
    """
    target = destination / source.name
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
        return
    shutil.copy2(source, target)


def _make_dirs_impl(path: Path) -> None:
    """Production implementation of :class:`MakeDirsProto`.

    Args:
        path: Directory to create. Existing directories are left alone.

    Raises:
        OSError: When the directory cannot be created.
    """
    path.mkdir(parents=True, exist_ok=True)


def _write_text_lines_impl(path: Path, lines: Sequence[str]) -> None:
    """Production implementation of :class:`WriteTextLinesProto`.

    Args:
        path: File to write.
        lines: Line contents, without trailing newlines.

    Raises:
        OSError: When the file cannot be written.
    """
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _spawn_match_impl(argv: Sequence[str], transcript: Path) -> SpawnedMatchProto:
    """Production implementation of :class:`SpawnMatchProto`.

    The transcript file handle is closed in the parent immediately after
    the spawn — the child keeps its inherited handle, so the report lines
    the planner prints at match end still land in the file.

    Args:
        argv: Argument vector, program first.
        transcript: File that receives the child's combined output.

    Returns:
        The spawned process handle.

    Raises:
        OSError: When the program cannot be started or the transcript
            cannot be opened.
    """
    transcript.parent.mkdir(parents=True, exist_ok=True)
    with transcript.open("ab") as sink:
        return subprocess.Popen(
            list(argv),
            stdout=sink,
            stderr=subprocess.STDOUT,
            # Same tenancy rule as run_capture: the match tree under this
            # spawn inherits the class and outranks batch co-tenants.
            creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS,
        )


def _serve_forever_impl(server: BaseServer) -> None:
    """Production implementation of :class:`ServeForeverProto`.

    Args:
        server: The bound server to run.
    """
    server.serve_forever()


def _kill_tree_impl(pid: int) -> None:
    """Production implementation of :class:`KillTreeProto`.

    ``taskkill /T`` is the Windows way to fell the whole tree — the fleet
    spawns ``make``, which runs PowerShell, which runs the game JVM and
    the planner; killing only the root would orphan the match.

    Args:
        pid: Root process id of the tree.
    """
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _read_argv_impl() -> list[str]:
    """Production implementation of :class:`ReadArgvProto`.

    Returns:
        ``sys.argv`` after the program name.
    """
    return list(sys.argv[1:])


def _read_text_lines_impl(path: Path) -> tuple[str, ...]:
    """Production implementation of :class:`ReadTextLinesProto`.

    Decoding is strict: a log that is not valid UTF-8 is a real problem with
    the launcher's ``-Dfile.encoding`` setting, and silently replacing bad
    bytes would hide it.

    Args:
        path: File to read.

    Returns:
        The file's lines with trailing newlines removed, in file order.

    Raises:
        OSError: When the file cannot be read.
        UnicodeDecodeError: When the file is not valid UTF-8.
    """
    return tuple(path.read_text(encoding="utf-8").splitlines())


def _path_exists_impl(path: Path) -> bool:
    """Production implementation of :class:`PathExistsProto`.

    Args:
        path: Path to test.

    Returns:
        ``True`` when the path exists, ``False`` otherwise.
    """
    return path.exists()


def _write_line_impl(text: str) -> None:
    """Production implementation of :class:`WriteLineProto`.

    Args:
        text: Line content, without a trailing newline.
    """
    sys.stdout.write(f"{text}\n")


copy_entry: CopyEntryProto = _copy_entry_impl
kill_tree: KillTreeProto = _kill_tree_impl
list_names: ListNamesProto = _list_names_impl
spawn_match: SpawnMatchProto = _spawn_match_impl
make_dirs: MakeDirsProto = _make_dirs_impl
path_exists: PathExistsProto = _path_exists_impl
read_argv: ReadArgvProto = _read_argv_impl
read_text_lines: ReadTextLinesProto = _read_text_lines_impl
run_capture: RunCaptureProto = _run_capture_impl
serve_forever: ServeForeverProto = _serve_forever_impl
write_line: WriteLineProto = _write_line_impl
write_text_lines: WriteTextLinesProto = _write_text_lines_impl


__all__ = [
    "CopyEntryProto",
    "KillTreeProto",
    "ListNamesProto",
    "MakeDirsProto",
    "PathExistsProto",
    "ReadArgvProto",
    "ReadTextLinesProto",
    "RunCaptureProto",
    "ServeForeverProto",
    "SpawnMatchProto",
    "SpawnedMatchProto",
    "WriteLineProto",
    "WriteTextLinesProto",
    "copy_entry",
    "kill_tree",
    "list_names",
    "make_dirs",
    "path_exists",
    "read_argv",
    "read_text_lines",
    "run_capture",
    "serve_forever",
    "spawn_match",
    "write_line",
    "write_text_lines",
]
