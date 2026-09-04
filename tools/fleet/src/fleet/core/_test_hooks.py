"""Dependency-injection seam for the core.

Every impure act -- running a command, reading the clock, touching a file --
is reached through a symbol here, bound to its real implementation at import
time. Production calls the hook directly, so there is no conditional dispatch
and no second code path; a test rebinds the symbol and exercises the same
lines.

THE CLOCK IS A HOOK AND THAT IS NOT INCIDENTAL. A lease has an expiry, so
every question this package answers about whether a resource is free is a
question about the current time. A test that could not control the clock could
only assert that an unexpired lease is unexpired, which is the case that never
breaks. Controlling it is how the expiry boundary gets tested at all.

NOTHING HERE CATCHES. ``subprocess.run`` is called with ``check=False`` and
the return code is inspected explicitly, so a remote failure becomes a typed
:class:`~platform_core.errors.AppError` at the call site that knows what the
command was for, rather than a ``CalledProcessError`` caught and re-raised
somewhere that does not.
"""

from __future__ import annotations

import pathlib
import subprocess
import time
from collections.abc import Sequence
from typing import Protocol

from typing_extensions import TypedDict


class CommandResult(TypedDict):
    """What running a command produced.

    Attributes:
        returncode: Process exit status.
        stdout: Standard output, decoded as UTF-8.
        stderr: Standard error, decoded as UTF-8. Carried because ssh puts
            the reason for a refusal here, and a failure that discards it
            sends the reader to the node to rediscover what happened.
    """

    returncode: int
    stdout: str
    stderr: str


class RunProtocol(Protocol):
    """Runs a local command to completion and collects its output."""

    def __call__(self, argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
        """Run a command.

        Args:
            argv: Executable and arguments. Never a shell string: a project
                path or a node name is arbitrary text, and shell
                interpretation of it would be a defect rather than a feature.
            stdin_bytes: Bytes to write to the process's standard input, or
                None to provide none.

        Returns:
            Exit status and captured streams. A non-zero status is returned
            rather than raised; the caller decides what it means.
        """


class NowProtocol(Protocol):
    """Reads the wall clock in whole seconds since the epoch."""

    def __call__(self) -> int:
        """Read the current time.

        Returns:
            Whole seconds since the epoch. Whole rather than fractional
            because every consumer is a lease boundary measured in seconds,
            and a float would invite comparisons that differ in their last
            bit between two readers of one file.
        """


class ReadTextProtocol(Protocol):
    """Reads a file's whole contents as UTF-8."""

    def __call__(self, path: pathlib.Path) -> str:
        """Read a file.

        Args:
            path: Absolute path to read.

        Returns:
            The file's contents.

        Raises:
            OSError: If the file cannot be read. Propagated rather than
                translated: the three files this package reads are its own
                records, and one being unreadable is an operational fault
                whose own message names the path and the reason.
        """


class AppendTextProtocol(Protocol):
    """Appends one line to an append-only file, creating it if absent."""

    def __call__(self, path: pathlib.Path, line: str) -> None:
        """Append a line.

        Args:
            path: Absolute path to append to.
            line: The line, without a trailing newline; the implementation
                adds it. Taking the line without one is what makes it
                impossible to append two records that share a line.

        Raises:
            OSError: If the file cannot be written.
        """


class ReadBytesProtocol(Protocol):
    """Reads a file's whole contents as bytes."""

    def __call__(self, path: pathlib.Path) -> bytes:
        """Read a file without decoding it.

        Distinct from :class:`ReadTextProtocol` rather than a mode on it,
        because the one caller is reading a gzip archive and the text reader
        decodes as UTF-8 with replacement -- which is right for a diagnostic
        and silently destroys an archive.

        Args:
            path: Absolute path to read.

        Returns:
            The file's contents.

        Raises:
            OSError: If the file cannot be read.
        """


class FileExistsProtocol(Protocol):
    """Reports whether a path names an existing regular file."""

    def __call__(self, path: pathlib.Path) -> bool:
        """Test a path.

        Args:
            path: Absolute path to test.

        Returns:
            True when the path exists and is a regular file. A directory at
            that path is False rather than an error: the caller's next act
            would be to read it, and letting that fail with its own message
            is better than inventing one here.
        """


class WriteTextProtocol(Protocol):
    """Replaces a file's whole contents, creating it if absent."""

    def __call__(self, path: pathlib.Path, text: str) -> None:
        """Write a file.

        Args:
            path: Absolute path to write.
            text: The complete new contents.

        Raises:
            OSError: If the file cannot be written.
        """


def _default_run(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
    """Run a command with the real subprocess module.

    Args:
        argv: Executable and arguments.
        stdin_bytes: Bytes for standard input, or None.

    Returns:
        The command's exit status and captured streams, decoded as UTF-8 with
        undecodable bytes replaced -- a mangled character in a diagnostic is
        better than losing the diagnostic.
    """
    completed = subprocess.run(
        list(argv),
        check=False,
        input=stdin_bytes,
        capture_output=True,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def _default_now() -> int:
    """Read the real wall clock.

    Returns:
        Whole seconds since the epoch.
    """
    return int(time.time())


def _default_read_text(path: pathlib.Path) -> str:
    """Read a real file as UTF-8.

    Args:
        path: Absolute path to read.

    Returns:
        The file's contents.
    """
    return path.read_text(encoding="utf-8")


def _default_append_text(path: pathlib.Path, line: str) -> None:
    """Append one line to a real file.

    The parent directory is created if absent, because the three records this
    package appends to are created by their first write and a workspace
    pointing at a fresh directory is the ordinary first-run case rather than
    a mistake.

    Args:
        path: Absolute path to append to.
        line: The line, without a trailing newline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _default_read_bytes(path: pathlib.Path) -> bytes:
    """Read a real file without decoding it.

    Args:
        path: Absolute path to read.

    Returns:
        The file's contents.
    """
    return path.read_bytes()


def _default_file_exists(path: pathlib.Path) -> bool:
    """Report whether a real path names an existing file.

    Args:
        path: Absolute path to test.

    Returns:
        True when the path exists and is a regular file.
    """
    return path.is_file()


def _default_write_text(path: pathlib.Path, text: str) -> None:
    """Replace a real file's contents.

    Args:
        path: Absolute path to write.
        text: The complete new contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


run: RunProtocol = _default_run
now: NowProtocol = _default_now
read_text: ReadTextProtocol = _default_read_text
read_bytes: ReadBytesProtocol = _default_read_bytes
file_exists: FileExistsProtocol = _default_file_exists
append_text: AppendTextProtocol = _default_append_text
write_text: WriteTextProtocol = _default_write_text


__all__ = [
    "AppendTextProtocol",
    "CommandResult",
    "FileExistsProtocol",
    "NowProtocol",
    "ReadBytesProtocol",
    "ReadTextProtocol",
    "RunProtocol",
    "WriteTextProtocol",
    "append_text",
    "file_exists",
    "now",
    "read_bytes",
    "read_text",
    "run",
    "write_text",
]
