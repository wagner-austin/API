"""Injection points for what this package does outside the process itself.

Module-level names, rebound by ``tests.conftest`` before each test and restored
after, exactly as in ``hpc_wake._test_hooks`` and ``fleet.core._test_hooks``.
Production binds the real implementations at import; a test binds fakes. There
is no conditional anywhere -- the call site calls the hook.

ONLY WHAT THIS PACKAGE REACHES FOR IS SEAMED HERE: the board POST, the position
file's reads and writes, the report stream, and the clock. The fleet LEDGER is
read through ``fleet.core.records``, which carries its own seam
(``fleet.core._test_hooks``); a second seam in front of that would be a wrapper
whose only content is another name for the same call, and two seams over one
file is how a test ends up pinning one and exercising the other.

THE POSITION FILE IS SEAMED SEPARATELY FROM THE LEDGER ON PURPOSE. A cycle
reads the ledger through fleet's hooks and writes its own record through these,
so a test can give the two different behaviour -- a readable ledger beside a
position file that fails to write, which is the case that decides whether an
announcement is repeated or lost.
"""

from __future__ import annotations

import datetime
import pathlib
import sys
from typing import Protocol

from platform_core.mcp_client import McpPostProtocol, urllib_mcp_post


class ReadTextProtocol(Protocol):
    """Read a whole file as UTF-8 text."""

    def __call__(self, path: pathlib.Path) -> str:
        """Read it.

        Args:
            path: Absolute path to read.

        Returns:
            The file's contents.
        """
        ...


class AppendTextProtocol(Protocol):
    """Append one line to a file."""

    def __call__(self, path: pathlib.Path, line: str) -> None:
        """Append it.

        Args:
            path: Absolute path to append to.
            line: The line, without a trailing newline.
        """
        ...


class FileExistsProtocol(Protocol):
    """Report whether a path is an existing file."""

    def __call__(self, path: pathlib.Path) -> bool:
        """Check it.

        Args:
            path: Absolute path to test.

        Returns:
            True when the file exists.
        """
        ...


class EmitProtocol(Protocol):
    """Write one line to the cycle's report stream."""

    def __call__(self, line: str) -> None:
        """Write it.

        Args:
            line: The line, without a trailing newline.
        """
        ...


class NowProtocol(Protocol):
    """Read the wall clock for a position record's timestamp."""

    def __call__(self) -> int:
        """Read it.

        Returns:
            Whole seconds since the epoch, matching what the fleet ledger
            records for ``started_unix`` and ``ended_unix``.
        """
        ...


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

    The parent directory is created if absent, because the position file is
    created by its first write and a workspace whose bridge has never run is
    the ordinary first-run case rather than a mistake.

    Args:
        path: Absolute path to append to.
        line: The line, without a trailing newline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _default_file_exists(path: pathlib.Path) -> bool:
    """Test a real path.

    Args:
        path: Absolute path to test.

    Returns:
        True when it is an existing file.
    """
    return path.is_file()


def _default_emit(line: str) -> None:
    """Write one line to standard output and flush it.

    The flush is required: a scheduler or Monitor reads this process's stdout
    as a stream, and a buffered line is an event that has not happened yet as
    far as the subscriber is concerned.

    Args:
        line: The line, without a trailing newline.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _default_now() -> int:
    """Read the wall clock for position timestamps.

    Returns:
        Whole seconds since the epoch, UTC.
    """
    return int(datetime.datetime.now(tz=datetime.UTC).timestamp())


http_post: McpPostProtocol = urllib_mcp_post
read_text: ReadTextProtocol = _default_read_text
append_text: AppendTextProtocol = _default_append_text
file_exists: FileExistsProtocol = _default_file_exists
emit: EmitProtocol = _default_emit
now: NowProtocol = _default_now


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global http_post, read_text, append_text, file_exists, emit, now
    http_post = urllib_mcp_post
    read_text = _default_read_text
    append_text = _default_append_text
    file_exists = _default_file_exists
    emit = _default_emit
    now = _default_now


__all__ = [
    "AppendTextProtocol",
    "EmitProtocol",
    "FileExistsProtocol",
    "NowProtocol",
    "ReadTextProtocol",
    "append_text",
    "emit",
    "file_exists",
    "http_post",
    "now",
    "read_text",
    "reset_hooks",
]
