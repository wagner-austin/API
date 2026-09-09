"""Injection points for what this package does outside the process itself.

Module-level names, rebound by ``tests.conftest`` before each test and
restored after, exactly as in ``ci_wake._test_hooks`` and its siblings.
Production binds the real implementations at import; a test binds fakes.
There is no conditional anywhere -- the call site calls the hook.

FOUR SEAMS, AND THE BYTE READER IS THE ONE WORTH EXPLAINING. The journal's
contract is a BYTE offset ("a subscriber keeps a byte offset, reads from
it, and cannot miss a transition"), and the file is written by another
process that may be mid-append when this one reads. So the reader takes
bytes, not text: a text read would decode a torn multi-byte character at
the tail into a replacement char and corrupt the offset arithmetic, where
a byte slice lets :mod:`lock_wake.journal` consume exactly through the
last complete line and leave the torn tail for the next cycle.

THE POSITION IS A WHOLE-FILE WRITE, NOT AN APPEND. One integer is the
entire memory; rewriting it is atomic enough for a single-writer file and
keeps the record self-pruning, where an append-only offset log would grow
forever saying nothing the last line does not.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Protocol

from platform_core.mcp_client import McpPostProtocol, urllib_mcp_post


class ReadBytesProtocol(Protocol):
    """Read a whole file as raw bytes."""

    def __call__(self, path: pathlib.Path) -> bytes:
        """Read it.

        Args:
            path: Absolute path to read.

        Returns:
            The file's contents, undecoded.
        """
        ...


class WriteTextProtocol(Protocol):
    """Replace a file's contents with one string."""

    def __call__(self, path: pathlib.Path, content: str) -> None:
        """Write it.

        Args:
            path: Absolute path to write.
            content: The full new contents.
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


def _default_read_bytes(path: pathlib.Path) -> bytes:
    """Read a real file's bytes.

    Args:
        path: Absolute path to read.

    Returns:
        The file's contents.
    """
    return path.read_bytes()


def _default_write_text(path: pathlib.Path, content: str) -> None:
    """Replace a real file's contents.

    The parent directory is created if absent: the position file is created
    by its first write, and a machine whose bridge has never run is the
    ordinary first-run case rather than a mistake.

    Args:
        path: Absolute path to write.
        content: The full new contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


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

    The flush is required: the pump reads this process's stdout as a
    stream, and a buffered line is an event that has not happened yet as
    far as the subscriber is concerned.

    Args:
        line: The line, without a trailing newline.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


http_post: McpPostProtocol = urllib_mcp_post
read_bytes: ReadBytesProtocol = _default_read_bytes
write_text: WriteTextProtocol = _default_write_text
file_exists: FileExistsProtocol = _default_file_exists
emit: EmitProtocol = _default_emit


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global http_post, read_bytes, write_text, file_exists, emit
    http_post = urllib_mcp_post
    read_bytes = _default_read_bytes
    write_text = _default_write_text
    file_exists = _default_file_exists
    emit = _default_emit


__all__ = [
    "EmitProtocol",
    "FileExistsProtocol",
    "ReadBytesProtocol",
    "WriteTextProtocol",
    "emit",
    "file_exists",
    "http_post",
    "read_bytes",
    "reset_hooks",
    "write_text",
]
