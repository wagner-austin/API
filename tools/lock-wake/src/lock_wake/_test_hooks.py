"""Injection points for what this package does outside the process itself.

Module-level names, rebound by ``tests.conftest`` before each test and
restored after, exactly as in ``ci_wake._test_hooks`` and its siblings.
Production binds the real implementations at import; a test binds fakes.
There is no conditional anywhere -- the call site calls the hook.

The file seams' Protocols and production implementations are
:mod:`platform_core.journal_cursor`'s, and the report stream's are
:mod:`platform_core.report_line`'s, lifted out of this module when a second
journal bridge needed them (MCPs board task ebc80a03); that module's header
says why the journal is read as bytes and the position rewritten whole.

The ssh seam is this package's own: only :mod:`lock_wake.remote` runs a
process, to read the unread window of a journal on another host (MCPs board
task 03590bf9), and its output is BYTES for the reason the local read is.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Protocol

from platform_core.journal_cursor import (
    FileExistsProtocol,
    ReadBytesProtocol,
    WriteTextProtocol,
    file_is_present,
    read_file_bytes,
    write_file_text,
)
from platform_core.mcp_client import McpPostProtocol, urllib_mcp_post
from platform_core.report_line import EmitProtocol, emit_line


class CompletedSshProtocol(Protocol):
    """The three fields the remote read takes off a finished ssh."""

    @property
    def returncode(self) -> int:
        """The process's exit status; 255 is ssh's own failure."""
        ...

    @property
    def stdout(self) -> bytes:
        """Captured standard output, undecoded."""
        ...

    @property
    def stderr(self) -> bytes:
        """Captured standard error, undecoded."""
        ...


class RunSshProtocol(Protocol):
    """Run one ssh command line to completion with its output captured."""

    def __call__(self, args: Sequence[str], timeout_seconds: int) -> CompletedSshProtocol:
        """Run it.

        Args:
            args: The full argv, ``ssh`` first.
            timeout_seconds: The deadline; past it the process is killed.

        Returns:
            The finished process.
        """
        ...


def run_ssh_command(args: Sequence[str], timeout_seconds: int) -> CompletedSshProtocol:
    """Run a real process: the production :class:`RunSshProtocol`.

    Args:
        args: The full argv, ``ssh`` first.
        timeout_seconds: The deadline; past it the process is killed.

    Returns:
        The finished process, its exit status inspected by the caller
        rather than raised here, so a refusal names the host it came from.

    Raises:
        subprocess.TimeoutExpired: The process outlived the deadline.
        OSError: The executable cannot be spawned.
    """
    return subprocess.run(list(args), capture_output=True, check=False, timeout=timeout_seconds)


http_post: McpPostProtocol = urllib_mcp_post
read_bytes: ReadBytesProtocol = read_file_bytes
write_text: WriteTextProtocol = write_file_text
file_exists: FileExistsProtocol = file_is_present
emit: EmitProtocol = emit_line
run_ssh: RunSshProtocol = run_ssh_command


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global http_post, read_bytes, write_text, file_exists, emit, run_ssh
    http_post = urllib_mcp_post
    read_bytes = read_file_bytes
    write_text = write_file_text
    file_exists = file_is_present
    emit = emit_line
    run_ssh = run_ssh_command


__all__ = [
    "CompletedSshProtocol",
    "RunSshProtocol",
    "emit",
    "file_exists",
    "http_post",
    "read_bytes",
    "reset_hooks",
    "run_ssh",
    "run_ssh_command",
    "write_text",
]
