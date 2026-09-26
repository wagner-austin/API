"""Injection points for what this package does outside the process itself.

Module-level names, rebound by ``tests.conftest`` before each test and
restored after, exactly as in ``lock_wake._test_hooks`` and its siblings.
Production binds the real implementations at import; a test binds fakes.
There is no conditional anywhere -- the call site calls the hook.

The file seams are :mod:`platform_core.journal_cursor`'s and the report
stream is :mod:`platform_core.report_line`'s; that module's header says why
the journal is read as bytes and the position rewritten whole.
"""

from __future__ import annotations

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

http_post: McpPostProtocol = urllib_mcp_post
read_bytes: ReadBytesProtocol = read_file_bytes
write_text: WriteTextProtocol = write_file_text
file_exists: FileExistsProtocol = file_is_present
emit: EmitProtocol = emit_line


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global http_post, read_bytes, write_text, file_exists, emit
    http_post = urllib_mcp_post
    read_bytes = read_file_bytes
    write_text = write_file_text
    file_exists = file_is_present
    emit = emit_line


__all__ = [
    "emit",
    "file_exists",
    "http_post",
    "read_bytes",
    "reset_hooks",
    "write_text",
]
