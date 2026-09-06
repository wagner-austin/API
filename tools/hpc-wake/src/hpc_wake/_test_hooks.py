"""Injection points for what this package does outside the process itself.

Module-level names, rebound by ``tests.conftest`` before each test and
restored after, exactly as in ``board_watch._test_hooks``. Only what THIS
package reaches for is seamed here: the board POST, the report stream, and
the clock. The cluster and the ledger are reached through ``hpc3``'s own
functions, which carry their own seams (``hpc3.core._test_hooks``) -- a
second seam in front of those would be a wrapper whose only content is
another name for the same call.
"""

from __future__ import annotations

import datetime
import sys
from typing import Protocol

from platform_core.mcp_client import McpPostProtocol, urllib_mcp_post


class EmitProtocol(Protocol):
    """Write one line to the cycle's report stream."""

    def __call__(self, line: str) -> None:
        """Write it.

        Args:
            line: The line, without a trailing newline.
        """
        ...


class NowIsoProtocol(Protocol):
    """Read the wall clock for a closure's ``closed_at``."""

    def __call__(self) -> str:
        """Read it.

        Returns:
            The current UTC time in ISO-8601 with an explicit offset.
        """
        ...


def _default_emit(line: str) -> None:
    """Write one line to standard output and flush it.

    The flush is required: a scheduler or Monitor reads this process's
    stdout as a stream, and a buffered line is an event that has not
    happened yet as far as the subscriber is concerned.

    Args:
        line: The line, without a trailing newline.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _default_now_iso() -> str:
    """Read the wall clock for closure timestamps.

    Returns:
        The current UTC time in ISO-8601 with an explicit offset, matching
        what the hpc3 ledger records for ``submitted_at``.
    """
    return datetime.datetime.now(tz=datetime.UTC).isoformat()


http_post: McpPostProtocol = urllib_mcp_post
emit: EmitProtocol = _default_emit
now_iso: NowIsoProtocol = _default_now_iso


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global http_post, emit, now_iso
    http_post = urllib_mcp_post
    emit = _default_emit
    now_iso = _default_now_iso


__all__ = [
    "EmitProtocol",
    "NowIsoProtocol",
    "emit",
    "http_post",
    "now_iso",
    "reset_hooks",
]
