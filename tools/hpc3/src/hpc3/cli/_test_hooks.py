"""Dependency-injection seam for the CLI layer.

The CLI's only impure acts beyond what the core already routes through hooks
are writing its report to stdout and its refusals to stderr. Both go through
hooks so a test asserts what a command reported rather than what pytest
managed to capture, which keeps the assertions about content instead of about
capture behaviour.
"""

from __future__ import annotations

import datetime
import sys
from collections.abc import Callable


def _default_emit(line: str) -> None:
    """Write one report line to stdout.

    Args:
        line: Line to write, without a trailing newline.
    """
    sys.stdout.write(line + "\n")


def _default_emit_error(line: str) -> None:
    """Write one refusal line to stderr.

    Separate stream from :func:`_default_emit` so a command's report can be
    piped or redirected while its refusals still reach the operator, and so
    a refusal never lands in a file something else will parse as output.

    Args:
        line: Line to write, without a trailing newline.
    """
    sys.stderr.write(line + "\n")


def _default_now_iso() -> str:
    """Read the wall clock for the ledger's timestamp.

    Returns:
        The current UTC time in ISO-8601, with an explicit offset. UTC
        because a ledger read on a different machine must not need to know
        where it was written; explicit offset because a bare timestamp gets
        assumed local by whoever reads it next.
    """
    return datetime.datetime.now(tz=datetime.UTC).isoformat()


emit: Callable[[str], None] = _default_emit
emit_error: Callable[[str], None] = _default_emit_error
now_iso: Callable[[], str] = _default_now_iso


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global emit, emit_error, now_iso
    emit = _default_emit
    emit_error = _default_emit_error
    now_iso = _default_now_iso


__all__ = ["emit", "emit_error", "now_iso", "reset_hooks"]
