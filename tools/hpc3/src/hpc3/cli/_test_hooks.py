"""Dependency-injection seam for the CLI layer.

The CLI's only impure act beyond what the core already routes through hooks
is writing its report to stdout. It goes through a hook so a test asserts what
a command reported rather than what pytest managed to capture, which keeps the
assertions about content instead of about capture behaviour.
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
now_iso: Callable[[], str] = _default_now_iso


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global emit, now_iso
    emit = _default_emit
    now_iso = _default_now_iso


__all__ = ["emit", "now_iso", "reset_hooks"]
