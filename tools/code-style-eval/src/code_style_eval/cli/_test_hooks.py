"""Dependency-injection seam for the CLI layer.

The CLI's only impure act beyond what the core already routes through hooks
is writing its summary to stdout. It goes through a hook so a test can assert
what a run reported without capturing process output, which keeps the
assertions about the summary's content rather than about pytest's capture
behaviour.
"""

from __future__ import annotations

import sys
from collections.abc import Callable


def _default_emit(line: str) -> None:
    """Write one summary line to stdout.

    Args:
        line: Line to write, without a trailing newline.
    """
    sys.stdout.write(line + "\n")


emit: Callable[[str], None] = _default_emit


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global emit
    emit = _default_emit


__all__ = ["emit", "reset_hooks"]
