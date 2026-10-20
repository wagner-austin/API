"""One line to a scheduled tool's report stream, flushed as it is written.

Lifted when ``tools/fleet-health-wake`` would have become the eighth
package carrying the same two-line ``_default_emit`` in its
``_test_hooks`` (board-watch, ci-wake, commit-scope, fleet-wake, hpc-wake,
lock-wake and maketools each hold one; MCPs board task ebc80a03). A
package binds :func:`emit_line` as its ``emit`` hook, and a test rebinds
that hook to capture the lines.
"""

from __future__ import annotations

import sys
from typing import Protocol


class EmitProtocol(Protocol):
    """Write one line to a report stream."""

    def __call__(self, line: str, /) -> None:
        """Write it.

        Positional-only, so a test's ``list.append`` is a writer too.

        Args:
            line: The line, without a trailing newline.
        """
        ...


def emit_line(line: str) -> None:
    """Write one line to standard output and flush it.

    The flush is required: the pump reads a publisher's stdout as a stream,
    and a buffered line is an event that has not happened yet as far as the
    reader is concerned.

    Args:
        line: The line, without a trailing newline.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def emit_error_line(line: str) -> None:
    """Write one line to standard error and flush it.

    The refusal stream :func:`platform_core.cli_boundary.run_cli` writes to,
    apart from :func:`emit_line`'s so a command's report can be piped or
    redirected while its refusals still reach the operator.

    Args:
        line: The line, without a trailing newline.
    """
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


__all__ = ["EmitProtocol", "emit_error_line", "emit_line"]
