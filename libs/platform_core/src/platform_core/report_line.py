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

    def __call__(self, line: str) -> None:
        """Write it.

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


__all__ = ["EmitProtocol", "emit_line"]
