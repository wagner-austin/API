"""Bounded subprocess input without a caller-thread pipe write."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO


@contextmanager
def process_input(payload: bytes | None) -> Iterator[int | IO[bytes]]:
    """Provide closed stdin or an anonymous file positioned at its first byte.

    A child that never reads stdin must still be governed by subprocess timeout.
    This lifts the HPC transport's file-input boundary for other CLI controllers.

    Args:
        payload: Input bytes; None supplies DEVNULL rather than inherited stdin.

    Yields:
        Input descriptor accepted by subprocess.run; closed on every exit.

    Raises:
        OSError: Temporary input storage cannot be created or written.
    """
    if payload is None:
        yield subprocess.DEVNULL
    else:
        with tempfile.TemporaryFile() as source:
            source.write(payload)
            source.seek(0)
            yield source


__all__ = ["process_input"]
