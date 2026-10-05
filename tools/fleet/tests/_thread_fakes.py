"""Waiting on another thread in a test (MCPs board tasks c1d48330 and 8993c306).

A serving node runner's watch polls and settles on a thread of its own
(:mod:`fleet.cli.node_watch`), so a case that fixes one interleaving, a
handover asked for while a settle is under way, say, waits for an event the
other thread sets, and fails rather than hanging the suite when it never
comes.
"""

from __future__ import annotations

import threading
from typing import Final

#: How long a case waits for an event before it fails, rather than hanging
#: the suite on an interleaving that never came.
WAIT_SECONDS: Final = 30


def await_event(event: threading.Event, *, what: str) -> None:
    """Wait for an event another thread sets.

    Args:
        event: The event.
        what: What it marks, for the failure message.

    Raises:
        AssertionError: When it has not happened within :data:`WAIT_SECONDS`.
    """
    assert event.wait(WAIT_SECONDS), f"waited {WAIT_SECONDS}s for {what}"


__all__ = ["WAIT_SECONDS", "await_event"]
