"""Threadsafe latest-wins channel for SPA mode overrides.

The aiohttp thread that services ``POST /api/tankbot/mode`` writes into
this channel; the sync tick loop reads it at the top of every tick.
Semantics are deliberately latest-wins — if the SPA fires several mode
changes between two consecutive ticks, only the most recent one bites.
Older overrides are discarded silently because the wire outcome the
user cares about is the final button they pressed, not every one they
scrubbed past.
"""

from __future__ import annotations

import threading
from typing import Protocol

from tankpit_bot.bus.session_status import WireMode


class ModeBridgeProtocol(Protocol):
    """Surface the tick loop uses to read SPA mode overrides.

    Kept alongside the concrete :class:`ModeBridge` so consumers that
    want DI (service main, tests) can accept a fake without inheriting
    from the real implementation. Structural typing — a fake with the
    same three method signatures satisfies the Protocol without
    importing this module.
    """

    def submit(self, mode: WireMode) -> None:
        """Replace the pending override.

        Args:
            mode: Wire-level mode literal the SPA selected.
        """
        ...

    def drain(self) -> WireMode | None:
        """Consume and return the pending override.

        Returns:
            The wire mode written since the last drain, or ``None``
            when nothing is queued.
        """
        ...

    def peek(self) -> WireMode | None:
        """Return the pending override without consuming it.

        Returns:
            The pending override, or ``None`` when nothing is queued.
        """
        ...


class ModeBridge:
    """Latest-wins :class:`WireMode` channel between HTTP and tick loop.

    ``submit`` is called from the aiohttp handler thread; ``drain`` is
    called from the sync tick-loop thread. Both are protected by a
    single :class:`threading.Lock` — there is no reader/writer
    contention worth optimising here, and the critical section is a
    single attribute write.
    """

    def __init__(self) -> None:
        """Create an empty bridge with no pending override."""
        self._lock = threading.Lock()
        self._pending: WireMode | None = None

    def submit(self, mode: WireMode) -> None:
        """Replace the pending override with ``mode``.

        Called from the HTTP handler thread. Any older override still
        queued from a previous ``POST /mode`` is overwritten — the
        latest button press wins.

        Args:
            mode: Wire-level mode literal the SPA selected.
        """
        with self._lock:
            self._pending = mode

    def drain(self) -> WireMode | None:
        """Consume the pending override.

        Called from the tick loop at the top of each tick. Returns the
        stored value and atomically clears the slot so the next tick
        does not see a stale override.

        Returns:
            The wire mode written by the SPA since the last ``drain``
            call, or ``None`` when no override is pending.
        """
        with self._lock:
            value = self._pending
            self._pending = None
        return value

    def peek(self) -> WireMode | None:
        """Return the pending override without consuming it.

        Provided for status introspection (e.g. tests + service
        readiness endpoints). The tick loop must use :meth:`drain`.

        Returns:
            The pending override, or ``None`` when nothing is queued.
        """
        with self._lock:
            return self._pending


__all__ = ["ModeBridge", "ModeBridgeProtocol"]
