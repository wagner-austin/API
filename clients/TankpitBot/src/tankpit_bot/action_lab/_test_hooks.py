"""Internal hooks for action-lab dependency injection.

Production code binds these hooks to real implementations. Tests override them
to exercise live-probe control flow deterministically.
"""

from __future__ import annotations

from typing import Protocol

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot.bot.world_sync import drain_messages as _real_drain_buffered_messages
from tankpit_bot.browser import get_current_time_ms as _real_get_current_time_ms
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_teleport_landed as _real_check_and_clear_teleport_landed,
)


class GetCurrentTimeMsProtocol(Protocol):
    """Protocol for retrieving the current wall-clock time."""

    def __call__(self) -> int:
        """Return the current Unix timestamp in milliseconds."""
        ...


get_current_time_ms: GetCurrentTimeMsProtocol = _real_get_current_time_ms


class CheckAndClearTeleportLandedProtocol(Protocol):
    """Protocol for draining the teleport-landed confirmation flag."""

    def __call__(self) -> bool:
        """Return True when a teleport landed confirmation is pending."""
        ...


check_and_clear_teleport_landed: CheckAndClearTeleportLandedProtocol = (
    _real_check_and_clear_teleport_landed
)


class DrainBufferedMessagesProtocol(Protocol):
    """Protocol for draining buffered protocol payloads into world state."""

    def __call__(self, source: BufferedMessageSourceProtocol, /) -> int:
        """Drain buffered protocol messages.

        Args:
            source: Source holding buffered protocol payloads.

        Returns:
            Number of drained payloads.
        """
        ...


drain_buffered_messages: DrainBufferedMessagesProtocol = _real_drain_buffered_messages


__all__ = [
    "CheckAndClearTeleportLandedProtocol",
    "DrainBufferedMessagesProtocol",
    "GetCurrentTimeMsProtocol",
    "check_and_clear_teleport_landed",
    "drain_buffered_messages",
    "get_current_time_ms",
]
