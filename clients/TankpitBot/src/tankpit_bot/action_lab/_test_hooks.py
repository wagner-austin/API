"""Internal hooks for action-lab dependency injection.

Production code binds these hooks to real implementations. Tests override them
to exercise live-probe control flow deterministically.
"""

from __future__ import annotations

from typing import Protocol

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot.action_lab.session import (
    BufferedWorldStateProviderProtocol,
    StartupStateDriverProtocol,
    WaitPageProtocol,
)
from tankpit_bot.action_lab.session import (
    advance_startup_state as _real_advance_startup_state,
)
from tankpit_bot.action_lab.session import (
    wait_for_initial_self_state as _real_wait_for_initial_self_state,
)
from tankpit_bot.action_lab.session import (
    wait_for_radar_sync as _real_wait_for_radar_sync,
)
from tankpit_bot.action_lab.session import (
    wait_for_world_sync as _real_wait_for_world_sync,
)
from tankpit_bot.bot.world_sync import drain_messages as _real_drain_buffered_messages
from tankpit_bot.browser import get_current_time_ms as _real_get_current_time_ms
from tankpit_bot.browser.lifecycle import (
    gather_intel as _real_gather_intel,
)
from tankpit_bot.browser.lifecycle import (
    navigate_and_login as _real_navigate_and_login,
)
from tankpit_bot.browser.lifecycle import (
    wait_for_game_ready as _real_wait_for_game_ready,
)
from tankpit_bot.sniffer.world_state import (
    check_and_clear_radar_scan_complete as _real_check_and_clear_radar_scan_complete,
)
from tankpit_bot.sniffer.world_state import (
    get_world_service,
)
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_teleport_landed as _real_check_and_clear_teleport_landed,
)
from tankpit_bot.state import SelfStateDict


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


def _default_check_and_clear_teleport_landed() -> bool:
    return _real_check_and_clear_teleport_landed(get_world_service())


check_and_clear_teleport_landed: CheckAndClearTeleportLandedProtocol = (
    _default_check_and_clear_teleport_landed
)


class CheckAndClearRadarScanCompleteProtocol(Protocol):
    """Protocol for draining the radar-scan-complete confirmation flag."""

    def __call__(self) -> bool:
        """Return True when a radar completion confirmation is pending."""
        ...


check_and_clear_radar_scan_complete: CheckAndClearRadarScanCompleteProtocol = (
    _real_check_and_clear_radar_scan_complete
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


# ---------------------------------------------------------------------------
# Session wait hooks
# ---------------------------------------------------------------------------


class WaitForWorldSyncProtocol(Protocol):
    """Protocol for waiting on a world state sync."""

    def __call__(
        self,
        page: WaitPageProtocol,
        provider: BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        """Wait for a world-state sync newer than started_ms.

        Args:
            page: Page-like object for short waits.
            provider: World-state provider to poll.
            started_ms: Lower bound timestamp.
            timeout_ms: Maximum wait time.

        Returns:
            Sync timestamp, or None on timeout.
        """
        ...


class WaitForRadarSyncProtocol(Protocol):
    """Protocol for waiting on radar completion."""

    def __call__(
        self,
        page: WaitPageProtocol,
        provider: BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        """Wait for radar completion.

        Args:
            page: Page-like object for short waits.
            provider: World-state provider to poll.
            started_ms: Lower bound timestamp.
            timeout_ms: Maximum wait time.

        Returns:
            Sync timestamp, or None on timeout.
        """
        ...


class WaitForInitialSelfStateProtocol(Protocol):
    """Protocol for waiting on initial self state."""

    def __call__(
        self,
        page: WaitPageProtocol,
        provider: BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        """Wait for an initial self state from a fresh world sync.

        Args:
            page: Page-like object for short waits.
            provider: World-state provider to poll.
            started_ms: Timestamp before polling begins.
            timeout_ms: Maximum wait time.

        Returns:
            Tuple of (world timestamp, self state).
        """
        ...


class AdvanceStartupStateProtocol(Protocol):
    """Protocol for advancing startup state to executable state."""

    def __call__(self, bot: StartupStateDriverProtocol) -> None:
        """Advance startup state transitions.

        Args:
            bot: Bot-like object with startup transitions.
        """
        ...


wait_for_world_sync: WaitForWorldSyncProtocol = _real_wait_for_world_sync
wait_for_radar_sync: WaitForRadarSyncProtocol = _real_wait_for_radar_sync
wait_for_initial_self_state: WaitForInitialSelfStateProtocol = _real_wait_for_initial_self_state
advance_startup_state: AdvanceStartupStateProtocol = _real_advance_startup_state


# ---------------------------------------------------------------------------
# Browser lifecycle hooks
# ---------------------------------------------------------------------------


gather_intel = _real_gather_intel
navigate_and_login = _real_navigate_and_login
wait_for_game_ready = _real_wait_for_game_ready


__all__ = [
    "AdvanceStartupStateProtocol",
    "CheckAndClearRadarScanCompleteProtocol",
    "CheckAndClearTeleportLandedProtocol",
    "DrainBufferedMessagesProtocol",
    "GetCurrentTimeMsProtocol",
    "WaitForInitialSelfStateProtocol",
    "WaitForRadarSyncProtocol",
    "WaitForWorldSyncProtocol",
    "advance_startup_state",
    "check_and_clear_radar_scan_complete",
    "check_and_clear_teleport_landed",
    "drain_buffered_messages",
    "gather_intel",
    "get_current_time_ms",
    "navigate_and_login",
    "wait_for_game_ready",
    "wait_for_initial_self_state",
    "wait_for_radar_sync",
    "wait_for_world_sync",
]
