"""Shared session-synchronization helpers for live action probes."""

from __future__ import annotations

from typing import Protocol

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.state import SelfStateDict, WorldStateDict

_POLL_INTERVAL_MS = 100.0


class ActionLabSessionError(Exception):
    """Raised when a live action probe session is not yet ready."""


class WaitPageProtocol(Protocol):
    """Minimal page protocol for action-lab wait loops."""

    def wait_for_timeout(self, timeout: float) -> None:
        """Block for the requested number of milliseconds.

        Args:
            timeout: Milliseconds to wait.
        """
        ...


class WorldStateProviderProtocol(Protocol):
    """Minimal world-state provider for action-lab wait loops."""

    def get_world_state(self) -> WorldStateDict:
        """Return the latest world-state snapshot.

        Returns:
            Current world-state snapshot.
        """
        ...


class BufferedWorldStateProviderProtocol(WorldStateProviderProtocol, Protocol):
    """Minimal world-state provider that also buffers raw protocol messages."""

    _cdp_message_buffer: list[str]


class StartupStateDriverProtocol(Protocol):
    """Minimal startup-state driver for a live action probe bot."""

    def get_state(self) -> str:
        """Return the current state-machine state name.

        Returns:
            Current state name.
        """
        ...

    def _update_state_from_world(self) -> None:
        """Advance the state machine from the current world snapshot."""
        ...


def wait_for_world_sync(
    page: WaitPageProtocol,
    provider: BufferedWorldStateProviderProtocol,
    started_ms: int,
    timeout_ms: int,
) -> int | None:
    """Wait for the first world-state sync newer than ``started_ms``.

    Args:
        page: Page-like object used for short waits.
        provider: World-state provider to poll.
        started_ms: Lower bound timestamp for a fresh sync.
        timeout_ms: Maximum time to wait.

    Returns:
        The fresh world-state timestamp, or None on timeout.
    """
    while action_hooks.get_current_time_ms() - started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(provider)
        world = provider.get_world_state()
        if world["timestamp_ms"] > started_ms:
            return world["timestamp_ms"]
        page.wait_for_timeout(_POLL_INTERVAL_MS)
    return None


def wait_for_initial_self_state(
    page: WaitPageProtocol,
    provider: BufferedWorldStateProviderProtocol,
    started_ms: int,
    timeout_ms: int,
) -> tuple[int, SelfStateDict]:
    """Wait for an initial self state from a fresh world sync.

    Args:
        page: Page-like object used for short waits.
        provider: World-state provider to poll.
        started_ms: Timestamp immediately before readiness polling begins.
        timeout_ms: Maximum time to wait for an initial self state.

    Returns:
        A tuple of the fresh world timestamp and the current self state.

    Raises:
        ActionLabSessionError: If a fresh self state does not arrive in time.
    """
    while action_hooks.get_current_time_ms() - started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(provider)
        world = provider.get_world_state()
        self_state = world["self_state"]
        if world["timestamp_ms"] > started_ms and self_state is not None:
            return world["timestamp_ms"], self_state
        page.wait_for_timeout(_POLL_INTERVAL_MS)
    raise ActionLabSessionError("initial self state is unavailable after initial sync wait")


def advance_startup_state(bot: StartupStateDriverProtocol) -> None:
    """Advance a bot from startup bootstrap states to an executable state.

    The real bot reaches ``IDLE`` over successive tick-loop sync passes:
    ``INITIALIZING -> WAITING_FOR_POSITION -> IDLE``. Action-lab probes do
    not run that tick loop, so they must explicitly drive those startup
    transitions after initial world/self-state sync has completed.

    Args:
        bot: Bot-like object with startup state transitions.

    Raises:
        ActionLabSessionError: If the bot remains in a bootstrap state after
            the expected startup transitions.
    """
    while bot.get_state() in ("INITIALIZING", "WAITING_FOR_POSITION"):
        previous_state = bot.get_state()
        bot._update_state_from_world()
        if bot.get_state() == previous_state:
            raise ActionLabSessionError(f"startup state did not advance from {previous_state}")


__all__ = [
    "ActionLabSessionError",
    "BufferedWorldStateProviderProtocol",
    "StartupStateDriverProtocol",
    "WaitPageProtocol",
    "WorldStateProviderProtocol",
    "advance_startup_state",
    "wait_for_initial_self_state",
    "wait_for_world_sync",
]
