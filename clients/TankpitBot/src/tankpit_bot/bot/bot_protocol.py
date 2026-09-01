"""The bot surface the executor dispatches against.

Lives here rather than in ``_test_hooks`` because it is a PRODUCTION
interface, not a test seam: ``bot/executor.py`` is its only importer,
and it must name :class:`WorldService`. ``_test_hooks`` sits below
``sniffer`` in the layering, so declaring the world service there
closes an import cycle through ``state`` (measured 2026-08-07)
([[session-state-deglobalisation]]).
"""

from __future__ import annotations

from typing import Protocol

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol, CDPSessionProtocol
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import WorldStateDict


class BotProtocol(BufferedMessageSourceProtocol, Protocol):
    """Interface for bot command dispatch used by executor and world_sync.

    Defines the minimal set of methods these consumers need from the Bot
    class.  tick_loop.py uses Bot directly for AI state access.  Tests
    inject a FakeBot satisfying this protocol instead of mocking.

    The message buffer and its session XOR table are inherited from
    :class:`BufferedMessageSourceProtocol` rather than redeclared.

    Attributes:
        world: The session's world service. Decoded frames land here
            and the executor reads its beliefs before dispatching.
    """

    world: WorldService

    @property
    def _cdp(self) -> CDPSessionProtocol | None:
        """CDP session for browser communication.

        Returns:
            CDP session or None if not connected.
        """
        ...

    def move_to(self, x: int, y: int) -> bool:
        """Send move command.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        ...

    def pickup_fuel_to(self, x: int, y: int) -> bool:
        """Send fuel pickup command.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        ...

    def pickup_equipment_to(self, x: int, y: int) -> bool:
        """Send equipment pickup command.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        ...

    def teleport_to(self, x: int, y: int) -> bool:
        """Send teleport command.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        ...

    def shoot_at(self, x: int, y: int, target_id: int) -> bool:
        """Send shoot command.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            target_id: Target entity ID.

        Returns:
            True if command was sent.
        """
        ...

    def use_radar(self) -> bool:
        """Send radar scan command.

        Returns:
            True if command was sent.
        """
        ...

    def drop_mine(self) -> bool:
        """Send the 3x3 self-centered mine placement command.

        Returns:
            True if command was sent.
        """
        ...

    def send_chat(self, message_id: int, x: int, y: int) -> bool:
        """Send a preset chat message.

        Args:
            message_id: Preset chat message ID (0-64).
            x: Sender's current X tile (0-255).
            y: Sender's current Y tile (0-255).

        Returns:
            True if command was sent.
        """
        ...

    def scope_shift(self, direction: int) -> bool:
        """Send the scope-extend command (shift the stored viewport).

        Args:
            direction: Compass byte, clockwise from north (0=N..7=NW).

        Returns:
            True if command was sent.
        """
        ...

    def open_map(self) -> bool:
        """Send map open command to reveal global enemy positions.

        Returns:
            True if command was sent.
        """
        ...

    def close_map(self) -> bool:
        """Close the map overlay client-side (synthetic keypress, no wire).

        Returns:
            True if the close event was dispatched.
        """
        ...

    def captured_message_count(self) -> int:
        """Return how many WebSocket messages have been captured so far.

        Returns:
            Length of the session's captured-message list.
        """
        ...

    def enable_equipment(self, slot: int) -> bool:
        """Enable equipment slot if not already enabled.

        Args:
            slot: Equipment slot (1-5).

        Returns:
            True if command was sent.
        """
        ...

    def disable_equipment(self, slot: int) -> bool:
        """Disable equipment slot if currently enabled.

        Args:
            slot: Equipment slot (1-5).

        Returns:
            True if command was sent.
        """
        ...

    def _has_equipment_stock(self, slot: int) -> bool:
        """Check if equipment slot has remaining stock.

        Args:
            slot: Equipment slot (1-5).

        Returns:
            True if equipment is available to use.
        """
        ...

    def get_world_state(self) -> WorldStateDict:
        """Return the current tracked world state.

        Returns:
            Current world-state snapshot.
        """
        ...


__all__ = [
    "BotProtocol",
]
