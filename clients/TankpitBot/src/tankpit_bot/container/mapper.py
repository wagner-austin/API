"""Player ID to Tank ID mapping.

This module provides the PlayerIdMapper class for correlating
session-specific player_id values with tank_id values.
"""

from __future__ import annotations

from tankpit_bot.container.types import MovementDict


class PlayerIdMapper:
    """Maps session-specific player_id to tank_id.

    Movement messages contain a player_id (LE uint32 at bytes 8-11) that is
    session-specific and different from tank_id. This class builds a mapping
    by correlating Movement start positions with MovementResponse positions.

    Usage:
        mapper = PlayerIdMapper()

        # Feed MovementResponse to learn tank positions
        mapper.record_movement_response(tank_id=638, x=36, y=122)

        # When Movement arrives, resolve tank_id
        movement_msg = decode_movement(data)
        mapper.resolve_movement(movement_msg)
        # movement_msg["tank_id"] is now set if position matched

        # Or query directly
        tank_id = mapper.get_tank_id(player_id=231214)
    """

    def __init__(self) -> None:
        """Initialize mapper."""
        # player_id -> tank_id mapping (persistent for session)
        self._player_to_tank: dict[int, int] = {}
        # (x, y) -> tank_id from recent MovementResponse (for correlation)
        self._position_to_tank: dict[tuple[int, int], int] = {}

    def record_movement_response(self, tank_id: int, x: int, y: int) -> None:
        """Record a MovementResponse for position correlation.

        Args:
            tank_id: Tank ID from MovementResponse.
            x: X coordinate from MovementResponse.
            y: Y coordinate from MovementResponse.
        """
        self._position_to_tank[(x, y)] = tank_id

    def resolve_movement(self, movement: MovementDict) -> None:
        """Resolve tank_id for a Movement message.

        Looks up tank_id by:
        1. Direct player_id lookup if already known
        2. Position correlation with recent MovementResponse

        Modifies movement dict in-place to set tank_id.

        Args:
            movement: Movement message to resolve (modified in-place).
        """
        player_id = movement["player_id"]

        # Check cached mapping first
        if player_id in self._player_to_tank:
            movement["tank_id"] = self._player_to_tank[player_id]
            return

        # Try position correlation
        pos = (movement["start_x"], movement["start_y"])
        tank_id = self._position_to_tank.get(pos)
        if tank_id is not None:
            # Learn the mapping
            self._player_to_tank[player_id] = tank_id
            movement["tank_id"] = tank_id

    def get_tank_id(self, player_id: int) -> int | None:
        """Get tank_id for a player_id.

        Args:
            player_id: Session-specific player ID from Movement message.

        Returns:
            Tank ID if known, None otherwise.
        """
        return self._player_to_tank.get(player_id)

    def get_player_id(self, tank_id: int) -> int | None:
        """Get player_id for a tank_id (reverse lookup).

        Args:
            tank_id: Tank ID from TankRegistry/MovementResponse.

        Returns:
            Player ID if known, None otherwise.
        """
        for pid, tid in self._player_to_tank.items():
            if tid == tank_id:
                return pid
        return None

    def clear(self) -> None:
        """Clear all mappings (call on new session)."""
        self._player_to_tank.clear()
        self._position_to_tank.clear()


__all__ = [
    "PlayerIdMapper",
]
