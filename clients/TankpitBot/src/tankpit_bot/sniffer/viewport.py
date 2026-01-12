"""Viewport position tracking for container coordinate calculation.

This module tracks the viewport position to convert container viewport-relative
coordinates to absolute world coordinates.
"""

from __future__ import annotations

# Viewport tracking for absolute container position calculation
# viewport_left is derived from PositionUpdate: player_absolute_x - player_viewport_x
_viewport_left: int | None = None
_self_tank_id: int | None = None


def update_viewport_from_position_update(tank_id: int, x: int, y: int, extra_data: bytes) -> None:
    """Update viewport_left from PositionUpdate message with absolute coords.

    The first PositionUpdate after join/teleport has absolute coordinates.
    extra_data[0] contains the player's viewport-relative x position.
    Formula: viewport_left = absolute_x - viewport_x

    Args:
        tank_id: Tank ID from the message.
        x: Absolute x coordinate from position_update.
        y: Absolute y coordinate from position_update.
        extra_data: Extra data bytes (extra_data[0] = player_viewport_x).
    """
    global _viewport_left, _self_tank_id

    # Only process if we have extra_data and position looks absolute (not 3,3)
    if len(extra_data) < 1:
        return

    # Skip viewport-relative messages (always show pos=(3,3))
    if x == 3 and y == 3:
        return

    # Track self tank ID from first absolute position message
    if _self_tank_id is None:
        _self_tank_id = tank_id

    # Only update viewport for self tank
    if tank_id != _self_tank_id:
        return

    player_viewport_x = extra_data[0]
    _viewport_left = x - player_viewport_x


def get_viewport_left() -> int | None:
    """Get current viewport left edge x coordinate.

    Returns:
        Viewport left edge x coordinate, or None if not yet determined.
    """
    return _viewport_left


def reset_viewport_tracking() -> None:
    """Reset viewport tracking state.

    Called when starting a new session or leaving the game.
    """
    global _viewport_left, _self_tank_id
    _viewport_left = None
    _self_tank_id = None


__all__ = [
    "get_viewport_left",
    "reset_viewport_tracking",
    "update_viewport_from_position_update",
]
