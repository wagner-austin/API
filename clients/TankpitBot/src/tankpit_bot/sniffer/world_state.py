"""World state tracking from radar and movement messages.

This module maintains the current world state (containers, mines, player position)
and renders ASCII visualizations of the game world.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks, protocol
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.container import RadarContainerDict, RadarMineDict
from tankpit_bot.state import (
    WorldStateDict,
    add_mine_from_radar,
    make_empty_world_state,
    render_world_ascii,
    update_container_from_radar,
    update_self_position_and_viewport,
)

log = get_logger(__name__)

# Module-level world state - updated as messages are processed
_world_state: WorldStateDict = make_empty_world_state()

# Module-level terrain map - loaded on first radar response
_terrain_map: _test_hooks.TerrainMapProtocol | None = None


def reset_world_state() -> None:
    """Reset world state for new session (used by tests)."""
    global _world_state, _terrain_map
    _world_state = make_empty_world_state()
    _terrain_map = None


def _load_terrain_map_if_needed() -> _test_hooks.TerrainMapProtocol | None:
    """Load terrain map from default GIF if not already loaded.

    Returns:
        TerrainMap instance, or None if file not found.
    """
    global _terrain_map
    if _terrain_map is not None:
        return _terrain_map

    # Try to load from known GIF paths
    gif_paths = [
        Path("field42-r.gif"),
        Path("field01_r.gif"),
    ]
    for gif_path in gif_paths:
        if _test_hooks.path_exists(gif_path):
            _terrain_map = _test_hooks.load_terrain_map(gif_path)
            log.info(f"Loaded terrain map from {gif_path}")
            return _terrain_map

    return None


def update_world_state_from_position(x: int, y: int) -> None:
    """Update world state with new self position.

    Args:
        x: Self X coordinate.
        y: Self Y coordinate.
    """
    global _world_state
    _world_state = update_self_position_and_viewport(_world_state, x, y, get_current_time_ms())


def update_world_state_from_radar(
    containers: list[RadarContainerDict],
    mines: list[RadarMineDict],
) -> None:
    """Update world state with radar scan results.

    Args:
        containers: List of containers from radar.
        mines: List of mines from radar.
    """
    global _world_state
    ts = get_current_time_ms()

    # Add containers
    for c in containers:
        _world_state = update_container_from_radar(_world_state, c["x"], c["y"], c["volume"], ts)

    # Add mines
    for m in mines:
        _world_state = add_mine_from_radar(_world_state, m["x"], m["y"], m["team"], ts)


def render_world_state_ascii() -> str | None:
    """Render current world state as ASCII.

    Returns:
        ASCII representation, or None if terrain map not loaded.
    """
    terrain = _load_terrain_map_if_needed()
    if terrain is None:
        return None
    return render_world_ascii(_world_state, terrain)


def _apply_waypoints(start_x: int, start_y: int, waypoints: str) -> tuple[int, int]:
    """Apply waypoints to calculate final position.

    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        waypoints: Path string (e.g., "wsss" = west, south, south, south).

    Returns:
        Tuple of (final_x, final_y) after applying all waypoints.
    """
    x, y = start_x, start_y
    for wp in waypoints:
        if wp == "n":
            y -= 1
        elif wp == "s":
            y += 1
        elif wp == "e":
            x += 1
        elif wp == "w":
            x -= 1
    return x, y


def _is_absolute_position(x: int, y: int) -> bool:
    """Check if position_update coordinates are absolute world coordinates.

    Position updates contain either:
    - Absolute world coordinates (after join/teleport): x,y >= 18
    - Viewport-relative coordinates (during movement): x,y < 18

    The viewport is 18x18, so coordinates within that range are viewport-relative
    and don't represent actual world position.

    Args:
        x: X coordinate from position_update.
        y: Y coordinate from position_update.

    Returns:
        True if coordinates are absolute world coordinates.
    """
    viewport_size = 18
    return x >= viewport_size or y >= viewport_size


def dispatch_world_state_update(decoded: protocol.BinaryMessage) -> None:
    """Dispatch decoded message to update world state and render ASCII.

    Handles:
    - radar_response: Update containers and mines, render ASCII
    - position_update: Update self position from absolute coords only
    - movement: Update self position from start coords (always absolute)
    - MovementResponse (0x3D): Update self position

    Args:
        decoded: Decoded binary protocol message.
    """
    match decoded:
        case {"msg_type": "radar_response", "containers": list(containers), "mines": list(mines)}:
            update_world_state_from_radar(containers, mines)
            ascii_view = render_world_state_ascii()
            if ascii_view is not None:
                log.info("[WorldState ASCII]\n%s", ascii_view)
        case {"msg_type": "position_update", "flags": int(flags), "x": int(x), "y": int(y)}:
            # flags=0x02 indicates self, flags=0x00 is other tanks
            is_self = (flags & 0x02) != 0
            if is_self and _is_absolute_position(x, y):
                update_world_state_from_position(x, y)
        case {
            "msg_type": "movement",
            "start_x": int(sx),
            "start_y": int(sy),
            "waypoints": str(waypoints),
            "is_self": True,
        }:
            # Calculate final position by applying waypoints to start position
            final_x, final_y = _apply_waypoints(sx, sy, waypoints)
            update_world_state_from_position(final_x, final_y)
        case {"msg_type": 0x3D, "x": int(x), "y": int(y)}:
            update_world_state_from_position(x, y)


__all__ = [
    "dispatch_world_state_update",
    "render_world_state_ascii",
    "reset_world_state",
    "update_world_state_from_position",
    "update_world_state_from_radar",
]
