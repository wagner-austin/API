"""Position, movement, tile, and map dispatch handlers."""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import protocol
from tankpit_bot.ledger.fuel_book import record_fuel_entry
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_tanks import (
    _update_tank_position,
    update_world_state_from_move_response_full,
)
from tankpit_bot.sniffer.world_state_tiles import (
    render_ascii_if_available,
    update_cache_tiles,
    update_overlay_tiles,
    update_terrain_tiles,
    update_viewport_entities,
)

log = get_logger(__name__)


# Dispatch — position / movement
# =============================================================================


def _resolve_waypoint_destination(
    start_x: int,
    start_y: int,
    waypoints: list[tuple[int, int]],
) -> tuple[int, int]:
    """Resolve the final destination from protocol waypoint tuples.

    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        waypoints: Waypoints list from the protocol movement decoder.

    Returns:
        Final destination after applying the waypoint tuple list.
    """
    final_x: int = start_x
    final_y: int = start_y
    if waypoints:
        final_x, final_y = waypoints[0]
    return (final_x, final_y)


def _handle_waypoint_movement(
    ws: WorldService,
    sx: int,
    sy: int,
    wps: list[tuple[int, int]],
) -> None:
    """Handle 0x47 waypoint movement for non-self tanks.

    Args:
        ws: World service instance.
        sx: Start X coordinate.
        sy: Start Y coordinate.
        wps: Waypoints list of (x, y) tuples from protocol decoder.
    """
    final_x, final_y = _resolve_waypoint_destination(sx, sy, wps)
    for tank in ws.world_state["tanks"].values():
        if tank["x"] == sx and tank["y"] == sy and not tank["is_self"]:
            _update_tank_position(ws, tank["tank_id"], final_x, final_y, "wire_0x47_movement")
            break


def _dispatch_protocol_movement_update(
    ws: WorldService,
    tank_id: int,
    start_x: int,
    start_y: int,
    waypoints: list[tuple[int, int]],
    path_tiles: int,
) -> bool:
    """Dispatch one decoded protocol ``0x47`` movement message.

    Args:
        ws: World service instance.
        tank_id: Moving tank id.
        start_x: Absolute movement start x.
        start_y: Absolute movement start y.
        waypoints: Absolute waypoint tuples from the protocol decoder.
        path_tiles: True wire step count of the commanded path.

    Returns:
        True after handling the movement.
    """
    self_state = ws.world_state["self_state"]
    is_self = self_state is not None and tank_id == self_state["tank_id"]
    if is_self:
        final_x, final_y = _resolve_waypoint_destination(start_x, start_y, waypoints)
        ws.update_world_state_from_position(final_x, final_y, "wire_0x47_movement")
        if path_tiles > 0:
            record_fuel_entry(book=ws.fuel_book, kind="walk", lo=-path_tiles, hi=0)
        render_ascii_if_available(ws, "SelfMovement")
    else:
        _handle_waypoint_movement(ws, start_x, start_y, waypoints)
    return True


# =============================================================================
# Dispatch — tile patches
# =============================================================================


def _dispatch_tile_patch_update(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch tile patch updates for cache, overlay, terrain, and viewport.

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {"msg_type": 0x4A, "updates": list(updates)}:
            update_terrain_tiles(ws, updates)
            return True
        case {"msg_type": 0x40, "updates": list(updates)}:
            update_overlay_tiles(ws, updates)
            return True
        case {"msg_type": 0x43, "updates": list(updates)}:
            update_cache_tiles(ws, updates)
            return True
        case {
            "msg_type": 0x5A,
            "viewport_left": int(viewport_left),
            "viewport_top": int(viewport_top),
            "entities": list(entities),
        }:
            update_viewport_entities(ws, viewport_left, viewport_top, entities)
            return True
    return False


# =============================================================================
# Dispatch — position updates (router)
# =============================================================================


def _dispatch_position_update(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch position and movement messages to update world state.

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {
            "msg_type": 0x47,
            "tank_id": int(tid),
            "start_x": int(sx),
            "start_y": int(sy),
            "waypoints": list(wps),
            "path_tiles": int(path_tiles),
        }:
            return _dispatch_protocol_movement_update(ws, tid, sx, sy, wps, path_tiles)
        case {
            "msg_type": 0x3D,
            "tank_id": int(tid),
            "x": int(x),
            "y": int(y),
            "team": int(team),
            "rank": int(rank),
            "direction": int(direction),
            "damage_state": int(dmg),
        }:
            # Protocol MovementResponse (0x3D) carries position +
            # direction (alive/dead) + damage + rank for every tank on
            # the map every ~2 seconds. Container's TankPositionStatus
            # equivalent was deleted 2026-06-19 -- this case now
            # surfaces all of the same fields plus the carrying byte.
            from tankpit_bot.sniffer.world_state_dispatch import _update_tank_from_position_status

            _update_tank_from_position_status(ws, tid, x, y, direction, dmg, rank, team)
            update_world_state_from_move_response_full(ws, tid, x, y, team, rank)
            render_ascii_if_available(ws, "MovementResponse")
            return True
        case _:
            if _dispatch_tile_patch_update(ws, decoded):
                return True
    return False


__all__: list[str] = []
