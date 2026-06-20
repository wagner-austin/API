"""Position, movement, tile, and map dispatch handlers."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_int
from platform_core.logging import get_logger
from typing_extensions import TypedDict

from tankpit_bot import browser, protocol
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_fuel_dots,
)
from tankpit_bot.sniffer.world_state_tanks import (
    _update_tank_position,
    update_world_state_from_move_response_full,
)
from tankpit_bot.sniffer.world_state_tiles import (
    is_absolute_position,
    render_ascii_if_available,
    update_cache_tiles,
    update_overlay_tiles,
    update_terrain_tiles,
    update_viewport_entities,
)

log = get_logger(__name__)


class MapPositionsParsedDiagnosticDict(TypedDict):
    """Structured payload for the ``map_positions_parsed`` diagnostic event."""

    tank_count: int
    blob_bytes: int
    fuel_dot_count: int


def encode_map_positions_parsed_diagnostic(
    payload: MapPositionsParsedDiagnosticDict,
) -> JSONObject:
    """Encode a ``map_positions_parsed`` diagnostic payload to JSON."""
    return {
        "tank_count": payload["tank_count"],
        "blob_bytes": payload["blob_bytes"],
        "fuel_dot_count": payload["fuel_dot_count"],
    }


def decode_map_positions_parsed_diagnostic(
    data: JSONObject,
) -> MapPositionsParsedDiagnosticDict:
    """Decode a ``map_positions_parsed`` diagnostic payload from JSON."""
    return MapPositionsParsedDiagnosticDict(
        tank_count=require_int(data, "tank_count"),
        blob_bytes=require_int(data, "blob_bytes"),
        fuel_dot_count=require_int(data, "fuel_dot_count"),
    )


def emit_map_positions_parsed_diagnostic(payload: MapPositionsParsedDiagnosticDict) -> None:
    """Emit one ``map_positions_parsed`` structured diagnostic event."""
    emit_diagnostic(
        diagnostic_kind="map_positions_parsed",
        tank_count=payload["tank_count"],
        blob_bytes=payload["blob_bytes"],
        fuel_dot_count=payload["fuel_dot_count"],
    )


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
            _update_tank_position(ws, tank["tank_id"], final_x, final_y)
            break


def _dispatch_binary_position_update(
    ws: WorldService,
    flags: int,
    tank_id: int,
    x: int,
    y: int,
) -> bool:
    """Dispatch one decoded ``position_update`` message.

    Args:
        ws: World service instance.
        flags: Position-update flags.
        tank_id: Tank identifier.
        x: Reported x coordinate.
        y: Reported y coordinate.

    Returns:
        True after handling the position update.
    """
    is_self = (flags & 0x02) != 0
    if is_self and is_absolute_position(x, y):
        ws.update_world_state_from_position(x, y)
        render_ascii_if_available(ws, "Enter/Teleport")
    elif not is_self and is_absolute_position(x, y):
        _update_tank_position(ws, tank_id, x, y)
    return True


def _dispatch_protocol_movement_update(
    ws: WorldService,
    tank_id: int,
    start_x: int,
    start_y: int,
    waypoints: list[tuple[int, int]],
) -> bool:
    """Dispatch one decoded protocol ``0x47`` movement message.

    Args:
        ws: World service instance.
        tank_id: Moving tank id.
        start_x: Absolute movement start x.
        start_y: Absolute movement start y.
        waypoints: Absolute waypoint tuples from the protocol decoder.

    Returns:
        True after handling the movement.
    """
    self_state = ws.world_state["self_state"]
    is_self = self_state is not None and tank_id == self_state["tank_id"]
    if is_self:
        final_x, final_y = _resolve_waypoint_destination(start_x, start_y, waypoints)
        ws.update_world_state_from_position(final_x, final_y)
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
            "msg_type": 0x4F,
            "cache_updates": list(cache_updates),
            "overlay_updates": list(overlay_updates),
        }:
            update_cache_tiles(ws, cache_updates)
            update_overlay_tiles(ws, overlay_updates)
            ws.mark_pending_radar_cache_refresh()
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
            "msg_type": "position_update",
            "flags": int(flags),
            "tank_id": int(tid),
            "x": int(x),
            "y": int(y),
        }:
            return _dispatch_binary_position_update(ws, flags, tid, x, y)
        case {
            "msg_type": 0x47,
            "tank_id": int(tid),
            "start_x": int(sx),
            "start_y": int(sy),
            "waypoints": list(wps),
        }:
            return _dispatch_protocol_movement_update(ws, tid, sx, sy, wps)
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


# =============================================================================
# Dispatch — world state blob
# =============================================================================


def _decode_fuel_dot_layer(section: bytes) -> list[tuple[int, int]]:
    """Decode the MAP_DATA fuel-dot layer into world coordinates.

    Args:
        section: Raw dot-layer bytes.

    Returns:
        Decoded ``(x, y)`` world coordinates of every fuel dot.
    """
    x, y = 1, 1
    dots: list[tuple[int, int]] = []
    for step in section:
        x += step
        if x > 255:
            y += 1
            x %= 256
        if step != 255:
            dots.append((x, y))
    return dots


def _parse_world_state_blob(ws: WorldService, wd: bytes) -> None:
    """Parse world_state blob from map response: fuel dots + tank positions.

    Args:
        ws: World service instance.
        wd: Raw world state blob bytes.
    """
    if len(wd) < 2:
        return

    dot_section_bytes = wd[0] | (wd[1] << 8)
    tank_data_start = 2 + dot_section_bytes

    if tank_data_start > len(wd):
        log.warning(
            "WorldState blob too short: %d bytes, dot_section_bytes=%d",
            len(wd),
            dot_section_bytes,
        )
        return

    fuel_dots = _decode_fuel_dot_layer(wd[2:tank_data_start])
    update_world_state_from_fuel_dots(ws, fuel_dots)

    remaining = wd[tank_data_start:]
    num_tanks = len(remaining) // 5
    if num_tanks == 0:
        ws.mark_map_data_processed()
        return

    updated = 0
    for i in range(num_tanks):
        entry = remaining[i * 5 : i * 5 + 5]
        x = entry[0]
        y = entry[1]
        tank_id = entry[2] | (entry[3] << 8)
        packed = entry[4]
        team = packed & 0x03
        rank = (packed >> 4) & 0x0F

        _update_map_tank(ws, tank_id, x, y, team, rank)
        updated += 1

    emit_map_positions_parsed_diagnostic(
        MapPositionsParsedDiagnosticDict(
            tank_count=updated,
            blob_bytes=len(wd),
            fuel_dot_count=len(fuel_dots),
        )
    )
    ws.mark_map_data_processed()


def _update_map_tank(
    ws: WorldService,
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
) -> None:
    """Update a tank's position/team/rank from map data.

    Args:
        ws: World service instance.
        tank_id: Tank identifier.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Team number.
        rank: Military rank.
    """
    from tankpit_bot.state.mutations import apply_tank_observation
    from tankpit_bot.state.types import make_tank_observation

    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = ws.world_state["tanks"].get(key)
    if existing and (existing["x"] != x or existing["y"] != y):
        emit_diagnostic(
            diagnostic_kind="map_tank_drift",
            tank_name=existing["name"] or key,
            tank_id=tank_id,
            map_x=x,
            map_y=y,
            was_x=existing["x"],
            was_y=existing["y"],
            delta_x=x - existing["x"],
            delta_y=y - existing["y"],
        )
    # Map snapshot updates carry position but are not wire-sourced;
    # apply_tank_observation enforces that neither last_wire_seen_ms
    # nor last_position_update_ms advance from this observation.
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=False,
        storage_source="world_state",
        position=(x, y),
        team=team,
        rank=rank,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


# =============================================================================


__all__: list[str] = []
