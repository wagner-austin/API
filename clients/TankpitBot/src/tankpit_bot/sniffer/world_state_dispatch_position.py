"""Position, movement, tile, and map dispatch handlers."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_int
from platform_core.logging import get_logger
from typing_extensions import TypedDict

import tankpit_bot.sniffer.world_state as _ws
from tankpit_bot import browser, protocol
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_fuel_dots,
)
from tankpit_bot.sniffer.world_state_tanks import (
    _update_tank_position,
    update_world_state_from_move_response_full,
)
from tankpit_bot.sniffer.world_state_tiles import (
    apply_waypoints,
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


def _handle_waypoint_movement(sx: int, sy: int, wps: list[tuple[int, int]]) -> None:
    """Handle 0x47 waypoint movement for non-self tanks.

    Args:
        sx: Start X coordinate.
        sy: Start Y coordinate.
        wps: Waypoints list of (x, y) tuples from protocol decoder.
    """
    final_x, final_y = _resolve_waypoint_destination(sx, sy, wps)
    for tank in _ws._world_state["tanks"].values():
        if tank["x"] == sx and tank["y"] == sy and not tank["is_self"]:
            _update_tank_position(tank["tank_id"], final_x, final_y)
            break


def _dispatch_container_movement(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch container-decoded movement messages (msg_type="movement").

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {
            "msg_type": "movement",
            "start_x": int(sx),
            "start_y": int(sy),
            "waypoints": str(wps),
            "is_self": True,
        }:
            fx, fy = apply_waypoints(sx, sy, wps)
            _ws.update_world_state_from_position(fx, fy)
            render_ascii_if_available("SelfMovement")
            return True
        case {
            "msg_type": "movement",
            "start_x": int(sx),
            "start_y": int(sy),
            "player_id": int(pid),
            "waypoints": str(wps),
            "is_self": False,
        }:
            from tankpit_bot.sniffer.player_tracking import _player_id_mapper

            resolved_tid = _player_id_mapper.get_tank_id(pid)
            if resolved_tid is not None:
                fx, fy = apply_waypoints(sx, sy, wps)
                _update_tank_position(resolved_tid, fx, fy)
            return True
    return False


def _dispatch_binary_position_update(
    flags: int,
    tank_id: int,
    x: int,
    y: int,
) -> bool:
    """Dispatch one decoded ``position_update`` message.

    Args:
        flags: Position-update flags.
        tank_id: Tank identifier.
        x: Reported x coordinate.
        y: Reported y coordinate.

    Returns:
        True after handling the position update.
    """
    is_self = (flags & 0x02) != 0
    if is_self and is_absolute_position(x, y):
        _ws.update_world_state_from_position(x, y)
        render_ascii_if_available("Enter/Teleport")
    elif not is_self and is_absolute_position(x, y):
        _update_tank_position(tank_id, x, y)
    return True


def _dispatch_protocol_movement_update(
    tank_id: int,
    start_x: int,
    start_y: int,
    waypoints: list[tuple[int, int]],
) -> bool:
    """Dispatch one decoded protocol ``0x47`` movement message.

    Args:
        tank_id: Moving tank id.
        start_x: Absolute movement start x.
        start_y: Absolute movement start y.
        waypoints: Absolute waypoint tuples from the protocol decoder.

    Returns:
        True after handling the movement.
    """
    self_state = _ws._world_state["self_state"]
    is_self = self_state is not None and tank_id == self_state["tank_id"]
    if is_self:
        final_x, final_y = _resolve_waypoint_destination(start_x, start_y, waypoints)
        _ws.update_world_state_from_position(final_x, final_y)
        render_ascii_if_available("SelfMovement")
    else:
        _handle_waypoint_movement(start_x, start_y, waypoints)
    return True


# =============================================================================
# Dispatch — tile patches
# =============================================================================


def _dispatch_tile_patch_update(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch tile patch updates for cache, overlay, terrain, and viewport.

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {"msg_type": 0x4A, "updates": list(updates)}:
            update_terrain_tiles(updates)
            return True
        case {"msg_type": 0x40, "updates": list(updates)}:
            update_overlay_tiles(updates)
            return True
        case {"msg_type": 0x43, "updates": list(updates)}:
            update_cache_tiles(updates)
            return True
        case {
            "msg_type": 0x4F,
            "cache_updates": list(cache_updates),
            "overlay_updates": list(overlay_updates),
        }:
            update_cache_tiles(cache_updates)
            update_overlay_tiles(overlay_updates)
            _ws._mark_pending_radar_cache_refresh()
            return True
        case {
            "msg_type": 0x5A,
            "viewport_left": int(viewport_left),
            "viewport_top": int(viewport_top),
            "entities": list(entities),
        }:
            update_viewport_entities(viewport_left, viewport_top, entities)
            return True
    return False


# =============================================================================
# Dispatch — position updates (router)
# =============================================================================


def _dispatch_position_update(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch position and movement messages to update world state.

    Args:
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
            return _dispatch_binary_position_update(flags, tid, x, y)
        case {
            "msg_type": 0x47,
            "tank_id": int(tid),
            "start_x": int(sx),
            "start_y": int(sy),
            "waypoints": list(wps),
        }:
            return _dispatch_protocol_movement_update(tid, sx, sy, wps)
        case {
            "msg_type": 0x3D,
            "tank_id": int(tid),
            "x": int(x),
            "y": int(y),
            "team": int(team),
            "rank": int(rank),
        }:
            update_world_state_from_move_response_full(tid, x, y, team, rank)
            render_ascii_if_available("MovementResponse")
            return True
        case _:
            if _dispatch_tile_patch_update(decoded):
                return True
    return _dispatch_container_movement(decoded)


# =============================================================================
# Dispatch — world state blob
# =============================================================================


def _decode_fuel_dot_layer(section: bytes) -> list[tuple[int, int]]:
    """Decode the MAP_DATA fuel-dot layer into world coordinates.

    The algorithm mirrors the live client's ``Ig.h`` parser exactly
    (tpclient-b45bd1ebc9c0c668.js): a cursor starts at world (1, 1);
    every byte advances x, wrapping past 255 to the next row; byte 255
    is a pure skip, any other byte also drops a dot at the resulting
    coordinate. The client draws these dots as the map's yellow fuel
    pixels. Verified 2026-06-11 across 15 captured sessions: fuel
    pickups land on dots 33-71% by gain bucket vs ~1% chance; radar
    equipment at exact coordinates 0/184.

    Args:
        section: Raw dot-layer bytes (the length-prefixed first section
            of the MAP_DATA blob, prefix excluded).

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


def _parse_world_state_blob(wd: bytes) -> None:
    """Parse world_state blob from map response: fuel dots + tank positions.

    Format (verified from world_state_dump.bin + client Ig.h parser):
    - [dot_section_bytes:2 LE] - length of the fuel-dot layer
    - [dot_section_bytes fuel-dot skip-RLE bytes] (see
      :func:`_decode_fuel_dot_layer`)
    - Repeated 5-byte tank entries until end:
      [x:1][y:1][id_lo:1][id_hi:1][packed:1]
      where tank_id = id_lo + id_hi*256 (LE), team = packed & 3, rank = (packed>>4) & 15
      (bits 2-3 of packed are an undecoded client field; the client
      stores them as the map tank's ``u``)

    On successful ingest -- including the zero-tank case -- this marks the
    MAP_DATA processed signal via :func:`mark_map_data_processed` so the tick
    loop's ``map_open`` completion gate fires on the AUTHORITATIVE response
    rather than any incidental sync.

    Args:
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
    update_world_state_from_fuel_dots(fuel_dots)

    remaining = wd[tank_data_start:]
    num_tanks = len(remaining) // 5
    if num_tanks == 0:
        # Empty MAP_DATA is still an authoritative server response to
        # ``map_open``; the HFSM gate must fire so replanning can resume.
        _ws.mark_map_data_processed()
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

        _update_map_tank(tank_id, x, y, team, rank)
        updated += 1

    emit_map_positions_parsed_diagnostic(
        MapPositionsParsedDiagnosticDict(
            tank_count=updated,
            blob_bytes=len(wd),
            fuel_dot_count=len(fuel_dots),
        )
    )
    _ws.mark_map_data_processed()


def _update_map_tank(tank_id: int, x: int, y: int, team: int, rank: int) -> None:
    """Update a tank's position/team/rank from map data.

    Preserves existing name and is_bot fields if the tank is already known.

    The map blob is server-authoritative and lists EVERY tank, including
    departed afterimages at stale cached positions (raw-capture
    2026-06-13: ghost 517 was re-listed 49 times over 425s while never
    once appearing on a per-tank wire path). It refreshes ``timestamp_ms``
    so acquisition can still navigate toward map-listed enemies, but
    passes ``wire_present=False`` so it never advances
    ``last_wire_seen_ms`` -- the map is not evidence the tank is in view,
    and the kill-shot gate must not fire at a map-only afterimage.

    Args:
        tank_id: Tank identifier.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Team number.
        rank: Military rank.
    """
    from tankpit_bot.state import update_tank_from_registry

    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = _ws._world_state["tanks"].get(key)
    _ws._world_state = update_tank_from_registry(
        _ws._world_state,
        tank_id,
        team,
        existing["name"] if existing else "",
        rank,
        existing["is_bot"] if existing else False,
        x,
        y,
        "world_state",
        ts,
        wire_present=False,
    )


# =============================================================================


__all__: list[str] = []
