"""Protocol message dispatch for world state updates.

Routes decoded protocol messages to the appropriate world-state mutation
functions. This module is the only consumer of the ``_dispatch_*`` family;
the public entry point is ``dispatch_world_state_update``.
"""

from __future__ import annotations

from platform_core.logging import get_logger

import tankpit_bot.sniffer.world_state as _ws
from tankpit_bot import protocol
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.runtime_logging import emit_world
from tankpit_bot.sniffer.world_state_combat import (
    mark_combat_hit,
    mark_tank_killed,
    mark_teleport_landed,
)
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_container_pickup,
    update_world_state_from_fuel_total,
    update_world_state_from_tank_registry_container,
)
from tankpit_bot.sniffer.world_state_inventory import (
    update_inventory_from_gain,
    update_inventory_from_protocol,
    update_inventory_from_toggle,
)
from tankpit_bot.sniffer.world_state_radar import (
    handle_radar_ack,
    update_world_state_from_radar,
)
from tankpit_bot.sniffer.world_state_tanks import (
    _update_enemy_from_detection,
    _update_tank_position,
    update_world_state_from_move_response_full,
    update_world_state_from_tank_damage,
    update_world_state_from_tank_entry,
    update_world_state_from_tank_exit,
    update_world_state_from_tank_info,
    update_world_state_from_tank_registry,
    update_world_state_from_tank_status,
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
from tankpit_bot.state import add_mine, remove_mine

log = get_logger(__name__)


# =============================================================================
# Dispatch — resource / inventory
# =============================================================================


def _dispatch_resource_update(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch resource and inventory messages.

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {"msg_type": 0x2E, "fuel": int(fuel)} if fuel is not None:
            update_world_state_from_fuel_total(fuel)
            return True
        case {"msg_type": 0x44, "fuel_total": int(fuel_total)}:
            update_world_state_from_fuel_total(fuel_total)
            return True
        case {"msg_type": 0x64, "fuel_total": int(fuel_total)}:
            update_world_state_from_fuel_total(fuel_total)
            return True
        case {"msg_type": 0x49, "counts": list(counts), "enabled": list(enabled)}:
            update_inventory_from_protocol(counts, enabled)
            return True
        case {"msg_type": 0x67, "gained": list(gained)}:
            update_inventory_from_gain(gained)
            return True
        case {"msg_type": 0x74, "enabled": list(enabled)}:
            update_inventory_from_toggle(enabled)
            return True
        case {"msg_type": 0x46, "found": bool(found)}:
            handle_radar_ack(found)
            return True
    return False


# =============================================================================
# Dispatch — tank state
# =============================================================================


def _dispatch_tank_update(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch tank-related messages to update world state.

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {"msg_type": 0x28, "tank_id": int(tid), "x": int(tx), "y": int(ty), "name": str(name)}:
            update_world_state_from_tank_entry(tid, tx, ty, name)
            return True
        case {"msg_type": 0x21, "tank_id": int(tid), "team": int(team), "name": str(name)}:
            update_world_state_from_tank_info(tid, team, name)
            return True
        case {
            "msg_type": 0x3E,
            "tank_id": int(tid),
            "team": int(team),
            "rank": int(rank),
            "name": str(name),
        }:
            update_world_state_from_tank_status(tid, team, rank, name)
            return True
        case {"msg_type": 0x2E, "tank_id": int(tid), "damage_state": int(dmg)}:
            update_world_state_from_tank_damage(tid, dmg)
            return True
        case {"msg_type": 0x58, "tank_id": int(tid)}:
            update_world_state_from_tank_exit(tid)
            return True
        case {
            "msg_type": 0x48,
            "tank_id": int(tid),
            "x": int(x),
            "y": int(y),
            "team": int(team),
            "rank": int(rank),
        }:
            _update_enemy_from_detection(tid, x, y, team, rank)
            return True
        case {
            "msg_type": 0x41,
            "victim_id": int(vid),
        }:
            _update_tank_position(vid, 0, 0)
            mark_tank_killed(vid)
            log.info("DEACTIVATED: tank=%d killed, position invalidated", vid)
            return True
    return False


# =============================================================================
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
# Dispatch — tank events (container-decoded)
# =============================================================================


def _dispatch_tank_event(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch tank lifecycle events (leave, deactivation, damage, update).

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {
            "msg_type": "tank_update_compact" | "tank_update_extended" | "tank_update_full",
            "flags": int(flags),
            "tank_id": int(tid),
            "status_data": bytes(sd),
        }:
            if flags == 0xCD:
                return True
            if len(sd) >= 2:
                _update_tank_position(tid, sd[0], sd[1])
            return True
        case {
            "msg_type": "tank_status_short",
            "tank_id": int(tid),
            "damage_state": int(dmg),
        }:
            update_world_state_from_tank_damage(tid, dmg)
            return True
        case {"msg_type": "tank_leave", "tank_id": int(tid)}:
            update_world_state_from_tank_exit(tid)
            return True
        case {"msg_type": "deactivation_kill", "victim_id": int(vid)}:
            known_tanks = list(_ws._world_state["tanks"].keys())
            log.info(
                "DEACTIVATION_KILL: victim_id=%d (0x%04X) known_tanks=%s",
                vid,
                vid,
                known_tanks[:10],
            )
            _update_tank_position(vid, 0, 0)
            mark_tank_killed(vid)
            return True
        case {"msg_type": "deactivation_death", "killer_id": int(kid)}:
            log.info("DEACTIVATION_DEATH: killed by tank=%d", kid)
            return True
    return False


# =============================================================================
# Dispatch — container messages (mines, registry, combat, pickup)
# =============================================================================


def _dispatch_mine_placement(
    mine_type: int,
    tank_id: int,
    positions: list[tuple[int, int]],
) -> bool:
    """Dispatch tunneled mine placement into world state.

    Args:
        mine_type: Mine type from protocol payload.
        tank_id: ID of the placing tank.
        positions: Absolute mine coordinates.

    Returns:
        True after attempting to apply the placement.
    """
    self_state = _ws._world_state["self_state"]
    team: int | None = None
    if self_state is not None and self_state["tank_id"] == tank_id:
        team = self_state["team"]
    else:
        tank_state = _ws._world_state["tanks"].get(str(tank_id))
        if tank_state is not None:
            team = tank_state["team"]
    if team is None:
        return True
    timestamp_ms = get_current_time_ms()
    for x, y in positions:
        _ws._world_state = add_mine(
            _ws._world_state,
            x,
            y,
            mine_type,
            tank_id,
            team,
            timestamp_ms,
        )
    return True


def _dispatch_mine_detonation(
    positions: list[tuple[int, int]],
) -> bool:
    """Dispatch tunneled mine detonation into world state.

    Args:
        positions: Absolute mine coordinates removed by the detonation.

    Returns:
        True after applying the removals.
    """
    timestamp_ms = get_current_time_ms()
    for x, y in positions:
        _ws._world_state = remove_mine(_ws._world_state, x, y, timestamp_ms)
    return True


def _dispatch_container_message(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch container-level messages (tank_registry, tank_update, etc.).

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {
            "msg_type": 0x4B,
            "mine_type": int(mine_type),
            "tank_id": int(tank_id),
            "positions": list(positions),
        }:
            return _dispatch_mine_placement(mine_type, tank_id, positions)
        case {"msg_type": 0x45, "positions": list(positions)}:
            return _dispatch_mine_detonation(positions)
        case {
            "msg_type": "tank_registry",
            "is_container": True,
            "container_y": int(cy),
            "container_viewport_x": int(cvx),
        }:
            update_world_state_from_tank_registry_container(cy, cvx)
            log.info("Container from tank_registry: y=%d vx=%d", cy, cvx)
            return True
        case {"msg_type": "container_pickup", "x": int(x), "y": int(y)}:
            update_world_state_from_container_pickup(x, y)
            return True
        case {"msg_type": "teleport_landed"}:
            emit_world("TELEPORT_LANDED: server confirmed teleport")
            mark_teleport_landed()
            return True
        case {
            "msg_type": "combat_hit",
            "attacker_id": int(aid),
            "direction": int(),
            "is_outgoing": bool(),
            "combat_data": bytes(cdata),
        }:
            self_state = _ws._world_state["self_state"]
            if self_state is not None and aid == self_state["tank_id"]:
                weapon_byte = cdata[-1] if len(cdata) > 0 else 0
                log.info("OUR_SHOT: weapon_byte=%d data=%s", weapon_byte, cdata.hex())
                mark_combat_hit(weapon_byte)
            return True
        case {
            "msg_type": "tank_registry",
            "is_container": False,
            "tank_id": int(tid),
            "tank_name": str(name),
            "team": str(team_str),
            "military_rank": int(rank),
            "is_bot": bool(is_bot),
            "tank_y": int(ty),
            "tank_viewport_x": int(tvx),
        }:
            update_world_state_from_tank_registry(tid, name, team_str, rank, is_bot, ty, tvx)
            return True
    return _dispatch_tank_event(decoded)


# =============================================================================
# Dispatch — world state blob
# =============================================================================


def _parse_world_state_blob(wd: bytes) -> None:
    """Parse world_state blob from map response to extract all tank positions.

    Format (verified from world_state_dump.bin):
    - [terrain_count:2 LE] - number of terrain delta bytes
    - [terrain_count terrain delta bytes]
    - Repeated 5-byte tank entries until end:
      [x:1][y:1][id_lo:1][id_hi:1][packed:1]
      where tank_id = id_lo + id_hi*256 (LE), team = packed & 3, rank = (packed>>4) & 15

    Args:
        wd: Raw world state blob bytes.
    """
    if len(wd) < 2:
        return

    terrain_count = wd[0] | (wd[1] << 8)
    tank_data_start = 2 + terrain_count

    if tank_data_start > len(wd):
        log.warning("WorldState blob too short: %d bytes, terrain_count=%d", len(wd), terrain_count)
        return

    remaining = wd[tank_data_start:]
    num_tanks = len(remaining) // 5
    if num_tanks == 0:
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

    log.info(
        "MAP_POSITIONS: parsed %d tanks from world_state blob (%d bytes, %d terrain)",
        updated,
        len(wd),
        terrain_count,
    )


def _update_map_tank(tank_id: int, x: int, y: int, team: int, rank: int) -> None:
    """Update a tank's position/team/rank from map data.

    Preserves existing name and is_bot fields if the tank is already known.

    Args:
        tank_id: Tank identifier.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Team number.
        rank: Military rank.
    """
    from tankpit_bot.state import update_tank_from_registry

    ts = get_current_time_ms()
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
    )


# =============================================================================
# Public entry point
# =============================================================================


def dispatch_world_state_update(decoded: protocol.BinaryMessage) -> None:
    """Dispatch decoded message to update world state, inventory, and render ASCII.

    Delegates to specialized dispatchers for resources, tanks, positions,
    and container messages.

    Args:
        decoded: Decoded binary protocol message.
    """
    if _dispatch_resource_update(decoded):
        return
    if _dispatch_tank_update(decoded):
        return
    if _dispatch_position_update(decoded):
        return
    if _dispatch_container_message(decoded):
        return

    match decoded:
        case {"msg_type": "world_state", "world_data": bytes(wd)}:
            _parse_world_state_blob(wd)
            return
        case {"msg_type": 0x4F, "containers": list(containers), "mines": list(mines)}:
            if not containers and not mines:
                _ws._mark_pending_radar_empty_delta()
            else:
                update_world_state_from_radar(containers, mines)
                render_ascii_if_available("Radar")
            return
        case {"msg_type": "radar_response", "containers": list(containers), "mines": list(mines)}:
            update_world_state_from_radar(containers, mines)
            render_ascii_if_available("Radar")


__all__ = [
    "dispatch_world_state_update",
]
