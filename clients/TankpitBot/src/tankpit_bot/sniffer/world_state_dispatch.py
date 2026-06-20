"""Protocol message dispatch for world state updates.

Routes decoded protocol messages to the appropriate world-state mutation
functions. This module is the only consumer of the ``_dispatch_*`` family;
the public entry point is ``dispatch_world_state_update``.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import browser, protocol
from tankpit_bot.runtime_logging import emit_diagnostic, emit_world
from tankpit_bot.sniffer.world_service import WorldService
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
from tankpit_bot.sniffer.world_state_dispatch_position import (
    _dispatch_position_update,
    _parse_world_state_blob,
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
    update_world_state_from_tank_damage,
    update_world_state_from_tank_entry,
    update_world_state_from_tank_exit,
    update_world_state_from_tank_info,
    update_world_state_from_tank_registry,
    update_world_state_from_tank_status,
)
from tankpit_bot.sniffer.world_state_tiles import (
    render_ascii_if_available,
)
from tankpit_bot.state import add_mine, remove_mine

log = get_logger(__name__)


def _update_tank_from_position_status(
    ws: WorldService,
    tank_id: int,
    x: int,
    y: int,
    direction: int,
    damage_state: int,
    rank: int,
    team: int,
) -> None:
    """Update tank from 0x3D MovementResponse: position + direction + damage + rank.

    Carries position, direction (alive/dead), damage, and rank for
    every tank on the map. Direction >= 32 indicates a corpse.

    Args:
        ws: World service instance.
        tank_id: Tank id.
        x: Map x position.
        y: Map y position.
        direction: Sprite direction (0-31 alive, 32-33 dead).
        damage_state: Damage tier (0-3).
        rank: Military rank (0-8).
        team: Team id (0-3).
    """
    from tankpit_bot.state.mutations import apply_tank_observation
    from tankpit_bot.state.types import make_tank_observation

    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        position=(x, y),
        team=team,
        rank=rank,
        damage_state=damage_state,
        direction=direction,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


def _dispatch_shoot_event(
    ws: WorldService,
    shooter_id: int,
    sx: int,
    sy: int,
    tx: int,
    ty: int,
    weapon: int,
) -> None:
    """Apply a 0x53 ShootEvent to world state.

    Own shot -> tile-occupancy hit detection: lookup tank at target tile,
    record victim id. Enemy shot -> position update from source tile.

    Args:
        ws: World service instance.
        shooter_id: Who fired the shot.
        sx: Shooter source tile X.
        sy: Shooter source tile Y.
        tx: Shot target tile X.
        ty: Shot target tile Y.
        weapon: Weapon byte (0=single, 1=dual, 2=missile, 3=homing).
    """
    self_state = ws.world_state["self_state"]
    own_tank_id = self_state["tank_id"] if self_state is not None else -1
    if shooter_id == own_tank_id:
        victim_id = _find_tank_at_tile(ws, tx, ty, exclude_id=own_tank_id)
        log.info(
            "OUR_SHOT: weapon=%d src=(%d,%d) tgt=(%d,%d) victim_id=%d",
            weapon,
            sx,
            sy,
            tx,
            ty,
            victim_id,
        )
        mark_combat_hit(ws, weapon, victim_id)
    elif shooter_id > 0:
        _update_tank_position(ws, shooter_id, sx, sy)


def _find_tank_at_tile(ws: WorldService, x: int, y: int, exclude_id: int) -> int:
    """Return the tank id occupying (x, y), or -1 if the tile is empty.

    Used by the ShootEvent dispatch to determine whether our shot landed
    on a tank (hit) or empty terrain (miss). Tile occupancy is the
    authoritative wire-side hit signal per JS Gg.prototype.h.

    Args:
        ws: World service instance.
        x: Tile x coordinate from ShootEvent.target_x.
        y: Tile y coordinate from ShootEvent.target_y.
        exclude_id: Tank id to skip (typically our own tank, since
            the bot never shoots itself).

    Returns:
        Tank id at the tile, or -1 if no tracked tank occupies it.
    """
    for tid_str, tank in ws.world_state["tanks"].items():
        if tank["x"] == x and tank["y"] == y and int(tid_str) != exclude_id:
            return tank["tank_id"]
    return -1


# =============================================================================
# Dispatch — resource / inventory
# =============================================================================


def _dispatch_resource_update(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch resource and inventory messages.

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {"msg_type": 0x2E, "fuel": int(fuel)} if fuel is not None:
            update_world_state_from_fuel_total(ws, fuel)
            return True
        case {"msg_type": 0x44, "fuel_total": int(fuel_total)}:
            update_world_state_from_fuel_total(ws, fuel_total)
            return True
        case {"msg_type": 0x64, "fuel_total": int(fuel_total)}:
            update_world_state_from_fuel_total(ws, fuel_total)
            return True
        case {"msg_type": 0x49, "counts": list(counts), "enabled": list(enabled)}:
            update_inventory_from_protocol(ws, counts, enabled)
            return True
        case {"msg_type": 0x67, "gained": list(gained)}:
            update_inventory_from_gain(ws, gained)
            return True
        case {"msg_type": 0x74, "enabled": list(enabled)}:
            update_inventory_from_toggle(ws, enabled)
            return True
        case {"msg_type": 0x46, "found": bool(found)}:
            handle_radar_ack(ws, found)
            return True
    return False


# =============================================================================
# Dispatch — tank state
# =============================================================================


def _dispatch_tank_update(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch tank-related messages to update world state.

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {
            "msg_type": 0x28,
            "tank_id": int(tid),
            "team": int(team),
            "rank": int(rank),
            "x": int(tx),
            "y": int(ty),
        }:
            update_world_state_from_tank_entry(ws, tid, team, rank, tx, ty)
            return True
        case {"msg_type": 0x21, "tank_id": int(tid), "team": int(team), "name": str(name)}:
            update_world_state_from_tank_info(ws, tid, team, name)
            return True
        case {
            "msg_type": 0x3E,
            "tank_id": int(tid),
            "team": int(team),
            "rank": int(rank),
            "name": str(name),
        }:
            update_world_state_from_tank_status(ws, tid, team, rank, name)
            return True
        case {"msg_type": 0x2E, "tank_id": int(tid), "damage_state": int(dmg)}:
            update_world_state_from_tank_damage(ws, tid, dmg)
            return True
        case {"msg_type": 0x58, "tank_id": int(tid)}:
            update_world_state_from_tank_exit(ws, tid)
            return True
        case {
            "msg_type": 0x53,
            "shooter_id": int(shooter_id),
            "source_x": int(sx),
            "source_y": int(sy),
            "target_x": int(tx),
            "target_y": int(ty),
            "weapon": int(weapon),
        }:
            _dispatch_shoot_event(ws, shooter_id, sx, sy, tx, ty, weapon)
            return True
        case {
            "msg_type": 0x48,
            "tank_id": int(tid),
            "x": int(x),
            "y": int(y),
            "team": int(team),
            "rank": int(rank),
        }:
            _update_enemy_from_detection(ws, tid, x, y, team, rank)
            return True
        case {
            "msg_type": 0x41,
            "victim_id": int(vid),
            "killer_id": int(kid),
        }:
            mark_tank_killed(ws, vid)
            _update_tank_position(ws, vid, 0, 0)
            emit_diagnostic(
                diagnostic_kind="tank_deactivated",
                origin="protocol_0x41",
                victim_id=vid,
                killer_id=kid,
            )
            log.info("DEACTIVATED: tank=%d killed, position invalidated", vid)
            return True
    return False


# =============================================================================
# Dispatch — tank events (container-decoded)
# =============================================================================


def _dispatch_tank_event(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch tank lifecycle events (leave, deactivation, damage, update).

    Args:
        ws: World service instance.
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
                _update_tank_position(ws, tid, sd[0], sd[1])
            return True
        case {
            "msg_type": "tank_status_short",
            "tank_id": int(tid),
            "damage_state": int(dmg),
        }:
            update_world_state_from_tank_damage(ws, tid, dmg)
            return True
        case {"msg_type": "tank_leave", "tank_id": int(tid)}:
            update_world_state_from_tank_exit(ws, tid)
            return True
        case {"msg_type": "deactivation_death", "killer_id": int(kid)}:
            emit_diagnostic(
                diagnostic_kind="tank_deactivated",
                origin="container_death",
                victim_id=-1,
                killer_id=kid,
            )
            log.info("DEACTIVATION_DEATH: killed by tank=%d", kid)
            return True
    return False


# =============================================================================
# Dispatch — container messages (mines, registry, combat, pickup)
# =============================================================================


def _dispatch_mine_placement(
    ws: WorldService,
    mine_type: int,
    tank_id: int,
    positions: list[tuple[int, int]],
) -> bool:
    """Dispatch tunneled mine placement into world state.

    Args:
        ws: World service instance.
        mine_type: Mine type from protocol payload.
        tank_id: ID of the placing tank.
        positions: Absolute mine coordinates.

    Returns:
        True after attempting to apply the placement.
    """
    self_state = ws.world_state["self_state"]
    team: int | None = None
    if self_state is not None and self_state["tank_id"] == tank_id:
        team = self_state["team"]
    else:
        tank_state = ws.world_state["tanks"].get(str(tank_id))
        if tank_state is not None:
            team = tank_state["team"]
    if team is None:
        return True
    timestamp_ms = browser.get_current_time_ms()
    for x, y in positions:
        ws.world_state = add_mine(
            ws.world_state,
            x,
            y,
            mine_type,
            tank_id,
            team,
            timestamp_ms,
        )
    return True


def _dispatch_mine_detonation(
    ws: WorldService,
    positions: list[tuple[int, int]],
) -> bool:
    """Dispatch tunneled mine detonation into world state.

    Args:
        ws: World service instance.
        positions: Absolute mine coordinates removed by the detonation.

    Returns:
        True after applying the removals.
    """
    timestamp_ms = browser.get_current_time_ms()
    for x, y in positions:
        ws.world_state = remove_mine(ws.world_state, x, y, timestamp_ms)
    return True


def _dispatch_container_message(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch container-level messages (tank_registry, tank_update, etc.).

    Args:
        ws: World service instance.
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
            return _dispatch_mine_placement(ws, mine_type, tank_id, positions)
        case {"msg_type": 0x45, "positions": list(positions)}:
            return _dispatch_mine_detonation(ws, positions)
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
            update_world_state_from_container_pickup(ws, x, y)
            return True
        case {"msg_type": "teleport_landed"}:
            emit_world("TELEPORT_LANDED: server confirmed teleport")
            mark_teleport_landed(ws)
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
            update_world_state_from_tank_registry(ws, tid, name, team_str, rank, is_bot, ty, tvx)
            return True
    return _dispatch_tank_event(ws, decoded)


# =============================================================================
# Public entry point
# =============================================================================


def dispatch_world_state_update(ws: WorldService, decoded: protocol.BinaryMessage) -> None:
    """Dispatch decoded message to update world state, inventory, and render ASCII.

    Delegates to specialized dispatchers for resources, tanks, positions,
    and container messages.

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.
    """
    if _dispatch_resource_update(ws, decoded):
        return
    if _dispatch_tank_update(ws, decoded):
        return
    if _dispatch_position_update(ws, decoded):
        return
    if _dispatch_container_message(ws, decoded):
        return

    match decoded:
        case {"msg_type": 0x52, "reset_action": int(), "error_code": int(error_code)}:
            ws.last_command_error = error_code
            emit_diagnostic(
                diagnostic_kind="command_error",
                error_code=error_code,
            )
            return
        case {"msg_type": "world_state", "world_data": bytes(wd)}:
            _parse_world_state_blob(ws, wd)
            return
        case {"msg_type": 0x4F, "containers": list(containers), "mines": list(mines)}:
            if not containers and not mines:
                ws.mark_pending_radar_empty_delta()
            else:
                update_world_state_from_radar(ws, containers, mines)
                render_ascii_if_available(ws, "Radar")
            return


__all__ = [
    "dispatch_world_state_update",
]
