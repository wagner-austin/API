"""Protocol message dispatch for world state updates.

Routes decoded protocol messages to the appropriate world-state mutation
functions. This module is the only consumer of the ``_dispatch_*`` family;
the public entry point is ``dispatch_world_state_update``.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import browser, protocol
from tankpit_bot.container.types import ContainerPickupRecordDict
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
)
from tankpit_bot.sniffer.world_state_dispatch_position import (
    _dispatch_position_update,
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
    update_world_state_from_tank_info,
    update_world_state_from_tank_remove,
    update_world_state_from_tank_status,
)
from tankpit_bot.sniffer.world_state_tiles import (
    render_ascii_if_available,
)
from tankpit_bot.state import (
    add_mine,
    deactivate_tank,
    remove_mine,
    replace_map_fuel_dots,
    set_self_rank,
)

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


def _dispatch_map_data(
    ws: WorldService, fuel_dots: list[tuple[int, int]], tanks: list[protocol.MapTankEntry]
) -> None:
    """Apply a 0x4C MapData snapshot to world state.

    Two effects, both wholesale: the fuel-dot atlas is replaced and
    every tank slot is lifted into world state via the observation
    pipeline. **Map snapshots are not wire-sourced** -- they're cached
    server state that can keep re-listing a tank at a stale position
    after the tank has actually left. The observations therefore
    declare ``is_wire_sourced=False`` so the wire-presence freshness
    counter (``last_wire_seen_ms``) does not advance, even as
    ``timestamp_ms`` does. The ghost-wire-presence regression in
    ``tests/replay/test_ghost_wire_presence_regression.py`` proves
    why: a tank that's wire-silent but map-fresh must still be
    acquisition-fresh, while the kill gate must reject it.

    Args:
        ws: World service instance.
        fuel_dots: Decoded ``(x, y)`` fuel dot positions.
        tanks: Decoded :class:`protocol.MapTankEntry` slots, one per
            tank visible on the map.
    """
    from tankpit_bot.state.types import make_tank_observation

    ts = browser.get_current_time_ms()
    ws.world_state = replace_map_fuel_dots(ws.world_state, fuel_dots, ts)
    for entry in tanks:
        obs = make_tank_observation(
            tank_id=entry["tank_id"],
            timestamp_ms=ts,
            is_wire_sourced=False,
            storage_source="world_state",
            position=(entry["x"], entry["y"]),
            team=entry["team"],
            rank=entry["rank"],
            damage_state=entry["damage"],
        )
        from tankpit_bot.state.mutations import apply_tank_observation

        ws.world_state = apply_tank_observation(ws.world_state, obs)
    ws.mark_map_data_processed()
    emit_diagnostic(
        diagnostic_kind="map_data_snapshot",
        fuel_dot_count=len(fuel_dots),
        tank_count=len(tanks),
    )


def _dispatch_self_promotion(ws: WorldService, new_rank: int, was_promoted: bool) -> None:
    """Apply a 0x2B Promotion (Rf) to self_state and emit a diagnostic.

    JS Rf.prototype.h: ``a.i.l = this.j`` -- the server-authoritative
    rank assignment to the player's own tank. ``was_promoted`` is the
    UI banner flag; ``new_rank`` is the absolute new rank index.

    Args:
        ws: World service instance.
        new_rank: New rank index (0-8).
        was_promoted: True when the server intends a "promoted" banner;
            False on silent rank resets (e.g. join-time initialization).
    """
    ws.world_state = set_self_rank(ws.world_state, new_rank, browser.get_current_time_ms())
    emit_diagnostic(
        diagnostic_kind="self_promotion",
        new_rank=new_rank,
        was_promoted=was_promoted,
    )


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


def _dispatch_tank_state(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch tank join / info / status / damage / removal."""
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
            update_world_state_from_tank_remove(ws, tid)
            return True
    return False


def _dispatch_tank_announcements(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch announcement-style messages with no positional effect.

    Covers 0x29 TankExit, 0x2B Promotion, 0x4E Decoration, 0x56
    Statistics. The 0x42 BuildPickup is handled here too because it
    behaves like an event observation -- it does mutate the actor's
    position via :func:`_update_tank_position` but contributes no
    structural world-state change beyond that.
    """
    match decoded:
        case {
            "msg_type": 0x29,
            "team": int(team),
            "tank_id": int(tid),
            "was_silent": bool(was_silent),
            "was_eliminated": bool(was_eliminated),
        }:
            emit_diagnostic(
                diagnostic_kind="tank_exit_announcement",
                team=team,
                tank_id=tid,
                was_silent=was_silent,
                was_eliminated=was_eliminated,
            )
            return True
        case {
            "msg_type": 0x2B,
            "new_rank": int(new_rank),
            "was_promoted": bool(was_promoted),
        }:
            _dispatch_self_promotion(ws, new_rank, was_promoted)
            return True
        case {
            "msg_type": 0x4E,
            "tank_id": int(tid),
            "slot": int(slot),
            "level": int(level),
        }:
            emit_diagnostic(
                diagnostic_kind="tank_decoration",
                tank_id=tid,
                slot=slot,
                level=level,
            )
            return True
        case {
            "msg_type": 0x42,
            "tank_id": int(tid),
            "source_x": int(sx),
            "source_y": int(sy),
            "drop_x": int(dx),
            "drop_y": int(dy),
            "obstacle_type": int(obstacle_type),
        }:
            _update_tank_position(ws, tid, sx, sy)
            emit_diagnostic(
                diagnostic_kind="build_pickup",
                tank_id=tid,
                source_x=sx,
                source_y=sy,
                drop_x=dx,
                drop_y=dy,
                obstacle_type=obstacle_type,
            )
            return True
        case {
            "msg_type": 0x56,
            "playtime_hours": int(hours),
            "playtime_minutes": int(minutes),
            "playtime_seconds": int(seconds),
            "destroyed": int(destroyed),
            "deactivated": int(deactivated),
            "score": int(score),
        }:
            emit_diagnostic(
                diagnostic_kind="self_statistics",
                playtime_hours=hours,
                playtime_minutes=minutes,
                playtime_seconds=seconds,
                destroyed=destroyed,
                deactivated=deactivated,
                score=score,
            )
            return True
        case {"msg_type": 0x3C, "message": str(message)}:
            # ``message`` is reserved by the runtime logger as the
            # human-readable channel line; use ``text`` for the payload.
            emit_diagnostic(diagnostic_kind="supervisor_text", text=message)
            return True
    return False


def _dispatch_tank_lifecycle(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch every tank lifecycle / announcement / stats message.

    Handled here: 0x28 TankEntry, 0x21 TankInfo, 0x3E TankStatusFull,
    0x2E TankStatusSync (damage), 0x58 TankRemove, 0x29 TankExit
    announcement, 0x2B Promotion, 0x4E Decoration, 0x42 BuildPickup,
    0x56 Statistics. None of these resolve combat geometry; the actual
    state-bearing ones (TankEntry/Info/Status/Damage/Remove) are
    factored into :func:`_dispatch_tank_state` and the rest into
    :func:`_dispatch_tank_announcements`.

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    if _dispatch_tank_state(ws, decoded):
        return True
    return _dispatch_tank_announcements(ws, decoded)


def _dispatch_tank_update(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch tank-related messages to update world state.

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    if _dispatch_tank_lifecycle(ws, decoded):
        return True
    match decoded:
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
            # 0x41 starts the corpse window. Empirical capture
            # 2026-06-20: 0x58 TankRemove arrives ~22 s later; in
            # between, the tile renders a corpse but the bot must not
            # re-target it. The liveness="deactivated" gate filters the
            # tank from analyze_threats; the position is preserved as
            # the death tile so the bot can still reason about the
            # geometry (mine-on-corpse, fuel-deposit-on-corpse, etc.).
            mark_tank_killed(ws, vid)
            ws.world_state = deactivate_tank(ws.world_state, vid, browser.get_current_time_ms())
            emit_diagnostic(
                diagnostic_kind="tank_deactivated",
                origin="protocol_0x41",
                victim_id=vid,
                killer_id=kid,
            )
            log.info("DEACTIVATED: tank=%d killed by %d", vid, kid)
            return True
    return False


# =============================================================================
# Dispatch — container messages (mines, combat, pickup)
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


#: Window during which a repeated ContainerPickup with identical pickup
#: signature is treated as the server's duplicate broadcast (one to the
#: picker, one to the world view). Empirically the two broadcasts arrive
#: within ~1-200 ms; 500 ms is the comfortable upper bound.
PICKUP_DEDUP_WINDOW_MS: int = 500


def _is_duplicate_pickup_broadcast(
    ws: WorldService,
    pickups: tuple[ContainerPickupRecordDict, ...],
) -> bool:
    """Suppress the second copy of a server-broadcast ContainerPickup.

    Builds a signature from the pickup records (x, y, remaining_volume
    tuples) and checks the per-session recent-pickup ledger. If the same
    signature was already seen within :data:`PICKUP_DEDUP_WINDOW_MS`,
    this is the world-view broadcast that pairs with the picker
    broadcast; return True and the caller skips the world-state update.

    Args:
        ws: World service instance carrying the dedup ledger.
        pickups: Pickup records from the decoded message.

    Returns:
        True when this is a duplicate of a recent broadcast (caller
        should skip), False on the first sighting (caller should apply).
    """
    signature = tuple((record["x"], record["y"], record["remaining_volume"]) for record in pickups)
    now_ms = browser.get_current_time_ms()
    last_seen = ws.recent_pickup_signatures.get(signature)
    if last_seen is not None and now_ms - last_seen <= PICKUP_DEDUP_WINDOW_MS:
        ws.recent_pickup_signatures[signature] = now_ms
        return True
    ws.recent_pickup_signatures[signature] = now_ms
    # Bound the ledger so it doesn't grow without limit during long
    # sessions. Drop entries older than 2 windows.
    cutoff = now_ms - 2 * PICKUP_DEDUP_WINDOW_MS
    ws.recent_pickup_signatures = {
        sig: ts for sig, ts in ws.recent_pickup_signatures.items() if ts >= cutoff
    }
    return False


def _apply_container_pickups(
    ws: WorldService,
    pickups: tuple[ContainerPickupRecordDict, ...],
) -> None:
    """Apply one decoded ContainerPickup body (single- or multi-record).

    Drops duplicate server broadcasts via :func:`_is_duplicate_pickup_broadcast`
    and forwards each unique record to the world-state mutator.

    Args:
        ws: World service instance.
        pickups: Tuple of pickup records from one wire message.
    """
    if _is_duplicate_pickup_broadcast(ws, pickups):
        return
    for record in pickups:
        update_world_state_from_container_pickup(ws, record["x"], record["y"])


def _dispatch_container_message(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch container-level messages (mines, pickup, teleport landed).

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
        case {"msg_type": "container_pickup", "pickups": tuple(pickups)}:
            _apply_container_pickups(ws, pickups)
            return True
        case {"msg_type": "teleport_landed"}:
            emit_world("TELEPORT_LANDED: server confirmed teleport")
            mark_teleport_landed(ws)
            return True
    return False


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
        case {"msg_type": 0x4F, "containers": list(containers), "mines": list(mines)}:
            if not containers and not mines:
                ws.mark_pending_radar_empty_delta()
            else:
                update_world_state_from_radar(ws, containers, mines)
                render_ascii_if_available(ws, "Radar")
            return
        case {"msg_type": 0x4C, "fuel_dots": list(fuel_dots), "tanks": list(map_tanks)}:
            _dispatch_map_data(ws, fuel_dots, map_tanks)
            return


__all__ = [
    "dispatch_world_state_update",
]
