"""World-state dispatch: the entry point and the tank channels.

``dispatch_world_state_update`` routes one decoded message into world
state. Tank state and lifecycle live here; combat, session broadcasts,
and containers are the three sibling modules this dispatches into.

This module was 1,283 lines. The split is deliberately orthogonal to
the session-state work: no ``get_world_service()`` call site moved, and
the dispatch chain still receives its ``WorldService`` instance as an
argument, so threading the instance stays a one-signature change.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import browser, protocol
from tankpit_bot.protocol.constants import SUPERVISOR_ERROR_NAMES
from tankpit_bot.runtime_logging import (
    emit_diagnostic,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_combat import (
    mark_tank_killed,
)
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_fuel_total,
)
from tankpit_bot.sniffer.world_state_dispatch_combat import (
    _dispatch_shoot_event,
    _record_shot_fuel_entry,
)
from tankpit_bot.sniffer.world_state_dispatch_containers import _dispatch_container_message
from tankpit_bot.sniffer.world_state_dispatch_position import (
    _dispatch_position_update,
)
from tankpit_bot.sniffer.world_state_dispatch_session import _dispatch_tank_announcements
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
    deactivate_tank,
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
    from tankpit_bot.state.tank_mutations import apply_tank_observation
    from tankpit_bot.state.types import make_tank_observation

    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        fact_source="wire_0x3D_movement",
        position=(x, y),
        team=team,
        rank=rank,
        damage_state=damage_state,
        direction=direction,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


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
        case {
            "msg_type": 0x21,
            "tank_id": int(tid),
            "team": int(team),
            "name": str(name),
            "persistent_tank_id": int(persistent_id),
            "decoration_state": bytes(decoration),
        }:
            update_world_state_from_tank_info(ws, tid, team, name)
            # Persistent identity + decoration are the cross-session
            # opponent-tracking signal: persistent_tank_id stays
            # constant across respawns and sessions (game-engine fact,
            # mined from JS Tf.h ``a.aa``); decoration_state is the
            # tank's cosmetic skin bytes, useful for visual ID. Emit
            # as a diagnostic so the bot's session log carries the
            # mapping name <-> persistent_id and downstream analyzers
            # ("did we fight this player last match?") can join on it.
            emit_diagnostic(
                diagnostic_kind="tank_identity",
                tank_id=tid,
                team=team,
                name=name,
                persistent_tank_id=persistent_id,
                decoration_state_hex=decoration.hex(),
            )
            self_state = ws.world_state["self_state"]
            if self_state is not None and self_state["tank_id"] == tid:
                ws.record_self_identity(
                    name,
                    persistent_id,
                    decoration.hex(),
                    browser.get_current_time_ms(),
                )
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
        case {
            "msg_type": 0x2E,
            "tank_id": int(tid),
            "damage_state": int(dmg),
            "promo_state": int(promo),
            "rank": int(rank),
        }:
            update_world_state_from_tank_damage(ws, tid, dmg)
            # promo_state is a live promotion-PROGRESS counter (JS
            # Og.h ``g`` field): it climbs with damage dealt and
            # RESETS to 0 at the promoting kill, at which tick the
            # rank field itself flips (measured bot-20260725-211120:
            # promo_state 0->3->5->6->0 and rank 0->1 at t+31.7s, the
            # first kill). The 0x2E is the promotion's earliest wire
            # signal — no 0x2B arrived that session — so the self
            # rank must be applied HERE for the rank-derived
            # readiness bars and capacities to follow a mid-session
            # promotion.
            self_state = ws.world_state["self_state"]
            if self_state is not None and self_state["tank_id"] == tid:
                ws.update_world_state_from_rank(rank, "wire_0x2E_tank_status_sync")
                if promo > 0:
                    emit_diagnostic(
                        diagnostic_kind="self_promo_eligible",
                        tank_id=tid,
                        promo_state=promo,
                    )
            return True
        case {"msg_type": 0x2E, "tank_id": int(tid), "damage_state": int(dmg)}:
            update_world_state_from_tank_damage(ws, tid, dmg)
            return True
        case {"msg_type": 0x58, "tank_id": int(tid)}:
            update_world_state_from_tank_remove(ws, tid)
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
            "aim_x": int(aim_x),
            "aim_y": int(aim_y),
            "weapon": int(weapon),
        }:
            _dispatch_shoot_event(ws, shooter_id, sx, sy, tx, ty, aim_x, aim_y, weapon)
            _record_shot_fuel_entry(ws, shooter_id, weapon)
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
            self_state = ws.world_state["self_state"]
            if self_state is not None and vid == self_state["tank_id"]:
                # Our own 0x41 (fires for own kills too — falsified
                # decoder blind spot fixed 2026-07-19). Record the
                # fact; the tick loop owns the session-exit decision
                # (dispatch also runs under replay/analysis, which
                # must not throw mid-stream).
                ws.self_deactivated = True
                emit_diagnostic(
                    diagnostic_kind="self_deactivated",
                    origin="protocol_0x41",
                    killer_id=kid,
                )
                log.info("SELF DEACTIVATED: killed by %d", kid)
                return True
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


def _dispatch_map_data(
    ws: WorldService,
    tanks: list[protocol.MapTankEntry],
    fuel_dots: list[tuple[int, int]],
) -> None:
    """Apply a 0x4C MapData snapshot to world state.

    Every tank slot is lifted into world state via the observation
    pipeline. **Map snapshots are not wire-sourced** -- they're cached
    server state that can keep re-listing a tank at a stale position
    after the tank has actually left. The observations therefore
    declare ``is_wire_sourced=False`` so the wire-presence freshness
    counter (``last_wire_seen_ms``) does not advance: a wire-silent but
    map-listed tank must NOT masquerade as present.

    Position is a different question: at the instant the server emits
    MAP_DATA, every listed tank's ``(x, y)`` is the server's
    authoritative statement of where that tank IS. So
    ``position_is_authoritative=True`` and the kill-shot
    ``last_position_update_ms`` gate advances. The wire-presence gate
    still filters departed-tank afterimages; this just stops a
    wire-quiet stationary target from being treated as
    position-stale during a fight (live run 20260620-191622: 22
    map_opens / 19 teleports / 0 kills because the kill-shot gate
    blocked targets the bot was actively engaging).

    Args:
        ws: World service instance.
        tanks: Decoded :class:`protocol.MapTankEntry` slots, one per
            tank visible on the map.
        fuel_dots: Decoded skip-RLE fuel-dot atlas positions. The
            atlas is server-cached per session, so this simply
            overwrites the stored copy on every map open.
    """
    from tankpit_bot.state.types import make_tank_observation

    ts = browser.get_current_time_ms()
    for entry in tanks:
        obs = make_tank_observation(
            tank_id=entry["tank_id"],
            timestamp_ms=ts,
            is_wire_sourced=False,
            position_is_authoritative=True,
            storage_source="world_state",
            fact_source="wire_0x4C_map_data",
            position=(entry["x"], entry["y"]),
            team=entry["team"],
            rank=entry["rank"],
            damage_state=entry["damage"],
        )
        from tankpit_bot.state.tank_mutations import apply_tank_observation

        ws.world_state = apply_tank_observation(ws.world_state, obs)
    ws.map_fuel_dots = tuple(fuel_dots)
    ws.mark_map_data_processed()
    emit_diagnostic(
        diagnostic_kind="map_data_snapshot",
        tank_count=len(tanks),
        fuel_dot_count=len(fuel_dots),
    )


def _dispatch_resource_update(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch resource and inventory messages.

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {"msg_type": 0x2E, "fuel": int(fuel), "rank": int(rank)} if fuel is not None:
            # The long (fuel-bearing) form is per-recipient — it is
            # ALWAYS the self tank, and it is the form the live
            # promotion arrived on (bot-20260725-211120: rank 0 -> 1
            # in the fuel-bearing 0x2E at the kill tick). Apply the
            # rank alongside the fuel so a mid-session promotion
            # reaches the rank-derived bars the tick it lands.
            update_world_state_from_fuel_total(ws, fuel, "wire_0x2E_tank_status_sync")
            ws.update_world_state_from_rank(rank, "wire_0x2E_tank_status_sync")
            return True
        case {
            "msg_type": 0x44,
            "fuel_total": int(fuel_total),
            "is_free": bool(is_free),
        }:
            update_world_state_from_fuel_total(ws, fuel_total, "wire_0x44_fuel_gain")
            emit_diagnostic(
                diagnostic_kind="fuel_gain",
                fuel_total=fuel_total,
                is_free=is_free,
            )
            return True
        case {"msg_type": 0x64, "fuel_total": int(fuel_total)}:
            update_world_state_from_fuel_total(ws, fuel_total, "wire_0x64_fuel_total")
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


def dispatch_world_state_update(ws: WorldService, decoded: protocol.BinaryMessage) -> None:
    """Dispatch decoded message to update world state, inventory, and render ASCII.

    Delegates to specialized dispatchers for resources, tanks, positions,
    and container messages.

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.
    """
    ws.last_game_message_ms = browser.get_current_time_ms()
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
            # The raw wire record for EVERY 0x52 — including the ones
            # that mean success (code 5 is the clamp receipt riding
            # with a completed fuel transfer; code 4 the stale-belief
            # purge). error_name keeps scorecard readers from
            # mistaking receipts for failures; the OUTCOME layer
            # (emit_collect_clamped_transfer etc.) carries the true
            # classification.
            emit_diagnostic(
                diagnostic_kind="command_error",
                error_code=error_code,
                error_name=SUPERVISOR_ERROR_NAMES.get(error_code, "unknown"),
            )
            return
        case {
            "msg_type": 0x4F,
            "containers": list(containers),
            "mines": list(mines),
            "mine_clears": list(mine_clears),
        }:
            if not containers and not mines and not mine_clears:
                ws.mark_pending_radar_empty_delta()
            else:
                update_world_state_from_radar(ws, containers, mines, mine_clears)
                render_ascii_if_available(ws, "Radar")
            return
        case {"msg_type": 0x4C, "fuel_dots": list(fuel_dots), "tanks": list(map_tanks)}:
            _dispatch_map_data(ws, map_tanks, fuel_dots)
            return


__all__ = [
    "dispatch_world_state_update",
    "log",
]
