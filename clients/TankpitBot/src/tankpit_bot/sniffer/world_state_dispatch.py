"""Protocol message dispatch for world state updates.

Routes decoded protocol messages to the appropriate world-state mutation
functions. This module is the only consumer of the ``_dispatch_*`` family;
the public entry point is ``dispatch_world_state_update``.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import browser, protocol
from tankpit_bot.container.types import ContainerPickupRecordDict
from tankpit_bot.ledger.ammo_book import record_ammo_enemy_shot, record_ammo_shot
from tankpit_bot.ledger.fuel_book import FuelEntryKind, record_fuel_entry
from tankpit_bot.physics.costs import (
    DUAL_SHOT_COST,
    HOMING_SHOT_COST,
    MISSILE_SHOT_COST,
    SINGLE_SHOT_COST,
)
from tankpit_bot.physics.damage import DUAL_HIT_VICTIM_COST, MINE_DETONATION_COST
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
        fact_source="wire_0x3D_movement",
        position=(x, y),
        team=team,
        rank=rank,
        damage_state=damage_state,
        direction=direction,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


_SHOT_ENTRY_KINDS: dict[int, FuelEntryKind] = {
    0: "shot_single",
    1: "shot_dual",
    2: "shot_missile",
    3: "shot_homing",
}
_SHOT_ENTRY_COSTS: dict[int, int] = {
    0: SINGLE_SHOT_COST,
    1: DUAL_SHOT_COST,
    2: MISSILE_SHOT_COST,
    3: HOMING_SHOT_COST,
}


def _record_shot_fuel_entry(ws: WorldService, shooter_id: int, weapon: int) -> None:
    """Record a 0x53 echo's fuel effect into the live fuel book.

    Own shots debit their physics cost exactly (homing may split its
    debit across the sync boundary, so its ceiling is -5 and the book
    seeds a carry); enemy shots are optional debits bounded by the
    worst known victim cost — the shot may have targeted someone else
    — and count toward the ammo book's armor feasibility bound
    (shields may only fall for observed incoming fire; unwired until
    the 2026-07-22 fighting soak caught ``enemy_shots`` frozen at 0,
    which would have raised a FALSE ammo divergence on the first
    armor-absorbed hit).

    Args:
        ws: World service instance.
        shooter_id: Who fired the shot.
        weapon: Weapon byte (0=single, 1=dual, 2=missile, 3=homing).
    """
    if weapon not in _SHOT_ENTRY_KINDS:
        return
    self_state = ws.world_state["self_state"]
    if self_state is not None and shooter_id == self_state["tank_id"]:
        cost = _SHOT_ENTRY_COSTS[weapon]
        hi = -(cost // 2) if weapon == 3 else -cost
        record_fuel_entry(book=ws.fuel_book, kind=_SHOT_ENTRY_KINDS[weapon], lo=-cost, hi=hi)
        record_ammo_shot(book=ws.ammo_book, weapon=weapon)
    else:
        record_fuel_entry(book=ws.fuel_book, kind="enemy_hit", lo=-DUAL_HIT_VICTIM_COST, hi=0)
        record_ammo_enemy_shot(book=ws.ammo_book)


def _dispatch_shoot_event(
    ws: WorldService,
    shooter_id: int,
    sx: int,
    sy: int,
    tx: int,
    ty: int,
    aim_x: int,
    aim_y: int,
    weapon: int,
) -> None:
    """Apply a 0x53 ShootEvent to world state.

    The wire carries TWO target-ish coordinates: ``(tx, ty)`` is the
    tile the shot ultimately resolves against, while ``(aim_x, aim_y)``
    is the shooter's barrel aim at the moment of fire. For straight
    shots (weapon=0 single, weapon=1 dual) the two coincide; for
    homing / missile fire they can diverge as the projectile redirects
    mid-flight. The split is empirically promoted from ``unk1`` /
    ``unk2`` in task #73 against production captures.

    Effects on world state:

    * Own shot -> tile-occupancy hit detection: lookup tank at the
      target tile (``tx, ty``), record victim id. The aim coords are
      logged for observability so live runs surface barrel-vs-impact
      drift on homing fire.
    * Enemy shot -> their source tile (``sx, sy``) is a fresh
      wire-sourced position update for the shooter; ``(aim_x, aim_y)``
      is recorded on the enemy tank as ``last_aim_x`` /
      ``last_aim_y`` so the combat AI can reason about their barrel
      direction next tick.

    Args:
        ws: World service instance.
        shooter_id: Who fired the shot.
        sx: Shooter source tile X.
        sy: Shooter source tile Y.
        tx: Shot target tile X (resolved impact tile).
        ty: Shot target tile Y (resolved impact tile).
        aim_x: Shooter's barrel-aim X at the moment of fire.
        aim_y: Shooter's barrel-aim Y at the moment of fire.
        weapon: Weapon byte (0=single, 1=dual, 2=missile, 3=homing).
    """
    self_state = ws.world_state["self_state"]
    own_tank_id = self_state["tank_id"] if self_state is not None else -1
    aim_drift = (aim_x, aim_y) != (tx, ty)
    if shooter_id == own_tank_id:
        # OUR_SHOT's (tx, ty) is the server's homing-tracked landing
        # tile, which is often off-viewport once the target teleports
        # away. Overwriting the registry with it poisoned the
        # user-contract stay-put loop: the planner's next shoot would
        # dispatch at the off-viewport coord, and the server rejects
        # shoot commands targeted outside the 18x18 viewport (see
        # [[shot-range]]). Pre-098d3d7 the registry was not refreshed
        # from own shots; the bot kept aiming at the last on-viewport
        # tile and the server auto-tracked with homing on every shot
        # -- unlimited homings until the kill. Restored 2026-06-26.
        victim_id = _find_tank_at_tile(ws, tx, ty, exclude_id=own_tank_id)
        log.info(
            "OUR_SHOT: weapon=%d src=(%d,%d) tgt=(%d,%d) aim=(%d,%d)%s victim_id=%d",
            weapon,
            sx,
            sy,
            tx,
            ty,
            aim_x,
            aim_y,
            " [drift]" if aim_drift else "",
            victim_id,
        )
        mark_combat_hit(ws, weapon, victim_id)
    elif shooter_id > 0:
        _update_tank_position(ws, shooter_id, sx, sy, "wire_0x53_shoot_event")
        _record_enemy_aim(ws, shooter_id, aim_x, aim_y, weapon)
        if aim_drift:
            log.info(
                "ENEMY_SHOT: tid=%d weapon=%d src=(%d,%d) tgt=(%d,%d) aim=(%d,%d) [drift]",
                shooter_id,
                weapon,
                sx,
                sy,
                tx,
                ty,
                aim_x,
                aim_y,
            )


def _record_enemy_aim(
    ws: WorldService,
    shooter_id: int,
    aim_x: int,
    aim_y: int,
    weapon: int,
) -> None:
    """Persist the enemy's last barrel-aim coordinates on the tank state.

    Threats consumers (combat AI, recover-fuel route planner) read
    ``last_aim_*`` on the tank state to reason about which tiles the
    enemy may fire on next tick. The fields are wire-fresh on every
    0x53 ShootEvent so they decay naturally with the tank's
    ``last_wire_seen_ms``.

    Args:
        ws: World service instance.
        shooter_id: Enemy tank id that fired.
        aim_x: Wire-reported barrel-aim X at the moment of fire.
        aim_y: Wire-reported barrel-aim Y at the moment of fire.
        weapon: Weapon byte (used downstream to discriminate which
            aim-target tile applies; logged here for traceability).
    """
    from tankpit_bot.state.mutations import set_tank_last_aim

    ws.world_state = set_tank_last_aim(
        ws.world_state,
        shooter_id,
        aim_x,
        aim_y,
        weapon,
        browser.get_current_time_ms(),
    )


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
        from tankpit_bot.state.mutations import apply_tank_observation

        ws.world_state = apply_tank_observation(ws.world_state, obs)
    ws.map_fuel_dots = tuple(fuel_dots)
    ws.mark_map_data_processed()
    emit_diagnostic(
        diagnostic_kind="map_data_snapshot",
        tank_count=len(tanks),
        fuel_dot_count=len(fuel_dots),
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
            update_world_state_from_fuel_total(ws, fuel, "wire_0x2E_tank_status_sync")
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
        }:
            update_world_state_from_tank_damage(ws, tid, dmg)
            # promo_state is the per-tank promotion-eligibility byte
            # (0 = no pending promotion, > 0 indicates eligibility per
            # JS Og.h ``g`` field). For OWN tank, this lets the AI
            # know "I'm about to rank up if I get one more kill", which
            # could influence aggression. For enemy tanks, it's a
            # marker that a player is on a hot streak. The promo banner
            # itself fires separately as 0x2B Promotion; this is the
            # passive eligibility signal that precedes it.
            self_state = ws.world_state["self_state"]
            if self_state is not None and self_state["tank_id"] == tid and promo > 0:
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
            _update_tank_position(ws, tid, sx, sy, "wire_0x42_build_pickup")
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
            playtime_total = hours * 3600 + minutes * 60 + seconds
            ws.career_destroyed = destroyed
            ws.career_deactivated = deactivated
            ws.career_score = score
            ws.career_playtime_seconds_total = playtime_total
            ws.career_stats_last_update_ms = browser.get_current_time_ms()
            emit_diagnostic(
                diagnostic_kind="self_statistics",
                playtime_hours=hours,
                playtime_minutes=minutes,
                playtime_seconds=seconds,
                playtime_seconds_total=playtime_total,
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
    return _dispatch_session_broadcasts(ws, decoded)


def _emit_active_players(
    ws: WorldService,
    players: list[protocol.ActivePlayerEntry],
) -> None:
    """Persist an 0x2F ActivePlayers roster and emit a structured diagnostic.

    Args:
        ws: World service instance.
        players: Decoded roster entries in server-sent order.
    """
    ws.active_players = [(player["tank_id"], player["rank"]) for player in players]
    emit_diagnostic(
        diagnostic_kind="active_players",
        count=len(players),
        tank_ids=",".join(str(player["tank_id"]) for player in players),
    )


def _emit_top10(
    ws: WorldService,
    team_filter: int,
    viewer_score: int,
    viewer_position: int,
    entries: list[protocol.Top10EntryDict],
) -> None:
    """Persist a 0x31 Top10 snapshot on the world service + emit a diagnostic.

    The Top10 broadcast can come with zero rows (very fresh sessions
    or empty leaderboards); guard the ``entries[0]`` peek so we still
    emit a structured event with row_count=0.

    Args:
        ws: World service instance.
        team_filter: Wire's team_filter byte (255 = all teams).
        viewer_score: 24-bit BE score for the viewing player.
        viewer_position: 1-based leaderboard rank for the viewer.
        entries: Decoded Top10 rows in server-sent order.
    """
    ws.top10_viewer_score = viewer_score
    ws.top10_viewer_position = viewer_position
    ws.top10_team_filter = team_filter
    top_name: str = entries[0]["name"] if entries else ""
    top_score: int = entries[0]["score"] if entries else 0
    emit_diagnostic(
        diagnostic_kind="top10",
        team_filter=team_filter,
        viewer_score=viewer_score,
        viewer_position=viewer_position,
        row_count=len(entries),
        top_name=str(top_name),
        top_score=int(top_score),
    )


def _dispatch_session_broadcasts(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch session-level server broadcasts.

    Covers 0x2F ActivePlayers, 0x31 Top10, 0x60 PingResponse, and 0x7E
    ConnectionLost -- all of which carry no tank-state geometry but
    DO carry session information the bot's events stream should
    capture. Split out of :func:`_dispatch_tank_announcements` to keep
    the latter under the C901 complexity ceiling.

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.

    Returns:
        True when the message matched one of the broadcast shapes,
        False otherwise (so the caller can fall through to other
        dispatchers).
    """
    match decoded:
        case {"msg_type": 0x2F, "players": list(players)}:
            _emit_active_players(ws, players)
            return True
        case {
            "msg_type": 0x31,
            "team_filter": int(team_filter),
            "viewer_score": int(viewer_score),
            "viewer_position": int(viewer_position),
            "entries": list(entries),
        }:
            _emit_top10(ws, team_filter, viewer_score, viewer_position, entries)
            return True
        case {"msg_type": 0x60}:
            ws.last_ping_response_ms = browser.get_current_time_ms()
            emit_diagnostic(diagnostic_kind="ping_response")
            return True
        case {"msg_type": 0x7E}:
            emit_diagnostic(diagnostic_kind="connection_lost")
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
        update_world_state_from_container_pickup(
            ws,
            record["x"],
            record["y"],
            record["remaining_volume"],
        )
        emit_diagnostic(
            diagnostic_kind="container_pickup_dispatched",
            x=record["x"],
            y=record["y"],
            remaining_volume=record["remaining_volume"],
            is_partial=record["remaining_volume"] > 0,
        )


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
            record_fuel_entry(book=ws.fuel_book, kind="detonation", lo=-MINE_DETONATION_COST, hi=0)
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
]
