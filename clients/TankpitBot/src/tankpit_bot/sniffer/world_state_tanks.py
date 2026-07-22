"""Tank state updates for world state.

Every tank-mutating helper here builds a strongly-typed
:class:`TankObservation` and routes it through
:func:`apply_tank_observation`. The mutator enforces the three-timestamp
freshness model (see :class:`TankStateDict`); these helpers exist only
to translate wire-message fields into observations and to thread the
``WorldService`` boundary.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import browser
from tankpit_bot.facts.provenance import make_provenance
from tankpit_bot.facts.source import FactSource
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import remove_tank
from tankpit_bot.state.mutations import apply_tank_observation
from tankpit_bot.state.types import WorldStateDict, make_self_state, make_tank_observation

log = get_logger(__name__)


def update_world_state_from_tank_entry(
    ws: WorldService,
    tank_id: int,
    team: int,
    rank: int,
    x: int,
    y: int,
) -> None:
    """Add or update tank from TankEntry (0x28).

    JS-verified 2026-06-19 against ``Uf.h``: this message carries team,
    rank, position, rank_category, and score. Name comes from TankInfo
    (0x21) separately.

    Args:
        ws: World service instance.
        tank_id: Tank identifier.
        team: Team id (0-3).
        rank: Military rank.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
    """
    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        fact_source="wire_0x28_tank_entry",
        position=(x, y),
        team=team,
        rank=rank,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


def update_world_state_from_tank_info(
    ws: WorldService,
    tank_id: int,
    team: int,
    name: str,
) -> None:
    """Store/update tank from TankInfo (0x21).

    Wire message: carries team and name. Position is preserved from the
    existing registry entry (TankInfo does not refresh position).

    Args:
        ws: World service instance.
        tank_id: Tank identifier.
        team: Team id (0-3).
        name: Player name.
    """
    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        fact_source="wire_0x21_tank_info",
        team=team,
        name=name,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


def update_world_state_from_tank_status(
    ws: WorldService,
    tank_id: int,
    team: int,
    rank: int,
    name: str,
) -> None:
    """Store/update tank from TankStatus (0x3E).

    Wire message: carries team, rank, and name. No position.

    Args:
        ws: World service instance.
        tank_id: Tank identifier.
        team: Team id (0-3).
        rank: Military rank.
        name: Player name.
    """
    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        fact_source="wire_0x3E_tank_status",
        team=team,
        rank=rank,
        name=name,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


def update_world_state_from_move_response_full(
    ws: WorldService,
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
) -> None:
    """Update self_state and tank position from MovementResponse (0x3D).

    For the bot's own tank id this both promotes the tank into
    ``self_state`` (creating a minimal record on first sight) and
    updates the registry entry. For non-self tank ids it only updates
    the registry entry.

    Args:
        ws: World service instance.
        tank_id: Tank identifier.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Team id (0-3).
        rank: Military rank.
    """
    ts = browser.get_current_time_ms()

    self_state = ws.world_state["self_state"]
    if self_state is None or self_state["tank_id"] == 0:
        ws.world_state = WorldStateDict(
            self_state=make_self_state(
                tank_id=tank_id,
                x=x,
                y=y,
                team=team,
                rank=rank,
                fuel=self_state["fuel"] if self_state else 0,
                leaderboard_position=0,
                observed_ms=ts,
                provenance=make_provenance("wire_0x3D_movement", []),
            ),
            tanks=ws.world_state["tanks"],
            containers=ws.world_state["containers"],
            mines=ws.world_state["mines"],
            terrain=ws.world_state["terrain"],
            viewport=ws.world_state["viewport"],
            scanned_tiles=ws.world_state["scanned_tiles"],
            timestamp_ms=ts,
        )
    elif self_state["tank_id"] == tank_id:
        ws.update_world_state_from_position(x, y)

    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        fact_source="wire_0x3D_movement",
        position=(x, y),
        team=team,
        rank=rank,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


def update_world_state_from_client_registry(
    ws: WorldService,
    tank_id: int,
    name: str,
    team: int,
    x: int,
    y: int,
) -> bool:
    """Refine a WIRE-KNOWN tank's position from the client-side registry.

    Client registry refinement is NOT a wire observation -- it is a
    DOM-scraped value used to nudge position when the wire has gone
    quiet. The observation therefore sets ``is_wire_sourced=False`` so
    neither ``last_wire_seen_ms`` nor ``last_position_update_ms``
    advance; only the position value is updated. Tanks that the wire
    has never confirmed are not refined (this method returns False).

    Args:
        ws: World service instance.
        tank_id: Tank id (shared with the wire ID space).
        name: Tank name from the registry entry.
        team: Team id from the verified ``h`` field.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.

    Returns:
        True when a wire-known tank was refined; False for unknown ids.
    """
    existing = ws.world_state["tanks"].get(str(tank_id))
    if existing is None:
        return False
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=existing["timestamp_ms"],
        is_wire_sourced=False,
        storage_source="viewport",
        fact_source="dom_registry_scrape",
        position=(x, y),
        team=team,
        name=name,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)
    return True


def update_world_state_from_tank_damage(
    ws: WorldService,
    tank_id: int,
    damage_state: int,
    *,
    refresh_wire_timestamp: bool = True,
) -> None:
    """Update tank damage from TankStatusSync (0x2E) or registry truth.

    Damage-only wire messages refresh ``last_wire_seen_ms`` (presence
    proof) but MUST NOT refresh ``last_position_update_ms`` -- the
    historical conflation of those two timestamps is the bug this
    refactor was built to prevent. The mutator enforces that invariant.

    Args:
        ws: World service instance.
        tank_id: Tank whose damage tier is being synced.
        damage_state: New damage tier (0-3).
        refresh_wire_timestamp: True for wire-sourced updates; False
            when the caller is a non-wire diagnostic (e.g.
            registry-truth recomputation) and must preserve the
            wire-seen timestamp.
    """
    previous = ws.world_state["tanks"].get(str(tank_id))
    if previous is None:
        return
    ts = browser.get_current_time_ms() if refresh_wire_timestamp else previous["timestamp_ms"]
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=refresh_wire_timestamp,
        storage_source=previous["source"],
        fact_source=(
            "wire_0x2E_tank_status_sync" if refresh_wire_timestamp else "dom_registry_scrape"
        ),
        damage_state=damage_state,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)
    if previous["damage_state"] != damage_state:
        emit_diagnostic(
            diagnostic_kind="tank_damage_changed",
            tank_id=tank_id,
            tank_name=previous["name"],
            previous_damage_state=previous["damage_state"],
            damage_state=damage_state,
        )


def update_world_state_from_tank_remove(ws: WorldService, tank_id: int) -> None:
    """Handle TankRemove (0x58) or container TankLeave — which does NOT delete.

    MISLEADING NAME, DELIBERATE NO-OP DOWNSTREAM: :func:`remove_tank`
    intentionally keeps the registry entry (0x58 fires on tracking
    churn as well as death — five removes across two actual kills in
    the 2026-06-20 ghost capture; 0x41 Deactivation is the only
    authoritative death signal). Keeping the frozen entry is what
    powers the pursuit volley: HUNT keeps firing homing at the last
    wire position and the server reroutes those shots to the departed
    tank for the ~12 s TTL ([[shoot-event-format]]). This docstring
    previously said "removal is a deletion" and misled a 2026-07-21
    analysis into believing the volley was blocked — read
    :func:`remove_tank` itself before trusting any summary here.

    Args:
        ws: World service instance.
        tank_id: Departing tank id.
    """
    # The 0x58 is the start of the server's ~12 s id-routing grace: a
    # departed tank stays resolvable for shoot-at-id for roughly 11-13
    # more seconds (run 2026-07-19 22:30:00 -- last rerouted homing
    # hit at +11.0 s after the remove, first miss fired at +13.0 s).
    # The diagnostic timestamps every removal so pursuit misses can be
    # correlated against it and the TTL constant narrowed across runs
    # (see wiki [[shoot-event-format]] section "Global action queue").
    emit_diagnostic(diagnostic_kind="tank_removed", tank_id=tank_id)
    ws.world_state = remove_tank(ws.world_state, tank_id, browser.get_current_time_ms())


def _update_tank_position(
    ws: WorldService,
    tank_id: int,
    x: int,
    y: int,
    fact_source: FactSource,
) -> None:
    """Update a tank's position from a position-bearing wire message.

    Used by the 0x42 BuildPickup source position, the 0x47 Movement
    waypoint destination resolution, and the 0x53 ShootEvent
    enemy-source position update. The observation declares
    wire-sourced + position so the position-freshness timestamp
    advances.

    Args:
        ws: World service instance.
        tank_id: Tank identifier.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        fact_source: Exact wire channel the position arrived on.
    """
    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        fact_source=fact_source,
        position=(x, y),
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


def _update_enemy_from_detection(
    ws: WorldService,
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
) -> None:
    """Update enemy tank position from EnemyDetection (0x48).

    EnemyDetect is the server's response to a radar action. Position
    semantics deliberately route through the non-wire observation path
    so neither ``last_wire_seen_ms`` nor ``last_position_update_ms``
    advance -- a radar-derived position is a map-style estimate, not
    structural wire-presence proof. The kill-shot gate must continue
    to require fresh wire-bearing position; radar alone does not
    suffice.

    Args:
        ws: World service instance.
        tank_id: Enemy tank id.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Enemy team id.
        rank: Enemy military rank.
    """
    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=False,
        storage_source="world_state",
        fact_source="wire_0x48_enemy_detect",
        position=(x, y),
        team=team,
        rank=rank,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)
    existing = ws.world_state["tanks"].get(str(tank_id))
    name = existing["name"] if existing is not None else ""
    log.info(
        "ENEMY_DETECT: tank=%d at (%d,%d) team=%d rank=%d name=%s",
        tank_id,
        x,
        y,
        team,
        rank,
        name,
    )
