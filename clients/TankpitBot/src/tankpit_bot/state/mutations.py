"""World state mutation functions.

This module provides pure functions that update WorldStateDict by creating
new state objects (immutable update pattern).
"""

from __future__ import annotations

from tankpit_bot.state.types import (
    DAMAGE_FULL,
    SelfStateDict,
    TankObservation,
    WorldStateDict,
    coord_key,
    make_self_state,
    make_tank_state,
    make_terrain_tile,
    viewport_scan_key,
)
from tankpit_bot.state.types.constants import DIRECTION_DEAD_THRESHOLD, TankLiveness
from tankpit_bot.state.viewport_geometry import (
    make_visible_viewport_state,
    viewport_patch_world_coords,
)

# =============================================================================
# World State Update Functions (from protocol messages)
# =============================================================================


def update_self_from_movement_response(
    state: WorldStateDict,
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
    leaderboard_position: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Update self state from MovementResponse message.

    Args:
        state: Current world state.
        tank_id: Player's tank ID.
        x: New X coordinate.
        y: New Y coordinate.
        team: Team ID.
        rank: Military rank.
        leaderboard_position: Leaderboard position.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated self state.
    """
    new_self = make_self_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
        fuel=state["self_state"]["fuel"] if state["self_state"] else 0,
        leaderboard_position=leaderboard_position,
    )
    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def update_self_position(
    state: WorldStateDict,
    x: int,
    y: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Update self position without changing viewport bounds.

    Updates the self_state x,y coordinates and preserves the current viewport.
    Creates a minimal self_state if none exists.

    Args:
        state: Current world state.
        x: New X coordinate.
        y: New Y coordinate.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated position.
    """
    # Get or create self_state with updated position
    if state["self_state"] is not None:
        new_self = SelfStateDict(
            tank_id=state["self_state"]["tank_id"],
            x=x,
            y=y,
            team=state["self_state"]["team"],
            rank=state["self_state"]["rank"],
            fuel=state["self_state"]["fuel"],
            leaderboard_position=state["self_state"]["leaderboard_position"],
        )
    else:
        # Create minimal self_state when we first learn position
        new_self = SelfStateDict(
            tank_id=0,
            x=x,
            y=y,
            team=0,
            rank=0,
            fuel=0,
            leaderboard_position=0,
        )

    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


# update_tank_from_registry and update_tank_damage were deleted
# 2026-06-19 with the freshness-model refactor. Every tank-state
# mutation now flows through apply_tank_observation, which enforces the
# three-timestamp freshness invariants. Callers build TankObservation
# values at the dispatch boundary; see
# tankpit_bot.sniffer.world_state_tanks for the wire-message-to-
# observation translators.


def apply_tank_observation(state: WorldStateDict, obs: TankObservation) -> WorldStateDict:
    """Apply a single tank observation, advancing exactly the right freshness.

    Single source of truth for tank-state mutations from wire and map
    observations. Freshness invariants -- LOCKED by tests in
    ``tests/state/test_tank_observation.py``:

    1. ``timestamp_ms`` advances to ``obs["timestamp_ms"]`` on every
       call (every observation is an observation).
    2. ``last_wire_seen_ms`` advances iff ``obs["is_wire_sourced"]`` is
       True. Map-snapshot observations leave it untouched, so a
       departed tank that the map still lists cannot masquerade as
       wire-present.
    3. ``last_position_update_ms`` advances iff BOTH
       ``obs["position_is_authoritative"]`` is True AND
       ``obs["position"]`` is not None. Damage-only wire messages
       (TankStatusSync, is_wire_sourced=True but position=None) leave
       it untouched. The 0x4C MAP_DATA snapshot advances it
       (position_is_authoritative=True, is_wire_sourced=False) so a
       stationary target stays kill-shot-fresh after the bot opens the
       map, without that snapshot lying about wire presence. Radar
       EnemyDetect and DOM-scraped client-registry refinements set
       both flags to False -- their position is a tile-coarse or
       out-of-band estimate that must not gate a kill shot.

    Field values: each present ``obs`` field overwrites the existing
    tank's corresponding value; each ``None`` field preserves the
    existing value. A non-existent tank is created with default values
    for fields the observation does not provide.

    Args:
        state: Current world state.
        obs: Observation event for one tank.

    Returns:
        New ``WorldStateDict`` with the tank updated or created. The
        outer ``state["timestamp_ms"]`` also advances to
        ``obs["timestamp_ms"]``.
    """
    key = str(obs["tank_id"])
    existing = state["tanks"].get(key)
    self_state = state["self_state"]
    is_self = self_state is not None and self_state["tank_id"] == obs["tank_id"]

    obs_position = obs["position"]
    if obs_position is not None:
        new_x, new_y = obs_position
    elif existing is not None:
        new_x, new_y = existing["x"], existing["y"]
    else:
        new_x, new_y = 0, 0

    new_team = obs["team"] if obs["team"] is not None else (existing["team"] if existing else 0)
    new_rank = obs["rank"] if obs["rank"] is not None else (existing["rank"] if existing else 0)
    new_damage = (
        obs["damage_state"]
        if obs["damage_state"] is not None
        else (existing["damage_state"] if existing else DAMAGE_FULL)
    )
    new_direction = (
        obs["direction"]
        if obs["direction"] is not None
        else (existing["direction"] if existing else 0)
    )
    new_name = obs["name"] if obs["name"] is not None else (existing["name"] if existing else "")
    new_is_bot = (
        obs["is_bot"] if obs["is_bot"] is not None else (existing["is_bot"] if existing else False)
    )

    timestamp_ms = obs["timestamp_ms"]
    if obs["is_wire_sourced"]:
        new_last_wire_seen_ms = timestamp_ms
    else:
        new_last_wire_seen_ms = existing["last_wire_seen_ms"] if existing else 0
    if obs["position_is_authoritative"] and obs_position is not None:
        new_last_position_update_ms = timestamp_ms
    else:
        new_last_position_update_ms = existing["last_position_update_ms"] if existing else 0
    # Viewport-observation gate. Only observations that the dispatch
    # layer routed through ``storage_source == "viewport"`` advance
    # this timestamp -- proof the tank was in the bot's local sensing
    # window when the wire arrived. 0x4C MapData and 0x2E TankStatusSync
    # are global broadcasts that route through ``"world_state"`` or
    # preserve the previous source, so they never advance the gate.
    # This is what ``analyze_threats`` reads to keep the threat list
    # to actually-visible enemies; without it, every alive tank on the
    # map shows up as a HUNT candidate after every ``open_map``.
    if obs["storage_source"] == "viewport":
        new_last_viewport_observation_ms = timestamp_ms
    else:
        new_last_viewport_observation_ms = (
            existing["last_viewport_observation_ms"] if existing else 0
        )

    # Liveness transition. Three rules, evaluated in order:
    #   1. New tank or wire-sourced position update with alive sprite
    #      direction (< 32) -- ``alive``. This covers fresh joins,
    #      MovementResponse arrivals, and the respawn flow (a previously
    #      deactivated tank that moved and emitted a non-corpse 0x3D).
    #   2. Any observation with a corpse direction (>= 32) -- mark
    #      ``deactivated``. tpclient.js Pg.prototype.h sets direction
    #      to 32/33 on deactivation; this rule catches both the kill
    #      we made (0x41 firing first, then 0x3D with dir=32) and the
    #      kill we observed someone else make (no 0x41 victim_id match
    #      but the 0x3D arrives).
    #   3. Otherwise preserve existing liveness. MapData entries
    #      (is_wire_sourced=False) don't change liveness; they're
    #      authoritative for position only.
    new_liveness: TankLiveness
    if existing is None or (
        obs["is_wire_sourced"]
        and obs_position is not None
        and (obs["direction"] is None or obs["direction"] < DIRECTION_DEAD_THRESHOLD)
    ):
        new_liveness = "alive"
    elif obs["direction"] is not None and obs["direction"] >= DIRECTION_DEAD_THRESHOLD:
        new_liveness = "deactivated"
    else:
        new_liveness = existing["liveness"]

    # Preserve aim-of-last-shot fields. Observations don't carry them
    # (only 0x53 ShootEvent writes them via ``set_tank_last_aim``); a
    # later regular wire update (movement, damage, viewport) must not
    # clobber the recorded barrel-aim.
    aim_x = existing["last_aim_x"] if existing else -1
    aim_y = existing["last_aim_y"] if existing else -1
    aim_weapon = existing["last_aim_weapon"] if existing else -1
    aim_ms = existing["last_aim_ms"] if existing else 0

    new_tank = make_tank_state(
        tank_id=obs["tank_id"],
        x=new_x,
        y=new_y,
        team=new_team,
        rank=new_rank,
        damage_state=new_damage,
        direction=new_direction,
        name=new_name,
        is_bot=new_is_bot,
        is_self=is_self,
        source=obs["storage_source"],
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=new_last_wire_seen_ms,
        last_position_update_ms=new_last_position_update_ms,
        last_viewport_observation_ms=new_last_viewport_observation_ms,
        liveness=new_liveness,
        last_aim_x=aim_x,
        last_aim_y=aim_y,
        last_aim_weapon=aim_weapon,
        last_aim_ms=aim_ms,
    )

    new_tanks = dict(state["tanks"])
    new_tanks[key] = new_tank

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=new_tanks,
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def update_terrain_from_viewport(
    state: WorldStateDict,
    viewport_left: int,
    viewport_top: int,
    entities: list[tuple[int, int, int, int, int]],
    timestamp_ms: int,
) -> WorldStateDict:
    """Update terrain from a visible viewport update.

    Args:
        state: Current world state.
        viewport_left: Viewport left X coordinate.
        viewport_top: Viewport top Y coordinate.
        entities: List of ``0x5A`` patch-grid
            ``(col, row, terrain_type, cache_value, overlay_value)`` tuples.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated terrain, viewport, and confirmed
        local-resource coverage for that viewport origin.
    """
    new_terrain = dict(state["terrain"])
    new_viewport = make_visible_viewport_state(viewport_left, viewport_top)
    key = viewport_scan_key(viewport_left, viewport_top)
    new_scanned_viewports = dict(state["scanned_viewports"])
    new_scanned_viewports[key] = timestamp_ms

    for col, row, terrain_type, cache_value, overlay_value in entities:
        x, y = viewport_patch_world_coords(viewport_left, viewport_top, col, row)
        key = coord_key(x, y)
        new_terrain[key] = make_terrain_tile(
            x=x,
            y=y,
            terrain_type=terrain_type,
            cache_value=cache_value,
            overlay_value=overlay_value,
        )

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=new_terrain,
        viewport=new_viewport,
        scanned_viewports=new_scanned_viewports,
        timestamp_ms=timestamp_ms,
    )


# update_self_fuel (delta variant) was deleted 2026-06-19: production
# wire fuel messages (0x2E TankStatusSync, 0x44 FuelGain, 0x64
# FuelDeposit) all carry the absolute fuel value and funnel through
# set_self_fuel; the additive-delta variant was dead in src/ and only
# alive in tests.


def set_self_fuel(
    state: WorldStateDict,
    fuel: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Set self fuel to absolute value (from inventory or sync messages).

    Args:
        state: Current world state.
        fuel: Absolute fuel value.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated fuel, or unchanged if no self_state.
    """
    if state["self_state"] is None:
        return state

    new_self = SelfStateDict(
        tank_id=state["self_state"]["tank_id"],
        x=state["self_state"]["x"],
        y=state["self_state"]["y"],
        team=state["self_state"]["team"],
        rank=state["self_state"]["rank"],
        fuel=max(0, fuel),
        leaderboard_position=state["self_state"]["leaderboard_position"],
    )

    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def set_self_rank(
    state: WorldStateDict,
    rank: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Set self rank to absolute value from a Promotion message (0x2B Rf).

    Args:
        state: Current world state.
        rank: New rank index (0-8 per the JS rank table).
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated self rank, or unchanged if no
        self_state has been established yet (rank can't precede join).
    """
    if state["self_state"] is None:
        return state

    new_self = SelfStateDict(
        tank_id=state["self_state"]["tank_id"],
        x=state["self_state"]["x"],
        y=state["self_state"]["y"],
        team=state["self_state"]["team"],
        rank=rank,
        fuel=state["self_state"]["fuel"],
        leaderboard_position=state["self_state"]["leaderboard_position"],
    )

    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def _set_tank_liveness(
    state: WorldStateDict,
    tank_id: int,
    liveness: TankLiveness,
    timestamp_ms: int,
) -> WorldStateDict:
    """Set a tank's liveness state. No-op when the tank is unknown.

    Args:
        state: Current world state.
        tank_id: Tank id to transition.
        liveness: Target liveness literal.
        timestamp_ms: Message timestamp.

    Returns:
        New ``WorldStateDict`` with the tank's liveness updated (and the
        outer ``timestamp_ms`` advanced).
    """
    key = str(tank_id)
    existing = state["tanks"].get(key)
    if existing is None:
        return state

    new_tank = make_tank_state(
        tank_id=existing["tank_id"],
        x=existing["x"],
        y=existing["y"],
        team=existing["team"],
        rank=existing["rank"],
        damage_state=existing["damage_state"],
        direction=existing["direction"],
        name=existing["name"],
        is_bot=existing["is_bot"],
        is_self=existing["is_self"],
        source=existing["source"],
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=existing["last_wire_seen_ms"],
        last_position_update_ms=existing["last_position_update_ms"],
        last_viewport_observation_ms=existing["last_viewport_observation_ms"],
        liveness=liveness,
        last_aim_x=existing["last_aim_x"],
        last_aim_y=existing["last_aim_y"],
        last_aim_weapon=existing["last_aim_weapon"],
        last_aim_ms=existing["last_aim_ms"],
    )

    new_tanks = dict(state["tanks"])
    new_tanks[key] = new_tank

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=new_tanks,
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def set_tank_last_aim(
    state: WorldStateDict,
    tank_id: int,
    aim_x: int,
    aim_y: int,
    weapon: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Record the barrel-aim of the last 0x53 ShootEvent for ``tank_id``.

    The four fields ``last_aim_x``, ``last_aim_y``, ``last_aim_weapon``,
    ``last_aim_ms`` on the tank state are written; everything else is
    preserved. No-op when the tank is unknown to the registry (the
    next per-tank wire will create it; the missed aim is OK to drop).

    Args:
        state: Current world state.
        tank_id: Shooter tank id.
        aim_x: Wire-reported barrel-aim X.
        aim_y: Wire-reported barrel-aim Y.
        weapon: Weapon byte (0=single, 1=dual, 2=missile, 3=homing).
        timestamp_ms: Message timestamp; written to ``last_aim_ms`` so
            consumers can age the aim out.

    Returns:
        New ``WorldStateDict`` with the tank's aim fields advanced (and
        the outer ``timestamp_ms`` advanced to match).
    """
    key = str(tank_id)
    existing = state["tanks"].get(key)
    if existing is None:
        return state

    new_tank = make_tank_state(
        tank_id=existing["tank_id"],
        x=existing["x"],
        y=existing["y"],
        team=existing["team"],
        rank=existing["rank"],
        damage_state=existing["damage_state"],
        direction=existing["direction"],
        name=existing["name"],
        is_bot=existing["is_bot"],
        is_self=existing["is_self"],
        source=existing["source"],
        timestamp_ms=existing["timestamp_ms"],
        last_wire_seen_ms=existing["last_wire_seen_ms"],
        last_position_update_ms=existing["last_position_update_ms"],
        last_viewport_observation_ms=existing["last_viewport_observation_ms"],
        liveness=existing["liveness"],
        last_aim_x=aim_x,
        last_aim_y=aim_y,
        last_aim_weapon=weapon,
        last_aim_ms=timestamp_ms,
    )

    new_tanks = dict(state["tanks"])
    new_tanks[key] = new_tank

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=new_tanks,
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def deactivate_tank(
    state: WorldStateDict,
    tank_id: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Mark a tank as a corpse on 0x41 Deactivation.

    Args:
        state: Current world state.
        tank_id: Victim tank id.
        timestamp_ms: Message timestamp.

    Returns:
        New ``WorldStateDict`` with the tank's liveness set to
        ``"deactivated"``. No-op when the tank is not in state.
    """
    return _set_tank_liveness(state, tank_id, "deactivated", timestamp_ms)


def remove_tank(
    state: WorldStateDict,
    tank_id: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """No-op handler for 0x58 TankRemove (the registry entry is kept).

    Per the 2026-06-20 ghost_observe capture, 0x58 TankRemove fires
    whenever the server stops broadcasting per-tank updates for a
    tank to this client -- this happens on actual deaths AND on
    benign tracking churn (orange-5 got five TankRemove events
    across two actual kills; three were churn). The old behavior
    was to delete the tank from the registry; that caused the bot
    to abandon pursuit of locked targets that merely teleported
    out of viewport (live capture 2026-06-22). 0x41 Deactivation
    is the only authoritative death signal -- it flips
    ``liveness`` to ``"deactivated"`` and the freshness gates do
    the rest.

    Keeping the tank in the registry lets ``find_locked_target_pursuit``
    keep firing homing shots toward the cached coords until either
    a real deactivation arrives or the global broadcast timestamp
    goes stale past the pursuit freshness window. The wire still
    refreshes ``timestamp_ms`` for every alive tank via 0x2E
    TankStatusSync (~every 2 s) so a truly gone tank ages out
    naturally.

    Args:
        state: Current world state.
        tank_id: Tank ID the server announced removal for. Unused;
            the entry stays in the registry.
        timestamp_ms: Message timestamp. Unused for the same reason.

    Returns:
        The input state unchanged. Treating this signal as "ignore"
        is correct because 0x58 carries no information that the
        existing freshness / liveness machinery cannot derive from
        the other wire messages.
    """
    del tank_id, timestamp_ms
    return state


def mark_viewport_scanned(
    state: WorldStateDict,
    viewport_left: int,
    viewport_top: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Record that a viewport origin has authoritative local-resource coverage.

    Args:
        state: Current world state.
        viewport_left: Viewport left X coordinate.
        viewport_top: Viewport top Y coordinate.
        timestamp_ms: Scan completion timestamp.

    Returns:
        New WorldStateDict with updated viewport confirmation metadata.
    """
    key = viewport_scan_key(viewport_left, viewport_top)
    new_scanned_viewports = dict(state["scanned_viewports"])
    new_scanned_viewports[key] = timestamp_ms
    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=new_scanned_viewports,
        timestamp_ms=timestamp_ms,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "apply_tank_observation",
    "mark_viewport_scanned",
    "remove_tank",
    "set_self_fuel",
    "set_tank_last_aim",
    "update_self_from_movement_response",
    "update_self_position",
    "update_terrain_from_viewport",
]
