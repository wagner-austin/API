"""Mutations to the observed-tank registry.

The observation-based mutator that every wire channel funnels
through, plus liveness, aim, and removal. Sibling of
:mod:`tankpit_bot.state.self_mutations`,
:mod:`tankpit_bot.state.terrain_mutations`, and
:mod:`tankpit_bot.state.container_mutations`.
"""

from __future__ import annotations

from tankpit_bot.facts.provenance import make_provenance
from tankpit_bot.state.types import (
    TankObservation,
    WorldStateDict,
    has_real_coordinates,
    make_tank_state,
)
from tankpit_bot.types.constants import (
    DAMAGE_FULL,
    DIRECTION_DEAD_THRESHOLD,
    TankLiveness,
)

MAP_POSITION_DEFER_MS = 2000


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
    # Map-position freshness defer (archive-mined 2026-08-06,
    # analysis_scripts/mine_map_position_delta.py — 2,851 same-tank
    # map/wire pairs: the 0x4C payload is a snapshot that AGES before
    # arrival; 53% disagree with a within-2s wire fix by a movement
    # spectrum of walk steps and teleport hops, with zero decode
    # artifacts). Presence stays exact, so liveness rule 3 below is
    # untouched, but a map position must never overwrite a position
    # updated within the snapshot's own aging window — arrival time
    # lies about content age. A stationary tank (no update in the
    # window) still takes the map position and its freshness.
    if (
        obs["position_is_authoritative"]
        and not obs["is_wire_sourced"]
        and obs_position is not None
        and existing is not None
        and has_real_coordinates(existing)
        and obs["timestamp_ms"] - existing["last_position_update_ms"] < MAP_POSITION_DEFER_MS
    ):
        obs_position = None
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

    # Liveness transition. Four rules, evaluated in order:
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
    #   3. A deactivated tank LISTED in a map snapshot is alive -- the
    #      server's map is a strictly living-tanks list (rule body
    #      below has the measurements). Radar and DOM refinements
    #      (both flags False) stay excluded.
    #   4. Otherwise preserve existing liveness -- including
    #      damage-only syncs and the corpse window's trailing wire.
    new_liveness: TankLiveness
    if existing is None or (
        obs["is_wire_sourced"]
        and obs_position is not None
        and (obs["direction"] is None or obs["direction"] < DIRECTION_DEAD_THRESHOLD)
    ):
        new_liveness = "alive"
    elif obs["direction"] is not None and obs["direction"] >= DIRECTION_DEAD_THRESHOLD:
        new_liveness = "deactivated"
    elif (
        existing["liveness"] == "deactivated"
        and not obs["is_wire_sourced"]
        and obs["position_is_authoritative"]
    ):
        # Rule 4 — map presence IS life (byte-proven 2026-08-05). The
        # 0x4C map snapshot is the only observation with this flag
        # pair (position-authoritative, not wire-sourced), and the
        # server curates it as a strictly LIVING-tanks list: across
        # session bot-20260805-095935, victims were absent from all
        # 58 in-corpse-window snapshots and present in all 204
        # post-window ones; the 08-03 run's human (Belton, id kept
        # across 3 deaths) vanished for every corpse window and
        # returned at +24 s. So a deactivated tank LISTED in map data
        # cannot be its corpse — it is the living respawn. This is
        # the ONLY revival path for idle respawns, which emit no wire
        # at all (27 of 32 victims never sent another byte; only
        # MOVING respawns self-revive via rule 1's global 0x3D).
        # Without this rule the registry filled with phantom corpses
        # and the session exited no_viable_targets in a room of 27
        # live enemies. The June "afterimage" fear that motivated
        # map-never-touches-liveness is disproven by the same
        # measurement: the server drops the dead from the map within
        # the same second they die.
        new_liveness = "alive"
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
        provenance=make_provenance(obs["fact_source"], []),
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
        scanned_tiles=state["scanned_tiles"],
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
        scanned_tiles=state["scanned_tiles"],
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
        scanned_tiles=state["scanned_tiles"],
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


def depart_tank(
    state: WorldStateDict,
    tank_id: int,
) -> WorldStateDict:
    """Delete a tank the 0x29 TankExit announced has left the game.

    0x29 is NOT 0x58: TankRemove is per-client tracking churn
    (deliberately a no-op above), while TankExit is the server's
    room-wide announcement that the tank LEFT or was ELIMINATED --
    the same signal that renders the "left the game" client banner.
    Ghost precedent (run arterial 2026-08-13 23:39-23:43, flag
    s11-6): the 0x29 for Artax arrived at 23:39:17, was dropped as a
    diagnostic-only event, and the registry ghost then powered a
    ~75 s map-open/teleport/shoot chase until a shot's err=3
    friendly-fire receipt said the same thing the wire already had.
    Deleting the entry ends every downstream pursuit at once:
    ``analyze_threats`` and ``find_locked_target_pursuit`` both read
    the registry, so the lock releases through the existing
    gone-from-registry path, and a rejoin (same id, fresh session)
    re-adds the tank via its first observation.

    Args:
        state: Current world state.
        tank_id: Tank ID the server announced the exit for.

    Returns:
        World state without the departed tank's registry entry (the
        input state unchanged when the id was not tracked).
    """
    key = str(tank_id)
    if key not in state["tanks"]:
        return state
    remaining = {tid: tank for tid, tank in state["tanks"].items() if tid != key}
    return WorldStateDict(**{**state, "tanks": remaining})


__all__ = [
    "MAP_POSITION_DEFER_MS",
    "apply_tank_observation",
    "deactivate_tank",
    "depart_tank",
    "remove_tank",
    "set_tank_last_aim",
]
