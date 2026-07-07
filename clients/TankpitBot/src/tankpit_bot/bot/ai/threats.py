"""Enemy threat analysis from world state.

Pure functions that convert raw world state tank data into sorted,
analyzed EnemyThreatDict lists for use by behavior evaluators.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.types import EnemyThreatDict, make_enemy_threat
from tankpit_bot.state.types import SelfStateDict, TankStateDict, WorldStateDict


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Compute Manhattan distance between two points.

    Args:
        x1: First point X coordinate.
        y1: First point Y coordinate.
        x2: Second point X coordinate.
        y2: Second point Y coordinate.

    Returns:
        Manhattan distance (|x1-x2| + |y1-y2|).
    """
    return abs(x1 - x2) + abs(y1 - y2)


def _is_enemy(tank: TankStateDict, self_team: int) -> bool:
    """Check if a tank is an enemy (different team, not self).

    Args:
        tank: Tank state to check.
        self_team: Player's team ID.

    Returns:
        True if tank is on a different team and not the player.
    """
    return not tank["is_self"] and tank["team"] != self_team


# Legacy wire-presence horizon -- kept as a constant for test clock
# arithmetic (advance_clock past this value to age a target out of the
# viewport-confirmed threat list). The combat-side gate that used this
# was removed 2026-06-23: off-viewport pursuit shots fire toward the
# last known wire position via _locked_target_pursuit, so wire silence
# is no longer a stop signal.
_WIRE_PRESENCE_TTL_MS = 7000

#: Public alias for cross-module consumers (tests).
WIRE_PRESENCE_TTL_MS = _WIRE_PRESENCE_TTL_MS

# The HUNT acquisition gate. Only tanks confirmed by viewport-bound
# wire (``storage_source == "viewport"``) within this window are
# eligible to engage. Live-run 2026-06-21 tracking probe captured
# the cost of skipping this gate: open_map's 0x4C MapData refreshed
# every-tank timestamps, then global TankStatusSync kept them fresh,
# so analyze_threats returned 27 tanks while the JS client's
# viewport registry held only 1. Set to 5 s -- viewport-bound
# updates (0x47 Movement, 0x3D MovementResponse, 0x28 TankEntry)
# arrive every 1-3 s for tanks the bot can actually see, so 5 s
# tolerates short cadence gaps but rejects the 5-6 s global
# broadcast cycle that wire-presence alone cannot.
_VIEWPORT_PRESENCE_TTL_MS = 5000

#: Public alias for cross-module consumers.
VIEWPORT_PRESENCE_TTL_MS = _VIEWPORT_PRESENCE_TTL_MS

# Position-bearing observations (0x3D MovementResponse, 0x47 Movement,
# 0x28 TankEntry, container TankUpdate*, radar response, MAP_DATA) refresh
# this timestamp. Combat tempo is ~4-6 s per cycle (teleport + scan +
# shoot + next decision); a 3 s TTL false-positives "stale-position" on
# stationary targets we are actively fighting, so the bot would block
# the live target after one hit and switch to a new one. Live run
# 20260620-191622: 22 map_opens / 19 teleports / 0 kills because the
# gate kept firing block_combat_target_and_replan after each shoot.
# Match the wire-presence TTL so a stationary target stays engageable
# as long as it is still talking on the wire (TankStatusSync every
# ~2 s). The long-stale "afterimage" case is now handled at acquisition
# time by the viewport-presence gate in analyze_threats; the wire-side
# combat gate was removed 2026-06-23.
_POSITION_FRESHNESS_TTL_MS = 7000

#: Public alias for cross-module consumers (combat_strategy).
POSITION_FRESHNESS_TTL_MS = _POSITION_FRESHNESS_TTL_MS


def analyze_threats(
    world: WorldStateDict,
    self_state: SelfStateDict,
    now_ms: int,
) -> list[EnemyThreatDict]:
    """Analyze all enemy tanks and return sorted threat list.

    Filters to enemy tanks only (different team, not self) confirmed
    alive and observed via viewport-bound wire within
    :data:`VIEWPORT_PRESENCE_TTL_MS` -- the viewport observation gate
    proves the position is current and the tank is actually visible to
    the bot. Computes Manhattan distance from the player and sorts by
    distance ascending.

    Args:
        world: Current world state with tank positions.
        self_state: Player's own state for position and team.
        now_ms: Current tick timestamp for viewport-freshness filtering.

    Returns:
        List of EnemyThreatDict sorted by distance ascending.
    """
    self_x = self_state["x"]
    self_y = self_state["y"]
    self_team = self_state["team"]

    threats: list[EnemyThreatDict] = []
    for tank in world["tanks"].values():
        if not _is_enemy(tank, self_team):
            continue
        # Liveness gate. ``deactivated`` is the corpse window after a
        # kill (empirical ~22 s, 2026-06-20 capture). 0x58 TankRemove
        # is NOT a death signal -- it's tracking removal -- so there
        # is no separate ``removed`` state; removed tanks are simply
        # deleted from the registry and re-added by the next MapData
        # or per-tank wire. The previous explicit ``direction >= 32``
        # corpse-sprite check was replaced by this single liveness
        # filter -- ``apply_tank_observation`` now routes both
        # corpse-direction wire arrivals and 0x41 Deactivation to
        # ``liveness == "deactivated"``.
        if tank["liveness"] != "alive":
            continue
        # Position-confirmation sentinel. A tank we only know via
        # 0x21 TankInfo carries name and team but no position; it sits
        # at default (0, 0). Acquiring it would fire at tile (0, 0).
        # Wait for a position-bearing wire (0x3D MovementResponse,
        # 0x28 TankEntry, 0x47 Movement, etc.) before treating it as
        # a threat. The same gate filtered deactivation sentinels
        # before the liveness machine landed; now the liveness gate
        # owns deactivation and this one owns the unsynced case.
        if tank["x"] == 0 and tank["y"] == 0:
            continue
        # Viewport-observation gate. ``timestamp_ms`` and
        # ``last_wire_seen_ms`` are both refreshed by global
        # broadcasts (0x4C MapData, 0x2E TankStatusSync) that fire
        # for every alive tank on the map -- they cannot
        # distinguish "in viewport" from "anywhere on the map".
        # ``last_viewport_observation_ms`` only advances when the
        # dispatch layer routes the observation through
        # ``storage_source == "viewport"`` -- the actual viewport-
        # bound proof.
        if now_ms - tank["last_viewport_observation_ms"] > VIEWPORT_PRESENCE_TTL_MS:
            continue
        dist = manhattan_distance(self_x, self_y, tank["x"], tank["y"])
        threats.append(
            make_enemy_threat(
                tank_id=tank["tank_id"],
                x=tank["x"],
                y=tank["y"],
                distance=dist,
                damage_state=tank["damage_state"],
                rank=tank["rank"],
                team=tank["team"],
                name=tank["name"],
                is_bot=tank["is_bot"],
                timestamp_ms=tank["timestamp_ms"],
                last_wire_seen_ms=tank["last_wire_seen_ms"],
                last_position_update_ms=tank["last_position_update_ms"],
                last_aim_x=tank["last_aim_x"],
                last_aim_y=tank["last_aim_y"],
                last_aim_weapon=tank["last_aim_weapon"],
                last_aim_ms=tank["last_aim_ms"],
            )
        )

    threats.sort(key=_threat_sort_key)
    return threats


def _finish_priority(damage_state: int) -> int:
    """Rank a damage tier for finish-off preference, most damaged first.

    The tier COUNTS DOWN toward deactivation (live run 20260610-231x:
    every fight ran 0 -> 3 -> 2 -> 1 and all five kills with tier data
    died from tier 1), so tier 1 is the closest to dead and tier 0
    (full or never synced) is the least attractive.

    Args:
        damage_state: Wire damage tier (0 = full/unsynced, 3 = light,
            2 = medium, 1 = critical).

    Returns:
        Ascending rank where the most damaged enemy ranks first.
    """
    return 4 if damage_state == 0 else damage_state


def _threat_sort_key(threat: EnemyThreatDict) -> tuple[int, int, int]:
    """Sort key: distance, then finish-off priority, then freshness.

    Closer threats come first. Among threats at equal distance, more
    damaged enemies come first (easier to finish off). Among equal
    distance and damage, prefer recently confirmed tanks.

    Args:
        threat: Enemy threat to compute sort key for.

    Returns:
        Tuple of (distance, finish_priority, -timestamp_ms) for sorting.
    """
    return (
        threat["distance"],
        _finish_priority(threat["damage_state"]),
        -threat["timestamp_ms"],
    )


# A second enemy this close to a target can reach our fight tile during
# the engagement: shots land at Manhattan <= 2 and the close-walk
# approach covers 6 more, so an 8-tile neighbor is one walk away from
# joining the fight. The bot cannot win 1-vN exchanges (user-confirmed
# tactical constraint), so targets with neighbors inside this radius
# rank behind isolated ones.


def find_closest_threat(
    threats: list[EnemyThreatDict],
) -> EnemyThreatDict | None:
    """Get the closest enemy threat.

    Args:
        threats: Sorted threat list from analyze_threats.

    Returns:
        Closest EnemyThreatDict, or None if no threats.
    """
    if not threats:
        return None
    return threats[0]


def threats_in_range(
    threats: list[EnemyThreatDict],
    combat_range: int,
) -> list[EnemyThreatDict]:
    """Filter threats to those within combat range.

    Args:
        threats: Sorted threat list from analyze_threats.
        combat_range: Maximum Manhattan distance for combat engagement.

    Returns:
        List of threats within combat_range, preserving sort order.
    """
    return [t for t in threats if t["distance"] <= combat_range]


def find_locked_target_pursuit(
    world: WorldStateDict,
    self_state: SelfStateDict,
    locked_target_id: int,
    killed: dict[str, int],
) -> EnemyThreatDict | None:
    """Build a pursuit threat for a locked target that left the viewport.

    Distinguishes "alive but moved" from "actually dead." When
    ``analyze_threats`` no longer returns the locked target (because
    they left the viewport and the strict viewport-presence gate
    filters them out), this function checks ``world["tanks"]`` for
    the same ``locked_target_id``. If the tank is still alive it
    synthesises an :class:`EnemyThreatDict` at the cached coords so
    HUNT can keep firing homing shots toward the last known position.
    ``SCAN_ON_LANDING`` handles viewport confirmation on arrival,
    so this never re-introduces the phantom-firing bug -- firing
    still requires a strict ``analyze_threats`` hit on the next
    tick when the target is back in viewport.

    Pursuit stops only on authoritative death signals: the locked
    id landing in ``killed`` (kill cooldown applied) or the tank's
    ``liveness`` flipping to ``"deactivated"`` via 0x41. The earlier
    freshness-window gate was removed 2026-06-22 because it tripped
    when the server stopped broadcasting 0x2E TankStatusSync for a
    far-away tank, ending pursuit prematurely (live capture
    18:17:00: 3 pursuit homings hit purple-4 then gate tripped on
    timestamp staleness even though purple-4 was very much alive).
    Ammo on dual / homing shots only decrements on confirmed hit,
    so a target that's truly unreachable burns zero ammo -- there's
    no over-fire cost for indefinite pursuit.

    Args:
        world: Filtered world state (killed tanks already removed).
        self_state: Player's own state.
        locked_target_id: ``combat_target_id`` from current AI state.
        killed: Tank IDs on kill cooldown.

    Returns:
        Pursuit :class:`EnemyThreatDict` synthesised from world state,
        or ``None`` when the locked target is genuinely missing
        (no entry, dead, or no lock).
    """
    if locked_target_id == -1:
        return None
    if str(locked_target_id) in killed:
        return None
    tank = world["tanks"].get(str(locked_target_id))
    if tank is None:
        return None
    if tank["liveness"] != "alive":
        return None
    if tank["x"] == 0 and tank["y"] == 0:
        return None
    dist = manhattan_distance(self_state["x"], self_state["y"], tank["x"], tank["y"])
    return make_enemy_threat(
        tank_id=tank["tank_id"],
        x=tank["x"],
        y=tank["y"],
        distance=dist,
        damage_state=tank["damage_state"],
        rank=tank["rank"],
        team=tank["team"],
        name=tank["name"],
        is_bot=tank["is_bot"],
        timestamp_ms=tank["timestamp_ms"],
        last_wire_seen_ms=tank["last_wire_seen_ms"],
        last_position_update_ms=tank["last_position_update_ms"],
        last_aim_x=tank["last_aim_x"],
        last_aim_y=tank["last_aim_y"],
        last_aim_weapon=tank["last_aim_weapon"],
        last_aim_ms=tank["last_aim_ms"],
    )


def _acquisition_rejection_reason(
    tank: TankStateDict,
    self_state: SelfStateDict,
    blocked: dict[str, int],
    killed: dict[str, int],
    terrain: TerrainMapProtocol | None,
    now_ms: int,
    map_open_cooldown_ms: int,
    engagement_reserve_fuel: int,
) -> str | None:
    """Return why an enemy fails the acquisition gates, or ``None`` if viable.

    Args:
        tank: Enemy tank under consideration.
        self_state: Player's own state.
        blocked: Tank IDs temporarily un-engageable.
        killed: Tank IDs on kill cooldown.
        terrain: Terrain map for passable-adjacent check.
        now_ms: Current tick timestamp.
        map_open_cooldown_ms: Freshness window for map-known positions.
        engagement_reserve_fuel: Fuel that must remain after the
            approach teleport.

    Returns:
        Rejection reason string, or ``None`` when the enemy is viable.
    """
    from tankpit_bot.bot.ai.combat_strategy import has_passable_adjacent
    from tankpit_bot.bot.ai.teleport_cost import compute_teleport_fuel_cost

    if tank["liveness"] != "alive":
        return "not_alive"
    if tank["x"] == 0 and tank["y"] == 0:
        return "unsynced_position"
    if str(tank["tank_id"]) in killed:
        return "killed_cooldown"
    if str(tank["tank_id"]) in blocked:
        return "blocked"
    if now_ms - tank["timestamp_ms"] >= map_open_cooldown_ms:
        return "stale_map_data"
    if not has_passable_adjacent(tank["x"], tank["y"], terrain):
        return "no_passable_adjacent"
    teleport_cost = compute_teleport_fuel_cost(
        self_state["x"],
        self_state["y"],
        tank["x"],
        tank["y"],
    )
    if teleport_cost + engagement_reserve_fuel > self_state["fuel"]:
        return "unaffordable"
    return None


def find_acquisition_target(
    world: WorldStateDict,
    self_state: SelfStateDict,
    blocked: dict[str, int],
    killed: dict[str, int],
    terrain: TerrainMapProtocol | None,
    now_ms: int,
    map_open_cooldown_ms: int,
    *,
    engagement_reserve_fuel: int,
) -> EnemyThreatDict | None:
    """Pick the nearest map-fresh enemy the bot can afford to fight.

    This is the **acquisition** gate, deliberately looser than the
    **firing** gate in :func:`analyze_threats`. Firing requires
    ``last_viewport_observation_ms`` freshness (the bot must have
    direct viewport evidence) because phantom firing at a snapshot
    position wastes shots and reveals the bot. Acquisition only
    requires ANY observation within the map-open cooldown -- the
    teleport just gets the bot near the enemy; ``SCAN_ON_LANDING``
    handles viewport confirmation before any shot.

    Filters: enemy team, alive, position not (0,0), not on
    ``killed`` or ``blocked`` lists, has a passable adjacent tile
    for a combat landing, ``timestamp_ms`` within
    ``map_open_cooldown_ms``, and **affordable end-to-end**: current
    fuel must cover the approach teleport plus
    ``engagement_reserve_fuel`` (a realistic kill cost plus the
    fuel-low reserve). The user contract (2026-07-02) is that the bot
    never picks a fight it cannot pay for -- live run 2026-07-01
    20:45 spent 505 fuel reaching the nearest enemy and hit the
    fuel-low interrupt eight shots into a fight it could not finish.
    Returns the closest survivor by Manhattan distance.

    Args:
        world: Filtered world state (killed tanks already removed).
        self_state: Player's own state.
        blocked: Tank IDs temporarily un-engageable.
        killed: Tank IDs on kill cooldown.
        terrain: Terrain map for passable-adjacent check.
        now_ms: Current tick timestamp.
        map_open_cooldown_ms: Freshness window for map-known positions.
        engagement_reserve_fuel: Fuel that must remain after the
            approach teleport (kill budget + fuel-low reserve).

    Returns:
        Nearest affordable acquisition target as
        :class:`EnemyThreatDict`, or ``None`` when no map-fresh enemy
        is viable.
    """
    from platform_core.json_utils import JSONObject, dump_json_str

    from tankpit_bot.runtime_logging import emit_diagnostic

    self_x = self_state["x"]
    self_y = self_state["y"]
    self_team = self_state["team"]

    candidates: list[EnemyThreatDict] = []
    candidate_log: list[JSONObject] = []
    for tank in world["tanks"].values():
        if not _is_enemy(tank, self_team):
            continue
        dist = manhattan_distance(self_x, self_y, tank["x"], tank["y"])
        rejected_reason = _acquisition_rejection_reason(
            tank,
            self_state,
            blocked,
            killed,
            terrain,
            now_ms,
            map_open_cooldown_ms,
            engagement_reserve_fuel,
        )
        candidate_log.append(
            {
                "tank_id": tank["tank_id"],
                "name": tank["name"],
                "x": tank["x"],
                "y": tank["y"],
                "dist": dist,
                "rejected_reason": rejected_reason,
            }
        )
        if rejected_reason is not None:
            continue
        candidates.append(
            make_enemy_threat(
                tank_id=tank["tank_id"],
                x=tank["x"],
                y=tank["y"],
                distance=dist,
                damage_state=tank["damage_state"],
                rank=tank["rank"],
                team=tank["team"],
                name=tank["name"],
                is_bot=tank["is_bot"],
                timestamp_ms=tank["timestamp_ms"],
                last_wire_seen_ms=tank["last_wire_seen_ms"],
                last_position_update_ms=tank["last_position_update_ms"],
                last_aim_x=tank["last_aim_x"],
                last_aim_y=tank["last_aim_y"],
                last_aim_weapon=tank["last_aim_weapon"],
                last_aim_ms=tank["last_aim_ms"],
            )
        )

    candidates.sort(key=_threat_sort_key)
    winner = candidates[0] if candidates else None
    emit_diagnostic(
        diagnostic_kind="acquisition_candidates",
        self_x=self_x,
        self_y=self_y,
        total_enemies=len(candidate_log),
        accepted_count=len(candidates),
        picked_id=winner["tank_id"] if winner is not None else -1,
        picked_name=winner["name"] if winner is not None else "",
        picked_x=winner["x"] if winner is not None else -1,
        picked_y=winner["y"] if winner is not None else -1,
        picked_dist=winner["distance"] if winner is not None else -1,
        candidates_json=dump_json_str({"candidates": candidate_log}),
    )
    return winner


def find_relay_travel_target(
    world: WorldStateDict,
    self_state: SelfStateDict,
    blocked: dict[str, int],
    killed: dict[str, int],
    terrain: TerrainMapProtocol | None,
    now_ms: int,
    map_open_cooldown_ms: int,
    *,
    engagement_reserve_fuel: int,
) -> EnemyThreatDict | None:
    """Pick the nearest map-fresh enemy that fails ONLY the affordability gate.

    The dot-relay travel planner needs a destination worth travelling
    toward: an enemy that would be a perfectly viable acquisition if
    the bot had the fuel for the end-to-end fight. Every other gate
    (alive, synced, not blocked/killed, map-fresh, passable-adjacent)
    must pass -- travelling toward a corpse or a blocked target wastes
    the relay.

    Args:
        world: Filtered world state (killed tanks already removed).
        self_state: Player's own state.
        blocked: Tank IDs temporarily un-engageable.
        killed: Tank IDs on kill cooldown.
        terrain: Terrain map for passable-adjacent check.
        now_ms: Current tick timestamp.
        map_open_cooldown_ms: Freshness window for map-known positions.
        engagement_reserve_fuel: Fuel that must remain after the
            approach teleport (kill budget + fuel-low reserve).

    Returns:
        Nearest unaffordable-but-otherwise-viable enemy, or ``None``
        when no enemy is worth relaying toward.
    """
    self_x = self_state["x"]
    self_y = self_state["y"]
    self_team = self_state["team"]

    candidates: list[EnemyThreatDict] = []
    for tank in world["tanks"].values():
        if not _is_enemy(tank, self_team):
            continue
        rejected_reason = _acquisition_rejection_reason(
            tank,
            self_state,
            blocked,
            killed,
            terrain,
            now_ms,
            map_open_cooldown_ms,
            engagement_reserve_fuel,
        )
        if rejected_reason != "unaffordable":
            continue
        candidates.append(
            make_enemy_threat(
                tank_id=tank["tank_id"],
                x=tank["x"],
                y=tank["y"],
                distance=manhattan_distance(self_x, self_y, tank["x"], tank["y"]),
                damage_state=tank["damage_state"],
                rank=tank["rank"],
                team=tank["team"],
                name=tank["name"],
                is_bot=tank["is_bot"],
                timestamp_ms=tank["timestamp_ms"],
                last_wire_seen_ms=tank["last_wire_seen_ms"],
                last_position_update_ms=tank["last_position_update_ms"],
                last_aim_x=tank["last_aim_x"],
                last_aim_y=tank["last_aim_y"],
                last_aim_weapon=tank["last_aim_weapon"],
                last_aim_ms=tank["last_aim_ms"],
            )
        )

    candidates.sort(key=_threat_sort_key)
    return candidates[0] if candidates else None


__all__ = [
    "POSITION_FRESHNESS_TTL_MS",
    "VIEWPORT_PRESENCE_TTL_MS",
    "WIRE_PRESENCE_TTL_MS",
    "analyze_threats",
    "find_acquisition_target",
    "find_closest_threat",
    "find_locked_target_pursuit",
    "find_relay_travel_target",
    "manhattan_distance",
    "threats_in_range",
]
