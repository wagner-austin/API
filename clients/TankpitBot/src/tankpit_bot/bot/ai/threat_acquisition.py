"""Target acquisition: choose a new enemy, or a relay hop toward one.

Owns the rejection reasons that gate acquisition and the three
searches built on them -- direct acquisition, the stale-human probe,
and relay travel. Reads
:mod:`tankpit_bot.bot.ai.threat_primitives` only.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.humans import (
    DEFAULT_HUMAN_MAX_RANK,
    DEFAULT_HUMAN_MIN_RANK,
    is_human_rank_protected,
)
from tankpit_bot.bot.ai.threat_primitives import (
    _is_enemy,
    _threat_sort_key_for,
    fleet_assist_ids,
    human_combat_consented,
    make_enemy_threat_from_tank,
    manhattan_distance,
)
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.protocol.naming import is_human_name
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    SelfStateDict,
    TankStateDict,
    WorldStateDict,
    has_known_position,
)


def _acquisition_rejection_reason(
    ws: WorldService,
    tank: TankStateDict,
    self_state: SelfStateDict,
    blocked: dict[str, int],
    killed: dict[str, int],
    terrain: TerrainMapProtocol | None,
    now_ms: int,
    map_open_cooldown_ms: int,
    engagement_reserve_fuel: int,
    human_min_rank: int,
    human_max_rank: int,
) -> str | None:
    """Return why an enemy fails the acquisition gates, or ``None`` if viable.

    Args:
        tank: Enemy tank under consideration.
        self_state: Player's own state.
        blocked: Tank IDs temporarily un-engageable.
        killed: Tank IDs on kill cooldown.
        terrain: Terrain map for the stand-off landing check.
        now_ms: Current tick timestamp.
        map_open_cooldown_ms: Freshness window for map-known positions.
        engagement_reserve_fuel: Fuel that must remain after the
            approach teleport.

    Returns:
        Rejection reason string, or ``None`` when the enemy is viable.
    """
    from tankpit_bot.bot.ai.combat_target import has_standoff_landing
    from tankpit_bot.physics.costs import teleport_cost

    if tank["liveness"] != "alive":
        return "not_alive"
    if is_human_rank_protected(
        tank["name"],
        tank["rank"],
        min_rank=human_min_rank,
        max_rank=human_max_rank,
    ):
        # User ruling 2026-07-28: humans outside the configured rank
        # window are never targeted (recruits below the floor; high
        # ranks above a respect ceiling when one is set). Checked
        # before the affordability gate so the relay path (which
        # accepts only "unaffordable" rejections) can never travel
        # toward one.
        return "protected_human_rank"
    if is_human_name(tank["name"]) and not human_combat_consented(ws, tank["tank_id"]):
        # Human-consent contract (2026-07-30): no acquisition of a
        # human who has neither responded to the HELLO nor engaged
        # first. Placed before the affordability gate for the same
        # relay-path reason as the rank window above.
        return "human_not_consented"
    if not has_known_position(tank):
        return "unsynced_position"
    if str(tank["tank_id"]) in killed:
        return "killed_cooldown"
    if str(tank["tank_id"]) in blocked:
        return "blocked"
    if now_ms - tank["timestamp_ms"] >= map_open_cooldown_ms:
        return "stale_map_data"
    if not has_standoff_landing(tank["x"], tank["y"], terrain):
        return "no_standoff_landing"
    approach_cost = teleport_cost(
        self_state["x"],
        self_state["y"],
        tank["x"],
        tank["y"],
    )
    if approach_cost + engagement_reserve_fuel > self_state["fuel"]:
        return "unaffordable"
    return None


def find_acquisition_target(
    ws: WorldService,
    world: WorldStateDict,
    self_state: SelfStateDict,
    blocked: dict[str, int],
    killed: dict[str, int],
    terrain: TerrainMapProtocol | None,
    now_ms: int,
    map_open_cooldown_ms: int,
    *,
    engagement_reserve_fuel: int,
    priority_target_name: str = "",
    human_min_rank: int = DEFAULT_HUMAN_MIN_RANK,
    human_max_rank: int = DEFAULT_HUMAN_MAX_RANK,
) -> EnemyThreatDict | None:
    """Pick the highest-priority map-fresh enemy the bot can afford.

    Human-classified enemies outrank every practice bot regardless of
    distance, and the configured priority account outranks even other
    humans (user doctrine 2026-07-28); within a tier the pick is
    nearest-first.

    This is the **acquisition** gate, deliberately looser than the
    **firing** gate in :func:`analyze_threats`. Firing requires
    ``last_viewport_observation_ms`` freshness (the bot must have
    direct viewport evidence) because phantom firing at a snapshot
    position wastes shots and reveals the bot. Acquisition only
    requires ANY observation within the map-open cooldown -- the
    teleport just gets the bot near the enemy; ``SCAN_ON_LANDING``
    handles viewport confirmation before any shot.

    Filters: enemy team, alive, position not (0,0), not on
    ``killed`` or ``blocked`` lists, has a passable stand-off
    landing within shot range, ``timestamp_ms`` within
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
        terrain: Terrain map for the stand-off landing check.
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
            ws,
            tank,
            self_state,
            blocked,
            killed,
            terrain,
            now_ms,
            map_open_cooldown_ms,
            engagement_reserve_fuel,
            human_min_rank,
            human_max_rank,
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
        candidates.append(make_enemy_threat_from_tank(tank, dist))

    candidates.sort(key=_threat_sort_key_for(priority_target_name, fleet_assist_ids(ws)))
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


def stale_human_exists(
    ws: WorldService,
    world: WorldStateDict,
    self_state: SelfStateDict,
    blocked: dict[str, int],
    killed: dict[str, int],
    terrain: TerrainMapProtocol | None,
    now_ms: int,
    map_open_cooldown_ms: int,
    *,
    engagement_reserve_fuel: int,
    human_min_rank: int = DEFAULT_HUMAN_MIN_RANK,
    human_max_rank: int = DEFAULT_HUMAN_MAX_RANK,
) -> bool:
    """Return whether a pursuit-worthy human exists with STALE map data.

    The freshness asymmetry that hid Yuppler (run 2026-07-29 21:19):
    practice bots move and shoot constantly, so the wire keeps them
    permanently map-fresh; a QUIET human generates no wire traffic and
    goes stale ``map_open_cooldown_ms`` after every map open. With a
    wire-fresh bot always available, acquisition never needed another
    map open and the human stayed invisible outside 5-second windows.
    The acquire path uses this predicate to force a map refresh before
    settling for bot farming (user doctrine: "farm bots but prioritize
    any human player that logs in").

    A human rejected for any OTHER reason (protected rank, blocked,
    killed-cooldown, dead) is not worth a refresh -- only the
    ``stale_map_data`` rejection is curable by a map open.

    Args:
        world: Filtered world state (killed tanks already removed).
        self_state: Player's own state.
        blocked: Tank IDs temporarily un-engageable.
        killed: Tank IDs on kill cooldown.
        terrain: Terrain map for the stand-off landing check.
        now_ms: Current tick timestamp.
        map_open_cooldown_ms: Freshness window for map-known positions.
        engagement_reserve_fuel: Fuel that must remain after the
            approach teleport.

    Returns:
        True when at least one rank-window human's only curable defect
        is stale map data.
    """
    for tank in world["tanks"].values():
        if not _is_enemy(tank, self_state["team"]):
            continue
        if not is_human_name(tank["name"]):
            continue
        rejected_reason = _acquisition_rejection_reason(
            ws,
            tank,
            self_state,
            blocked,
            killed,
            terrain,
            now_ms,
            map_open_cooldown_ms,
            engagement_reserve_fuel,
            human_min_rank,
            human_max_rank,
        )
        if rejected_reason == "stale_map_data":
            return True
    return False


def find_relay_travel_target(
    ws: WorldService,
    world: WorldStateDict,
    self_state: SelfStateDict,
    blocked: dict[str, int],
    killed: dict[str, int],
    terrain: TerrainMapProtocol | None,
    now_ms: int,
    map_open_cooldown_ms: int,
    *,
    engagement_reserve_fuel: int,
    priority_target_name: str = "",
    human_min_rank: int = DEFAULT_HUMAN_MIN_RANK,
    human_max_rank: int = DEFAULT_HUMAN_MAX_RANK,
) -> EnemyThreatDict | None:
    """Pick the nearest map-fresh enemy that fails ONLY the affordability gate.

    The same human-first tiering as acquisition applies, so the relay
    chain hops toward a distant human even when a cheaper practice bot
    is also unaffordable.

    The dot-relay travel planner needs a destination worth travelling
    toward: an enemy that would be a perfectly viable acquisition if
    the bot had the fuel for the end-to-end fight. Every other gate
    (alive, synced, not blocked/killed, map-fresh, stand-off landable)
    must pass -- travelling toward a corpse or a blocked target wastes
    the relay.

    Args:
        world: Filtered world state (killed tanks already removed).
        self_state: Player's own state.
        blocked: Tank IDs temporarily un-engageable.
        killed: Tank IDs on kill cooldown.
        terrain: Terrain map for the stand-off landing check.
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
            ws,
            tank,
            self_state,
            blocked,
            killed,
            terrain,
            now_ms,
            map_open_cooldown_ms,
            engagement_reserve_fuel,
            human_min_rank,
            human_max_rank,
        )
        if rejected_reason != "unaffordable":
            continue
        candidates.append(
            make_enemy_threat_from_tank(
                tank,
                manhattan_distance(self_x, self_y, tank["x"], tank["y"]),
            )
        )

    candidates.sort(key=_threat_sort_key_for(priority_target_name, fleet_assist_ids(ws)))
    return candidates[0] if candidates else None


__all__ = [
    "find_acquisition_target",
    "find_relay_travel_target",
    "stale_human_exists",
]
