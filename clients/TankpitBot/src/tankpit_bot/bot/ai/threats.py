"""Enemy threat analysis from world state.

Pure functions that convert raw world state tank data into sorted,
analyzed EnemyThreatDict lists for use by behavior evaluators.
"""

from __future__ import annotations

from collections.abc import Callable

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.humans import (
    DEFAULT_HUMAN_MAX_RANK,
    DEFAULT_HUMAN_MIN_RANK,
    is_human_name,
    is_human_rank_protected,
    threat_priority_tier,
)
from tankpit_bot.bot.ai.types import EnemyThreatDict, make_enemy_threat
from tankpit_bot.sniffer.world_state import get_world_service
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


def make_enemy_threat_from_tank(tank: TankStateDict, distance: int) -> EnemyThreatDict:
    """Build an enemy-threat record from a registry tank.

    The single construction path from registry truth to a threat --
    used by map acquisition, the relay travel scan, and the greeting
    approach, so a new registry field lands in every consumer at once.

    Args:
        tank: Registry tank record.
        distance: Precomputed Manhattan distance from self.

    Returns:
        Threat record mirroring the tank's registry fields.
    """
    return make_enemy_threat(
        tank_id=tank["tank_id"],
        x=tank["x"],
        y=tank["y"],
        distance=distance,
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


def human_combat_consented(tank_id: int) -> bool:
    """Return True when a human target has consented to combat.

    User ruling 2026-07-30 (session 8 killed over it: "i felt bad, we
    were gonna kill a defenseless human there. to engage in combat.
    the human must respond hello or engage the bot first"). Consent is
    either signal the wire can prove:

    * they CHATTED this session -- any non-self-echo 0x4D from their
      id (the HELLO response, or anything else they say), or
    * they SHOT US -- the damage book's ``taken`` side holds a row
      for their id.

    Practice bots never pass through this predicate; callers gate it
    behind :func:`~tankpit_bot.bot.ai.humans.is_human_name`.

    Args:
        tank_id: The human tank's id.

    Returns:
        True when the human has responded or struck first.
    """
    service = get_world_service()
    if tank_id in service.chat_seen_tank_ids:
        return True
    return str(tank_id) in service.damage_book["taken"]


def analyze_threats(
    world: WorldStateDict,
    self_state: SelfStateDict,
    now_ms: int,
    *,
    human_min_rank: int = DEFAULT_HUMAN_MIN_RANK,
    human_max_rank: int = DEFAULT_HUMAN_MAX_RANK,
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
        # Human rank-window protection (user ruling 2026-07-28):
        # humans outside [human_min_rank, human_max_rank] never enter
        # the threat list, so they can be neither locked nor fired at.
        # Practice bots are farmed at any rank.
        if is_human_rank_protected(
            tank["name"],
            tank["rank"],
            min_rank=human_min_rank,
            max_rank=human_max_rank,
        ):
            continue
        # Human-consent contract (2026-07-30): a human who has neither
        # chatted nor shot us never enters the threat list -- no lock,
        # no fire. An attacker consents by attacking (their id lands
        # in the damage book), so defense is never blocked.
        if is_human_name(tank["name"]) and not human_combat_consented(tank["tank_id"]):
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

    The tier is the tank's FUEL QUARTILE (corpus-fitted 2026-07-23,
    19,658 sync samples, zero exceptions — [[deactivation-format]]):
    tier 0 is the bottom quartile (near death), tier 3 the top
    (healthy). The June "counts down 0 -> 3 -> 2 -> 1" reading was an
    artifact of watching fresh tanks (briefly unsynced at 0) drain
    through 3 -> 2 -> 1; victims "died from tier 1" because the
    killing hit took fuel below zero before a tier-0 sync could ride
    the wire. Unknown tanks default to ``DAMAGE_FULL`` (= 3, assume
    healthy), so the ascending tier IS the finish-off order.

    Args:
        damage_state: Wire damage tier (0 = near death .. 3 = full).

    Returns:
        Ascending rank where the most damaged enemy ranks first.
    """
    return damage_state


def _threat_sort_key(threat: EnemyThreatDict) -> tuple[int, int, int, int]:
    """Sort key: human tier, then distance, finish-off priority, freshness.

    Human-classified enemies outrank every practice bot regardless of
    distance (user doctrine 2026-07-28: farm bots normally, but any
    human who logs in becomes the priority). Within a tier, closer
    threats come first; among equal distance, more damaged enemies
    come first (easier to finish off); then recently confirmed tanks.

    Args:
        threat: Enemy threat to compute sort key for.

    Returns:
        Tuple of (human_tier, distance, finish_priority, -timestamp_ms).
    """
    return _threat_sort_key_for("")(threat)


def _threat_sort_key_for(
    priority_target_name: str,
) -> Callable[[EnemyThreatDict], tuple[int, int, int, int]]:
    """Build the threat sort key with a configured priority account.

    Args:
        priority_target_name: Account name that outranks even other
            humans (case-insensitive), or ``""`` for none.

    Returns:
        Sort-key callable ordering by (tier, distance, finish, age).
    """

    def _key(threat: EnemyThreatDict) -> tuple[int, int, int, int]:
        return (
            threat_priority_tier(threat["name"], priority_target_name),
            threat["distance"],
            _finish_priority(threat["damage_state"]),
            -threat["timestamp_ms"],
        )

    return _key


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


# The server's homing trace on a departed target, measured live (run
# 194658, [[shoot-event-format]]#reroute-ttl-ms): pursuit homings hit
# to +12.0 s after the target left the viewport and missed at +14.0 s.
# Firing on the safe side of the wall saves the guaranteed-miss shot
# and its tick (flag s-latest-4: seven homings hit, the eighth always
# missed -- "couldnt we avoid the missed shot entirely?").
PURSUIT_TRACE_TTL_MS = 12_000


def pursuit_trace_is_live(
    world: WorldStateDict,
    locked_target_id: int,
    now_ms: int,
) -> bool:
    """Return True while pursuit homings at the departed target can hit.

    Args:
        world: Filtered world state.
        locked_target_id: The locked target's tank id.
        now_ms: Current tick timestamp.

    Returns:
        True when the target's last in-viewport observation is inside
        the homing-trace window; False once a fired homing would
        resolve after the wall (a booked miss and a wasted tick).
    """
    tank = world["tanks"].get(str(locked_target_id))
    if tank is None:
        return False
    return now_ms - tank["last_viewport_observation_ms"] <= PURSUIT_TRACE_TTL_MS


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
    from tankpit_bot.bot.ai.combat_strategy import has_standoff_landing
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
    if is_human_name(tank["name"]) and not human_combat_consented(tank["tank_id"]):
        # Human-consent contract (2026-07-30): no acquisition of a
        # human who has neither responded to the HELLO nor engaged
        # first. Placed before the affordability gate for the same
        # relay-path reason as the rank window above.
        return "human_not_consented"
    if tank["x"] == 0 and tank["y"] == 0:
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

    candidates.sort(key=_threat_sort_key_for(priority_target_name))
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

    candidates.sort(key=_threat_sort_key_for(priority_target_name))
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
    "stale_human_exists",
    "threats_in_range",
]
