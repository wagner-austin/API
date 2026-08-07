"""Threat vocabulary: distances, freshness windows, and sort order.

The base layer every threat consumer reads -- the TTL constants that
define "fresh", the enemy predicate, the tank-to-threat projection,
the human-consent gate, and the finish-priority sort. Imports no
other threat module.
"""

from __future__ import annotations

from collections.abc import Callable

from tankpit_bot.bot.ai.humans import (
    threat_priority_tier,
)
from tankpit_bot.bot.ai.world_types import (
    EnemyThreatDict,
    make_enemy_threat,
)
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state.types import (
    TankStateDict,
    WorldStateDict,
)

# Legacy wire-presence horizon -- kept as a constant for test clock
# arithmetic (advance_clock past this value to age a target out of the
# viewport-confirmed threat list). The combat-side gate that used this
# was removed 2026-06-23: off-viewport pursuit shots fire toward the
# last known wire position via _locked_target_pursuit, so wire silence
# is no longer a stop signal.
_WIRE_PRESENCE_TTL_MS = 7000


#: Public alias for cross-module consumers (tests).
WIRE_PRESENCE_TTL_MS = _WIRE_PRESENCE_TTL_MS


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


# The server's homing trace on a departed target, measured live (run
# 194658, [[shoot-event-format]]#reroute-ttl-ms): pursuit homings hit
# to +12.0 s after the target left the viewport and missed at +14.0 s.
# Firing on the safe side of the wall saves the guaranteed-miss shot
# and its tick (flag s-latest-4: seven homings hit, the eighth always
# missed -- "couldnt we avoid the missed shot entirely?").
PURSUIT_TRACE_TTL_MS = 12_000


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


def pursuit_homing_budget_spent(
    world: WorldStateDict,
    locked_target_id: int,
    pursuit_shot_target_id: int,
    pursuit_shot_ms: int,
) -> bool:
    """Return True when this departure window's pursuit shot is spent.

    The human homing cap (user ruling 2026-07-31: firing all ~7
    reroute-tracked homings the 12 s wall allows "is cheating"): one
    pursuit shot per departure. The departure window is delimited by
    the registry's ``last_viewport_observation_ms`` — a pursuit shot
    stamped at or after the target's last in-viewport observation
    means the budget for THIS departure is used, and the target
    re-entering the viewport re-arms it with no explicit reset.

    Args:
        world: Filtered world state.
        locked_target_id: The locked target's tank id.
        pursuit_shot_target_id: ``ai_state["pursuit_shot_target_id"]``.
        pursuit_shot_ms: ``ai_state["pursuit_shot_ms"]``.

    Returns:
        True when a pursuit shot at this target was already dispatched
        since it was last seen in the viewport.
    """
    if pursuit_shot_target_id != locked_target_id:
        return False
    tank = world["tanks"].get(str(locked_target_id))
    if tank is None:
        return True
    return pursuit_shot_ms >= tank["last_viewport_observation_ms"]


__all__ = [
    "POSITION_FRESHNESS_TTL_MS",
    "PURSUIT_TRACE_TTL_MS",
    "WIRE_PRESENCE_TTL_MS",
    "find_closest_threat",
    "human_combat_consented",
    "make_enemy_threat_from_tank",
    "manhattan_distance",
    "pursuit_homing_budget_spent",
    "pursuit_trace_is_live",
    "threats_in_range",
]
