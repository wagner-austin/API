"""Enemy threat analysis from world state.

Pure functions that convert raw world state tank data into sorted,
analyzed EnemyThreatDict lists for use by behavior evaluators.
"""

from __future__ import annotations

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


# A tank actually in the viewport talks on the wire constantly; a stale
# registry afterimage is silent. Raw-capture measurement 2026-06-11:
# purple-3 mid-fight produced 12 wire messages in 40s (~one per 3.3s);
# after leaving, 0 in 60s; the orange-6 afterimage that absorbed 52
# wasted shots produced 2 in 270s. Two fight-cadence periods of silence
# means the tank is no longer here (respawn cycles in ~10s, so nothing
# lingers longer for a good reason).
_WIRE_PRESENCE_TTL_MS = 7000

#: Public alias for cross-module consumers (combat_strategy, recover_fuel_mode).
WIRE_PRESENCE_TTL_MS = _WIRE_PRESENCE_TTL_MS

# Position-bearing messages (0x3D MovementResponse, 0x47 Movement, 0x28
# TankEntry, container TankUpdate*) arrive every ~2 s for tanks in view.
# Three TTL periods (~1.5 cycles + jitter) keeps the kill gate strict
# enough to filter out tanks whose position has gone stale -- e.g. a
# teleport whose new (x, y) has not yet hit the wire -- without
# rejecting position-bearing messages that slipped a tick.
_POSITION_FRESHNESS_TTL_MS = 3000

#: Public alias for cross-module consumers (combat_strategy).
POSITION_FRESHNESS_TTL_MS = _POSITION_FRESHNESS_TTL_MS


def is_wire_present(last_wire_seen_ms: int, now_ms: int) -> bool:
    """Return True when a tank's last wire timestamp is fresh.

    A tank actually in the viewport talks on the wire constantly; a
    stale registry afterimage is silent.  Two fight-cadence periods of
    silence means the tank is no longer here, so a
    ``last_wire_seen_ms`` older than :data:`_WIRE_PRESENCE_TTL_MS` is
    treated as absent. This gate guards ACQUISITION (HUNT candidate
    selection); the tighter :func:`is_position_fresh` gate guards the
    KILL SHOT.

    Args:
        last_wire_seen_ms: Timestamp of the tank's most recent wire message.
        now_ms: Current tick timestamp.

    Returns:
        True if the wire timestamp is within the presence TTL.
    """
    return now_ms - last_wire_seen_ms <= _WIRE_PRESENCE_TTL_MS


def is_position_fresh(last_position_update_ms: int, now_ms: int) -> bool:
    """Return True when a tank's position was wire-confirmed recently.

    Position-bearing wire messages (0x3D MovementResponse, 0x47
    Movement, 0x28 TankEntry, container TankUpdate*) refresh
    ``last_position_update_ms``; damage-only wire messages (0x2E
    TankStatusSync, container TankStatusShort) do NOT. This gate is
    the kill-shot guard: a tank whose presence is fresh but whose
    position has gone stale (the historical stale-registry combat
    bug) must NOT be fired at.

    Args:
        last_position_update_ms: Timestamp of the tank's most recent
            wire-sourced position-bearing message.
        now_ms: Current tick timestamp.

    Returns:
        True if the position timestamp is within the freshness TTL.
    """
    return now_ms - last_position_update_ms <= _POSITION_FRESHNESS_TTL_MS


def analyze_threats(
    world: WorldStateDict,
    self_state: SelfStateDict,
    now_ms: int,
) -> list[EnemyThreatDict]:
    """Analyze all enemy tanks and return sorted threat list.

    Filters to enemy tanks only (different team, not self) whose WIRE
    timestamp is fresh -- the wire vouches for presence, the registry
    only refines positions of wire-vouched tanks -- computes Manhattan
    distance from the player, and sorts by distance ascending.

    Args:
        world: Current world state with tank positions.
        self_state: Player's own state for position and team.
        now_ms: Current tick timestamp for wire-freshness filtering.

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
        if now_ms - tank["timestamp_ms"] > WIRE_PRESENCE_TTL_MS:
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


__all__ = [
    "POSITION_FRESHNESS_TTL_MS",
    "WIRE_PRESENCE_TTL_MS",
    "analyze_threats",
    "find_closest_threat",
    "is_position_fresh",
    "is_wire_present",
    "manhattan_distance",
    "threats_in_range",
]
