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


def analyze_threats(
    world: WorldStateDict,
    self_state: SelfStateDict,
) -> list[EnemyThreatDict]:
    """Analyze all enemy tanks and return sorted threat list.

    Filters to enemy tanks only (different team, not self), computes
    Manhattan distance from the player, and sorts by distance ascending
    (closest threats first).

    Args:
        world: Current world state with tank positions.
        self_state: Player's own state for position and team.

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
        # Skip dead tanks — deactivation sets position to (0, 0)
        if tank["x"] == 0 and tank["y"] == 0:
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
            )
        )

    threats.sort(key=_threat_sort_key)
    return threats


def _threat_sort_key(threat: EnemyThreatDict) -> tuple[int, int, int]:
    """Sort key: distance ascending, damage descending, freshness descending.

    Closer threats come first. Among threats at equal distance,
    more damaged enemies come first (easier to finish off). Among
    equal distance and damage, prefer recently confirmed tanks.

    Args:
        threat: Enemy threat to compute sort key for.

    Returns:
        Tuple of (distance, -damage_state, -timestamp_ms) for sorting.
    """
    return (threat["distance"], -threat["damage_state"], -threat["timestamp_ms"])


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
    "analyze_threats",
    "find_closest_threat",
    "manhattan_distance",
    "threats_in_range",
]
