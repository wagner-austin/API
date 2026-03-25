"""Behavior scoring evaluators for the AI system.

Each evaluator is a pure function that takes world state and AI config,
returning a BehaviorScoreDict. The select_best_behavior function picks
the highest-scoring behavior each tick.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.equipment import (
    find_best_fuel,
    find_nearest_deposit,
    find_nearest_equipment,
    find_nearest_fuel,
)
from tankpit_bot.bot.ai.threats import analyze_threats, threats_in_range
from tankpit_bot.bot.ai.types import (
    AIStateDict,
    BehaviorScoreDict,
    EnemyThreatDict,
    make_behavior_score,
)
from tankpit_bot.state.types import SelfStateDict, WorldStateDict

# Rank bonus: lower-rank enemies are easier kills. Max rank is 7.
# Bonus = (7 - rank) * 15, so rank 0 = +105, rank 7 = 0.
_RANK_BONUS_PER_LEVEL = 15
_MAX_RANK = 7


def score_hunt(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    threats: list[EnemyThreatDict],
) -> BehaviorScoreDict:
    """Score the HUNT behavior.

    HUNT activates when there are enemies in range and fuel is sufficient.
    Score increases with enemy proximity, enemy damage, and lower enemy rank.

    Args:
        world: Current world state.
        self_state: Player's own state.
        ai_state: Current AI state with config.
        threats: Pre-computed sorted threat list.

    Returns:
        BehaviorScoreDict for HUNT behavior.
    """
    config = ai_state["config"]

    if self_state["fuel"] < config["hunt_min_fuel"]:
        return make_behavior_score("HUNT", 0, 0, 0, "fuel too low")

    in_range = threats_in_range(threats, config["combat_range"])
    if not in_range:
        return make_behavior_score("HUNT", 0, 0, 0, "no enemies in range")

    target = in_range[0]
    # Base score 700 + bonus for damaged enemies (up to +200 for critical)
    damage_bonus = target["damage_state"] * 65
    # Proximity bonus: closer = higher score (max +100)
    proximity_bonus = max(0, 100 - target["distance"] * 5)
    # Rank bonus: lower rank = weaker = easier kill (max +105)
    rank_bonus = (_MAX_RANK - target["rank"]) * _RANK_BONUS_PER_LEVEL
    score = min(1000, 700 + damage_bonus + proximity_bonus + rank_bonus)

    return make_behavior_score(
        "HUNT",
        score,
        target["x"],
        target["y"],
        f"enemy {target['name']} dist={target['distance']}"
        f" dmg={target['damage_state']} rank={target['rank']}",
    )


def score_collect_fuel(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    terrain: TerrainMapProtocol | None = None,
) -> BehaviorScoreDict:
    """Score the COLLECT_FUEL behavior.

    Score increases dramatically when fuel is low. Highest priority
    when fuel drops below fuel_low_threshold. When terrain is available,
    only considers reachable fuel containers.

    Args:
        world: Current world state.
        self_state: Player's own state.
        ai_state: Current AI state with config.
        terrain: Optional terrain map for reachability checks.

    Returns:
        BehaviorScoreDict for COLLECT_FUEL behavior.
    """
    config = ai_state["config"]
    fuel = self_state["fuel"]

    # Critical: below critical threshold → emergency priority, enable shields
    if fuel < config["fuel_critical_threshold"]:
        target = find_best_fuel(world, self_state, terrain)
        if target is None:
            return make_behavior_score("COLLECT_FUEL", 0, 0, 0, "no fuel visible")
        return make_behavior_score(
            "COLLECT_FUEL",
            950,
            target["x"],
            target["y"],
            f"fuel critical ({fuel}) vol={target['volume']}",
        )

    # Low: below low threshold → high priority, prefer volume
    if fuel < config["fuel_low_threshold"]:
        target = find_best_fuel(world, self_state, terrain)
        if target is None:
            return make_behavior_score("COLLECT_FUEL", 0, 0, 0, "no fuel visible")
        fuel_ratio = (config["fuel_low_threshold"] - fuel) / (
            config["fuel_low_threshold"] - config["fuel_critical_threshold"]
        )
        score = int(700 + fuel_ratio * 200)
        return make_behavior_score(
            "COLLECT_FUEL",
            score,
            target["x"],
            target["y"],
            f"fuel low ({fuel}) vol={target['volume']}",
        )

    nearest = find_nearest_fuel(world, self_state, terrain)
    if nearest is None:
        return make_behavior_score("COLLECT_FUEL", 0, 0, 0, "no fuel visible")

    if fuel >= config["fuel_full_threshold"]:
        return make_behavior_score("COLLECT_FUEL", 0, nearest["x"], nearest["y"], "fuel full")

    # Normal: linear scale (200-700 range)
    fuel_ratio = (config["fuel_full_threshold"] - fuel) / (
        config["fuel_full_threshold"] - config["fuel_low_threshold"]
    )
    score = int(200 + fuel_ratio * 500)
    reason = f"fuel={fuel} ratio={fuel_ratio:.2f}"

    return make_behavior_score("COLLECT_FUEL", score, nearest["x"], nearest["y"], reason)


def score_collect_equipment(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    terrain: TerrainMapProtocol | None = None,
) -> BehaviorScoreDict:
    """Score the COLLECT_EQUIPMENT behavior.

    Equipment collection is opportunistic — moderate priority when
    equipment is nearby and fuel is adequate. When terrain is available,
    only considers reachable equipment containers.

    Args:
        world: Current world state.
        self_state: Player's own state.
        ai_state: Current AI state with config.
        terrain: Optional terrain map for reachability checks.

    Returns:
        BehaviorScoreDict for COLLECT_EQUIPMENT behavior.
    """
    config = ai_state["config"]

    nearest = find_nearest_equipment(world, self_state, terrain)
    if nearest is None:
        return make_behavior_score("COLLECT_EQUIPMENT", 0, 0, 0, "no equipment visible")

    # Don't collect equipment when fuel is critical
    if self_state["fuel"] < config["fuel_critical_threshold"]:
        return make_behavior_score(
            "COLLECT_EQUIPMENT",
            0,
            nearest["x"],
            nearest["y"],
            "fuel too low",
        )

    # Base 400, bonus for proximity (closer = higher)
    from tankpit_bot.bot.ai.threats import manhattan_distance

    dist = manhattan_distance(self_state["x"], self_state["y"], nearest["x"], nearest["y"])
    proximity_bonus = max(0, 150 - dist * 5)
    score = min(600, 400 + proximity_bonus)

    return make_behavior_score(
        "COLLECT_EQUIPMENT",
        score,
        nearest["x"],
        nearest["y"],
        f"equipment dist={dist}",
    )


def score_deposit_fuel(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    terrain: TerrainMapProtocol | None = None,
) -> BehaviorScoreDict:
    """Score the DEPOSIT_FUEL behavior.

    Deposit becomes eligible when fuel exceeds fuel_full_threshold.
    Higher fuel surplus → higher priority. When terrain is available,
    only considers reachable deposit targets.

    Args:
        world: Current world state.
        self_state: Player's own state.
        ai_state: Current AI state with config.
        terrain: Optional terrain map for reachability checks.

    Returns:
        BehaviorScoreDict for DEPOSIT_FUEL behavior.
    """
    config = ai_state["config"]
    fuel = self_state["fuel"]

    if fuel < config["fuel_full_threshold"]:
        return make_behavior_score("DEPOSIT_FUEL", 0, 0, 0, "not enough fuel")

    nearest = find_nearest_deposit(world, self_state, terrain)
    if nearest is None:
        return make_behavior_score("DEPOSIT_FUEL", 0, 0, 0, "no deposit target")

    # Score 600-800 based on fuel surplus
    surplus = fuel - config["fuel_full_threshold"]
    score = min(800, 600 + surplus // 2)

    return make_behavior_score(
        "DEPOSIT_FUEL",
        score,
        nearest["x"],
        nearest["y"],
        f"fuel={fuel} surplus={surplus}",
    )


def score_patrol(
    ai_state: AIStateDict,
) -> BehaviorScoreDict:
    """Score the PATROL behavior.

    Patrol is the default fallback behavior with low constant score.
    Follows the waypoint circuit defined in config.

    Args:
        ai_state: Current AI state with config and waypoint index.

    Returns:
        BehaviorScoreDict for PATROL behavior.
    """
    config = ai_state["config"]
    waypoints = config["patrol_waypoints"]
    idx = ai_state["patrol_waypoint_index"] % len(waypoints)
    wx, wy = waypoints[idx]

    return make_behavior_score("PATROL", 100, wx, wy, f"waypoint {idx}")


def score_defend(
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    threats: list[EnemyThreatDict],
) -> BehaviorScoreDict:
    """Score the DEFEND behavior.

    Defend activates when enemies are very close and fuel is too low
    to hunt. It's a reactive survival behavior.

    Args:
        self_state: Player's own state.
        ai_state: Current AI state with config.
        threats: Pre-computed sorted threat list.

    Returns:
        BehaviorScoreDict for DEFEND behavior.
    """
    config = ai_state["config"]

    # Only defend if enemies are within half combat range
    close_range = config["combat_range"] // 2
    close_threats = threats_in_range(threats, close_range)
    if not close_threats:
        return make_behavior_score("DEFEND", 0, 0, 0, "no close threats")

    target = close_threats[0]

    # Defend is high priority when fuel is low but enemies are close
    if self_state["fuel"] < config["hunt_min_fuel"]:
        score = 850
        reason = f"low fuel defense vs {target['name']}"
    else:
        # Moderate priority defense when under attack
        score = 500
        reason = f"defensive vs {target['name']} dist={target['distance']}"

    return make_behavior_score("DEFEND", score, target["x"], target["y"], reason)


def select_best_behavior(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    terrain: TerrainMapProtocol | None = None,
) -> BehaviorScoreDict:
    """Run all evaluators and select the highest-scoring behavior.

    Args:
        world: Current world state.
        self_state: Player's own state.
        ai_state: Current AI state with config.
        terrain: Optional terrain map for reachability checks.

    Returns:
        BehaviorScoreDict with the highest score.
    """
    threats = analyze_threats(world, self_state)

    scores: list[BehaviorScoreDict] = [
        score_hunt(world, self_state, ai_state, threats),
        score_collect_fuel(world, self_state, ai_state, terrain),
        score_collect_equipment(world, self_state, ai_state, terrain),
        score_deposit_fuel(world, self_state, ai_state, terrain),
        score_patrol(ai_state),
        score_defend(self_state, ai_state, threats),
    ]

    best = scores[0]
    for candidate in scores[1:]:
        if candidate["score"] > best["score"]:
            best = candidate

    return best


__all__ = [
    "score_collect_equipment",
    "score_collect_fuel",
    "score_defend",
    "score_deposit_fuel",
    "score_hunt",
    "score_patrol",
    "select_best_behavior",
]
