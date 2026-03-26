"""Behavior scoring evaluators for the AI system.

Hard priority chain — not a scoring competition:
1. COLLECT_FUEL — fuel low and containers visible
2. COLLECT_EQUIPMENT — equipment low and containers visible
3. HUNT — find and kill enemies
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.equipment import (
    find_best_fuel,
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
from tankpit_bot.sniffer.world_state import get_inventory_state
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

    if self_state["fuel"] <= config["hunt_min_fuel"]:
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
        target_id=target["tank_id"],
    )


def score_collect_fuel(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    terrain: TerrainMapProtocol | None = None,
) -> BehaviorScoreDict:
    """Score the COLLECT_FUEL behavior.

    Score increases dramatically when fuel is low. Highest priority
    when fuel drops below fuel_low_threshold.

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

    # Critical: below critical threshold → emergency priority
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
    equipment is nearby and fuel is adequate.

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

    # Don't collect when inventory is full (rank capacity: 20 + rank * 5)
    inventory = get_inventory_state()
    rank_capacity = 20 + self_state["rank"] * 5
    all_full = (
        inventory["armor_shields"]["count"] >= rank_capacity
        and inventory["dual_shots"]["count"] >= rank_capacity
        and inventory["missile_shots"]["count"] >= rank_capacity
        and inventory["homing_shots"]["count"] >= rank_capacity
        and inventory["extra_radars"]["count"] >= rank_capacity
    )
    if all_full:
        return make_behavior_score("COLLECT_EQUIPMENT", 0, 0, 0, "inventory full")

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


def select_best_behavior(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    terrain: TerrainMapProtocol | None = None,
) -> BehaviorScoreDict:
    """Hard priority chain: fuel > equipment > hunt.

    Not a scoring competition. Fuel collection wins unconditionally
    when fuel is low and containers are visible.

    Args:
        world: Current world state.
        self_state: Player's own state.
        ai_state: Current AI state with config.
        terrain: Optional terrain map for reachability checks.

    Returns:
        BehaviorScoreDict for the chosen behavior.
    """
    config = ai_state["config"]
    fuel = self_state["fuel"]

    # Priority 1: COLLECT_FUEL when fuel is low and containers visible
    if fuel <= config["fuel_low_threshold"]:
        fuel_score = score_collect_fuel(world, self_state, ai_state, terrain)
        if fuel_score["score"] > 0:
            return fuel_score

    # Priority 2: COLLECT_EQUIPMENT when nearby and inventory has room
    equip_score = score_collect_equipment(world, self_state, ai_state, terrain)
    if equip_score["score"] > 0:
        return equip_score

    # Priority 3: HUNT enemies
    threats = analyze_threats(world, self_state)
    hunt_score = score_hunt(world, self_state, ai_state, threats)
    if hunt_score["score"] > 0:
        return hunt_score

    # Nothing to do — return zero-score HUNT as fallback
    return make_behavior_score("HUNT", 0, 0, 0, "nothing to do")


__all__ = [
    "score_collect_equipment",
    "score_collect_fuel",
    "score_hunt",
    "select_best_behavior",
]
