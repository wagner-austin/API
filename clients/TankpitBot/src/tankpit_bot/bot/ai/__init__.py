"""Shared AI support modules for the canonical tick-loop strategy.

This package exposes reusable planning primitives such as pathfinding,
threat analysis, equipment selection, and strongly typed AI data shapes.
The live planner is ``tankpit_bot.bot.ai_strategy.decide``.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_landing import (
    choose_combat_landing_tile,
    combat_landing_candidates,
    has_cardinal_enemy_adjacency,
)
from tankpit_bot.bot.ai.equipment_search import (
    find_best_fuel,
    find_nearest_deposit,
    find_nearest_equipment,
    find_nearest_fuel,
    is_reachable,
)
from tankpit_bot.bot.ai.pathfinding import find_path, path_length
from tankpit_bot.bot.ai.tactics import (
    compute_desired_equipment,
    should_proactive_radar,
)
from tankpit_bot.bot.ai.threats import (
    analyze_threats,
    find_closest_threat,
    manhattan_distance,
    threats_in_range,
)
from tankpit_bot.bot.ai.types import (
    BEHAVIOR_MODES,
    AIConfigDict,
    AIStateDict,
    BehaviorMode,
    BehaviorScoreDict,
    EnemyThreatDict,
    PathStepDict,
    make_behavior_score,
    make_default_ai_config,
    make_enemy_threat,
    make_initial_ai_state,
    make_path_step,
)
from tankpit_bot.bot.ai.types_codecs import (
    decode_ai_config,
    decode_ai_state,
    decode_behavior_score,
    decode_enemy_threat,
    decode_path_step,
    encode_ai_config,
    encode_ai_state,
    encode_behavior_score,
    encode_enemy_threat,
    encode_path_step,
)

__all__ = [
    "BEHAVIOR_MODES",
    "AIConfigDict",
    "AIStateDict",
    "BehaviorMode",
    "BehaviorScoreDict",
    "EnemyThreatDict",
    "PathStepDict",
    "analyze_threats",
    "choose_combat_landing_tile",
    "combat_landing_candidates",
    "compute_desired_equipment",
    "decode_ai_config",
    "decode_ai_state",
    "decode_behavior_score",
    "decode_enemy_threat",
    "decode_path_step",
    "encode_ai_config",
    "encode_ai_state",
    "encode_behavior_score",
    "encode_enemy_threat",
    "encode_path_step",
    "find_best_fuel",
    "find_closest_threat",
    "find_nearest_deposit",
    "find_nearest_equipment",
    "find_nearest_fuel",
    "find_path",
    "has_cardinal_enemy_adjacency",
    "is_reachable",
    "make_behavior_score",
    "make_default_ai_config",
    "make_enemy_threat",
    "make_initial_ai_state",
    "make_path_step",
    "manhattan_distance",
    "path_length",
    "should_proactive_radar",
    "threats_in_range",
]
