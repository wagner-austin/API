"""AI behavior system for autonomous tank control.

This package provides a modular AI decision system built on pure functions
and immutable TypedDicts. The architecture has two layers:

- **Decision layer**: Evaluators score candidate behaviors based on world state.
- **Execution layer**: Actions translate chosen behaviors into Bot commands.

Submodules:
- types: Core TypedDicts (AIConfigDict, AIStateDict, BehaviorScoreDict, etc.)
- threats: Enemy analysis from world state
- pathfinding: Terrain-aware A* pathfinding
- equipment: Equipment management decisions
- evaluators: Behavior scoring functions
- actions: Behavior execution (Bot command dispatch)
- tactics: Tactical decisions (radar, teleport search, equipment state)
- loop: Main AI tick orchestrator
"""

from __future__ import annotations

from tankpit_bot.bot.ai.actions import execute_behavior
from tankpit_bot.bot.ai.equipment import (
    find_best_fuel,
    find_nearest_deposit,
    find_nearest_equipment,
    find_nearest_fuel,
    is_reachable,
)
from tankpit_bot.bot.ai.evaluators import (
    score_collect_equipment,
    score_collect_fuel,
    score_defend,
    score_deposit_fuel,
    score_hunt,
    score_patrol,
    select_best_behavior,
)
from tankpit_bot.bot.ai.loop import ai_tick
from tankpit_bot.bot.ai.pathfinding import find_path, path_length
from tankpit_bot.bot.ai.tactics import (
    compute_desired_equipment,
    find_teleport_target,
    should_proactive_radar,
    should_teleport_search,
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
    make_behavior_score,
    make_default_ai_config,
    make_enemy_threat,
    make_initial_ai_state,
    make_path_step,
)

__all__ = [
    "BEHAVIOR_MODES",
    "AIConfigDict",
    "AIStateDict",
    "BehaviorMode",
    "BehaviorScoreDict",
    "EnemyThreatDict",
    "PathStepDict",
    "ai_tick",
    "analyze_threats",
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
    "execute_behavior",
    "find_best_fuel",
    "find_closest_threat",
    "find_nearest_deposit",
    "find_nearest_equipment",
    "find_nearest_fuel",
    "find_path",
    "find_teleport_target",
    "is_reachable",
    "make_behavior_score",
    "make_default_ai_config",
    "make_enemy_threat",
    "make_initial_ai_state",
    "make_path_step",
    "manhattan_distance",
    "path_length",
    "score_collect_equipment",
    "score_collect_fuel",
    "score_defend",
    "score_deposit_fuel",
    "score_hunt",
    "score_patrol",
    "select_best_behavior",
    "should_proactive_radar",
    "should_teleport_search",
    "threats_in_range",
]
