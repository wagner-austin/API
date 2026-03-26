"""Main AI tick orchestrator.

The ai_tick function is the single entry point called each game tick.
It evaluates all behaviors, selects the best one, executes it, and
returns the updated AI state plus the bot command to send.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.actions import execute_behavior
from tankpit_bot.bot.ai.evaluators import select_best_behavior
from tankpit_bot.bot.ai.types import AIStateDict, BehaviorScoreDict
from tankpit_bot.bot.types import BotCommand
from tankpit_bot.state.types import SelfStateDict, WorldStateDict


def ai_tick(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    timestamp_ms: int,
    terrain: TerrainMapProtocol | None = None,
) -> tuple[AIStateDict, BotCommand, BehaviorScoreDict]:
    """Run one AI decision tick.

    Evaluates all behavior scorers, picks the highest-scoring behavior,
    executes it, and returns updated state, command, and chosen behavior.

    Args:
        world: Current world state.
        self_state: Player's own state.
        ai_state: Current AI state.
        timestamp_ms: Current game timestamp in milliseconds.
        terrain: Optional terrain map for reachability checks.

    Returns:
        Tuple of (updated AIStateDict, BotCommand to send, chosen BehaviorScoreDict).
    """
    behavior = select_best_behavior(world, self_state, ai_state, terrain)
    new_state, command = execute_behavior(behavior, ai_state, self_state, timestamp_ms, terrain)
    return new_state, command, behavior


__all__ = [
    "ai_tick",
]
