"""Top-level durable AI owner selection.

This module owns only the orchestration layer that selects exactly one durable
mode owner per tick and rewrites the returned AI state with the matching
mode/substate fields.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.ferry import compose_decision_terrain
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.mode_controller import (
    apply_mode_to_decision,
    clear_ai_mode,
    derive_collect_mode_state,
    derive_hunt_mode_state,
    should_enter_collect,
    should_exit_collect,
    should_exit_hunt,
)
from tankpit_bot.bot.ai.modes import AIMode, AIModeState, is_valid_ai_mode_state
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.inventory import InventoryState
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.types import SelfStateDict, WorldStateDict


def decide(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    inventory: InventoryState,
    timestamp_ms: int,
    terrain: TerrainMapProtocol | None,
    combat_feedback: CombatFeedback = "",
    map_fuel_dots: tuple[tuple[int, int], ...] = (),
) -> TickDecisionDict:
    """Run one AI decision cycle under durable owner routing.

    Args:
        world: Current world state with tanks, containers, and mines.
        self_state: Player state for this tick.
        ai_state: Current durable AI state.
        inventory: Current inventory state.
        timestamp_ms: Current game timestamp in milliseconds.
        terrain: Optional terrain map for reachability and landing checks.
        combat_feedback: Protocol-level hit or miss feedback for the last shot.
        map_fuel_dots: 0x4C fuel-dot atlas positions (empty before the
            first map open of the session).

    Returns:
        Tick decision produced by the selected durable owner.
    """
    normalized_state = _normalize_ai_state(ai_state)
    ctx = DecideCtx(
        world,
        self_state,
        normalized_state,
        inventory,
        timestamp_ms,
        compose_decision_terrain(world, terrain),
        combat_feedback,
        map_fuel_dots,
    )
    mode = _select_owner_mode(ctx)
    if mode == "COLLECT":
        collect_decision = decide_collect_mode(ctx)
        if collect_decision is not None:
            return apply_mode_to_decision(
                collect_decision,
                "COLLECT",
                derive_collect_mode_state(collect_decision),
                timestamp_ms,
            )
        # Collection exhausted with healthy fuel: the tank is stocked,
        # so this tick belongs to the hunt owner (fall through).
        emit_ai("collect owner yielded, handing tick to hunt owner")
    decision = decide_hunt_mode(ctx)
    return apply_mode_to_decision(
        decision,
        "HUNT",
        derive_hunt_mode_state(decision),
        timestamp_ms,
    )


def _normalize_ai_state(ai_state: AIStateDict) -> AIStateDict:
    """Return AI state with invalid durable ownership cleared.

    Args:
        ai_state: Current AI state.

    Returns:
        Original AI state when the durable mode pair is valid, otherwise the
        same state with durable ownership reset to ``UNSET``.
    """
    if is_valid_ai_mode_state(ai_state["mode"], ai_state["mode_state"]):
        if ai_state["mode"] == "UNSET" and ai_state["combat_target_id"] != -1:
            return _migrate_unset_combat_state(ai_state)
        return ai_state
    if ai_state["combat_target_id"] != -1:
        return _migrate_unset_combat_state(ai_state)
    return clear_ai_mode(ai_state)


def _select_owner_mode(ctx: DecideCtx) -> AIMode:
    """Select the durable owner for this tick.

    Args:
        ctx: Decision context.

    Returns:
        Durable top-level owner for the current tick.
    """
    current_mode = ctx.mode
    if current_mode == "COLLECT" and not should_exit_collect(ctx):
        return "COLLECT"
    if current_mode == "HUNT" and not should_exit_hunt(ctx):
        return "HUNT"
    if should_enter_collect(ctx):
        return "COLLECT"
    return "HUNT"


def _migrate_unset_combat_state(ai_state: AIStateDict) -> AIStateDict:
    """Return unset AI state migrated into durable HUNT ownership.

    Args:
        ai_state: Current AI state with a locked combat target.

    Returns:
        AI state migrated into durable HUNT ownership.
    """
    migrated_mode_state: AIModeState = "REFRESH"
    if ai_state["last_shot_target_id"] == ai_state["combat_target_id"]:
        migrated_mode_state = "ENGAGE"
    return AIStateDict(
        **{
            **ai_state,
            "mode": "HUNT",
            "mode_state": migrated_mode_state,
            "mode_started_ms": 0,
        }
    )


__all__ = ["decide"]
