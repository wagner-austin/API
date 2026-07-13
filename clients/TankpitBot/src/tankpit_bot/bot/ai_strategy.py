"""Top-level durable AI owner selection.

This module owns only the orchestration layer that selects exactly one durable
mode owner per tick and rewrites the returned AI state with the matching
mode/substate fields.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.combat_strategy import has_cardinal_combat_shot
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.ferry import compose_decision_terrain
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.mode_controller import (
    apply_dispatch_counters,
    apply_mode_to_decision,
    clear_ai_mode,
    derive_collect_mode_state,
    derive_hunt_mode_state,
    make_hold_decision,
    resolve_owner_from_manual,
    should_enter_collect,
    should_exit_collect,
    should_exit_hunt,
)
from tankpit_bot.bot.ai.modes import AIMode, AIModeState, is_valid_ai_mode_state
from tankpit_bot.bot.ai.threats import analyze_threats
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
    manual = resolve_owner_from_manual(normalized_state)
    if manual == "UNSET":
        emit_ai("manual mode UNSET: holding position")
        return make_hold_decision(normalized_state, timestamp_ms)
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
    mode = _resolve_owner_mode(ctx, manual)
    if mode == "COLLECT":
        collect_decision = decide_collect_mode(ctx)
        if collect_decision is not None:
            owned = apply_mode_to_decision(
                collect_decision,
                "COLLECT",
                derive_collect_mode_state(collect_decision),
                timestamp_ms,
            )
            return apply_dispatch_counters(owned)
        # Collection exhausted with healthy fuel: the tank is stocked,
        # so this tick belongs to the hunt owner (fall through).
        emit_ai("collect owner yielded, handing tick to hunt owner")
    decision = decide_hunt_mode(ctx)
    owned = apply_mode_to_decision(
        decision,
        "HUNT",
        derive_hunt_mode_state(decision),
        timestamp_ms,
    )
    return apply_dispatch_counters(owned)


def _resolve_owner_mode(ctx: DecideCtx, manual: AIMode | None) -> AIMode:
    """Select the durable owner for this tick, honouring the manual pin.

    Args:
        ctx: Decision context.
        manual: SPA-pinned mode override, or ``None`` when the
            arbitrator should run auto-selection. ``UNSET`` is handled
            upstream in :func:`decide` and never reaches this function.

    Returns:
        The mode owner for this tick. Manual ``HUNT`` and ``COLLECT``
        skip auto-arbitration; ``None`` delegates to
        :func:`_select_owner_mode`.
    """
    if manual == "HUNT":
        return "HUNT"
    if manual == "COLLECT":
        return "COLLECT"
    return _select_owner_mode(ctx)


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
        Durable top-level owner for the current tick. A cardinal-adjacent
        live wire-fresh enemy overrides every other selector: HUNT wins
        the tick regardless of the current durable mode, inventory
        reserves, or COLLECT candidates. A cardinal shot is a free kill
        and skipping it for a fuel pickup was the 2026-07-06 22:37 loop
        (bot teleported adjacent to orange-8 at (46,159) five times,
        each time dispatching a fuel pickup instead of firing).
    """
    if _has_any_cardinal_combat_shot(ctx):
        return "HUNT"
    current_mode = ctx.mode
    if current_mode == "COLLECT" and not should_exit_collect(ctx):
        return "COLLECT"
    if current_mode == "HUNT" and not should_exit_hunt(ctx):
        return "HUNT"
    if should_enter_collect(ctx):
        return "COLLECT"
    return "HUNT"


def _has_any_cardinal_combat_shot(ctx: DecideCtx) -> bool:
    """Return True when a live wire-fresh enemy sits cardinally adjacent.

    Cardinal adjacency (Manhattan distance 1) guarantees a hit at
    point-blank: aim is viewport-legal by construction, the server
    picks the enabled weapon, and the shot lands the same tick. The
    check runs over :func:`analyze_threats`, which already gates for
    liveness and viewport-freshness -- so ``True`` means "there is at
    this instant an enemy the bot could shoot for free."

    Args:
        ctx: Decision context.

    Returns:
        True if any live viewport-fresh enemy is at Manhattan distance
        1 from the bot.
    """
    threats = analyze_threats(ctx.filtered, ctx.self_state, ctx.timestamp_ms)
    return any(has_cardinal_combat_shot(ctx.self_state, threat) for threat in threats)


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
