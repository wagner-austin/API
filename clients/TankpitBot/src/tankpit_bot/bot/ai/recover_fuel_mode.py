"""Durable fuel-recovery owner and shared fuel recovery helpers."""

from __future__ import annotations

from typing import Literal, overload

from tankpit_bot.bot.ai.combat_strategy import clear_combat_target
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_use_radar,
    clear_resource_target,
    locked_resource_target,
    make_decision,
    set_resource_target,
    should_scan_resources_in_current_viewport,
)
from tankpit_bot.bot.ai.equipment import (
    describe_container_search,
    find_best_fuel,
    find_known_fuel_candidates,
)
from tankpit_bot.bot.ai.movement import select_exploration_command, walk_or_teleport
from tankpit_bot.bot.ai.resource_search import make_resource_search_hop
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import BotCommand, make_radar_command
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.types import ContainerStateDict

_RADAR_FUEL_COST = 10


def minimum_recovery_fuel_volume(ctx: DecideCtx) -> int:
    """Return the minimum fuel volume worth collecting for recovery.

    Args:
        ctx: Decision context.

    Returns:
        Minimum actionable fuel volume for the current recovery state.
    """
    if ctx.fuel <= ctx.config["fuel_critical_threshold"]:
        return 1
    return 100


def can_use_fuel_radar(ctx: DecideCtx) -> bool:
    """Return True when a radar scan is legal for fuel recovery.

    Args:
        ctx: Decision context.

    Returns:
        True when the bot can afford a radar scan this tick.
    """
    return can_use_radar(ctx) and ctx.fuel >= _RADAR_FUEL_COST


def select_fuel_target(
    ctx: DecideCtx,
    *,
    allow_unreachable: bool,
) -> tuple[ContainerStateDict, BotCommand] | None:
    """Return the best executable visible fuel target.

    Args:
        ctx: Decision context.
        allow_unreachable: Whether terrain-blocked targets may use teleport fallback.

    Returns:
        ``(container, command)`` for the best executable fuel target, or
        ``None`` when no visible fuel target can currently be executed.
    """
    container = find_best_fuel(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        allow_unreachable=allow_unreachable,
        now_ms=ctx.timestamp_ms,
        minimum_volume=minimum_recovery_fuel_volume(ctx),
    )
    if container is None:
        return None

    command = walk_or_teleport(ctx, container["x"], container["y"], pickup_kind="fuel")
    if command is None:
        return None
    return (container, command)


def try_collect_critical_fuel(ctx: DecideCtx) -> TickDecisionDict | None:
    """Allow critical fuel to interrupt any mode, including combat.

    Args:
        ctx: Decision context.

    Returns:
        Fuel recovery decision, or ``None`` when fuel is not critical.
    """
    if ctx.fuel > ctx.config["fuel_critical_threshold"]:
        return None
    return _plan_fuel_recovery(ctx, owner_required=False)


def try_collect_fuel(ctx: DecideCtx) -> TickDecisionDict | None:
    """Collect fuel when below the low threshold.

    Args:
        ctx: Decision context.

    Returns:
        Fuel recovery decision, or ``None`` when fuel is healthy.
    """
    if ctx.fuel > ctx.config["fuel_low_threshold"]:
        return None
    return _plan_fuel_recovery(ctx, owner_required=False)


def decide_recover_fuel_mode(ctx: DecideCtx) -> TickDecisionDict:
    """Run the durable ``RECOVER_FUEL`` owner for this tick.

    Args:
        ctx: Decision context.

    Returns:
        Mode-owned fuel recovery decision.

    Raises:
        ValueError: If the durable owner cannot legally produce a recovery
            action.
    """
    return _plan_fuel_recovery(ctx, owner_required=True)


@overload
def _plan_fuel_recovery(
    ctx: DecideCtx,
    *,
    owner_required: Literal[True],
) -> TickDecisionDict: ...


@overload
def _plan_fuel_recovery(
    ctx: DecideCtx,
    *,
    owner_required: Literal[False],
) -> TickDecisionDict | None: ...


def _plan_fuel_recovery(
    ctx: DecideCtx,
    *,
    owner_required: bool,
) -> TickDecisionDict | None:
    """Plan the current tick's fuel recovery action.

    Args:
        ctx: Decision context.
        owner_required: Whether failing to produce an action is a hard error.

    Returns:
        Fuel recovery decision, or ``None`` when the non-owner helper cannot
        produce a legal recovery action.

    Raises:
        ValueError: If ``owner_required`` is true and no legal recovery action
            can be produced.
    """
    base_state, locked_target = locked_resource_target(ctx, "fuel")
    base_state = clear_combat_target(base_state)
    if locked_target is not None:
        target_x = locked_target["x"]
        target_y = locked_target["y"]
        locked_command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="fuel")
        if locked_command is not None:
            emit_ai(
                "continue locked fuel target at (%d,%d) vol=%d (fuel=%d)",
                target_x,
                target_y,
                locked_target["volume"],
                ctx.fuel,
            )
            return make_decision(
                locked_command,
                "COLLECT_FUEL",
                900,
                target_x,
                target_y,
                f"fuel={locked_target['volume']}",
                set_resource_target(base_state, "fuel", target_x, target_y),
                ctx.equip,
            )
        emit_ai("locked fuel target at (%d,%d) no longer executable", target_x, target_y)
        base_state = clear_resource_target(base_state)

    selection = select_fuel_target(ctx, allow_unreachable=True)
    if selection is not None:
        container, command = selection
        target_x = container["x"]
        target_y = container["y"]
        emit_ai(
            "collect fuel at (%d,%d) vol=%d (fuel=%d)",
            target_x,
            target_y,
            container["volume"],
            ctx.fuel,
        )
        return make_decision(
            command,
            "COLLECT_FUEL",
            900,
            target_x,
            target_y,
            f"fuel={container['volume']}",
            set_resource_target(base_state, "fuel", target_x, target_y),
            ctx.equip,
        )

    emit_ai(
        "no actionable fuel target (%s)",
        describe_container_search(
            ctx.filtered,
            ctx.self_state,
            ctx.terrain,
            want_fuel=True,
            allow_unreachable=True,
            minimum_volume=minimum_recovery_fuel_volume(ctx),
        ),
    )
    known_target_decision = _plan_known_fuel_target(ctx, base_state)
    if known_target_decision is not None:
        return known_target_decision
    return _plan_fuel_sense_or_search(ctx, base_state, owner_required=owner_required)


def _plan_known_fuel_target(
    ctx: DecideCtx,
    ai_state: AIStateDict,
) -> TickDecisionDict | None:
    """Approach a previously validated fuel target before blind searching."""
    for container in find_known_fuel_candidates(
        ctx.filtered,
        ctx.self_state,
        now_ms=ctx.timestamp_ms,
        minimum_volume=minimum_recovery_fuel_volume(ctx),
    ):
        target_x = container["x"]
        target_y = container["y"]
        command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="fuel")
        if command is None:
            continue
        emit_ai(
            "approach known fuel at (%d,%d) vol=%d (fuel=%d)",
            target_x,
            target_y,
            container["volume"],
            ctx.fuel,
        )
        return make_decision(
            command,
            "COLLECT_FUEL",
            900,
            target_x,
            target_y,
            f"known_fuel={container['volume']}",
            set_resource_target(ai_state, "fuel", target_x, target_y),
            ctx.equip,
        )
    return None


def _plan_fuel_sense_or_search(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    *,
    owner_required: bool,
) -> TickDecisionDict | None:
    """Sense or reposition when no immediate fuel target exists.

    Args:
        ctx: Decision context.
        ai_state: Base AI state to rewrite.
        owner_required: Whether failing to produce an action is a hard error.

    Returns:
        Fuel recovery decision, or ``None`` when the non-owner helper cannot
        produce a legal recovery action.

    Raises:
        ValueError: If ``owner_required`` is true and no legal search action
            can be produced.
    """
    if can_use_fuel_radar(ctx) and should_scan_resources_in_current_viewport(ctx):
        emit_ai("radar to find fuel (fuel=%d)", ctx.fuel)
        return make_decision(
            make_radar_command(),
            "COLLECT_FUEL",
            900,
            0,
            0,
            "radar_for_fuel",
            AIStateDict(
                **{
                    **ai_state,
                    "last_scan_ms": ctx.timestamp_ms,
                }
            ),
            ctx.equip,
        )

    search = make_resource_search_hop(
        ctx,
        mode="COLLECT_FUEL",
        score=900,
        reason="search_fuel_local",
        ai_state=ai_state,
    )
    if search is not None:
        return search

    emit_ai("walk to viewport edge for fuel (fuel=%d)", ctx.fuel)
    exploration = select_exploration_command(
        ctx,
        candidate_offset=ai_state["patrol_waypoint_index"],
    )
    if exploration is not None:
        edge_x, edge_y, edge_command = exploration
        return make_decision(
            edge_command,
            "COLLECT_FUEL",
            900,
            edge_x,
            edge_y,
            "edge_for_fuel",
            AIStateDict(
                **{
                    **ai_state,
                    "patrol_waypoint_index": ai_state["patrol_waypoint_index"] + 1,
                }
            ),
            ctx.equip,
        )

    if owner_required:
        raise ValueError("RECOVER_FUEL owner expected executable recovery action")
    return None


__all__ = [
    "can_use_fuel_radar",
    "decide_recover_fuel_mode",
    "minimum_recovery_fuel_volume",
    "select_fuel_target",
    "try_collect_critical_fuel",
    "try_collect_fuel",
]
