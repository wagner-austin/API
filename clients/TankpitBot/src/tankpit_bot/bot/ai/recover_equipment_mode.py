"""Durable equipment-recovery owner and shared equipment recovery helpers."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_use_radar,
    clear_resource_target,
    locked_resource_target,
    make_decision,
    needs_emergency_equipment,
    set_resource_target,
    should_scan_resources_in_current_viewport,
)
from tankpit_bot.bot.ai.equipment import (
    describe_container_search,
    find_equipment_candidates,
    find_known_equipment_candidates,
)
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.resource_search import make_resource_search_hop
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import BotCommand, make_radar_command
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.types import ContainerStateDict


def select_equipment_target(
    ctx: DecideCtx,
    *,
    allow_unreachable: bool,
) -> tuple[ContainerStateDict, BotCommand] | None:
    """Return the nearest executable equipment target.

    Args:
        ctx: Decision context.
        allow_unreachable: Whether terrain-blocked targets may use teleport fallback.

    Returns:
        ``(container, command)`` for the nearest executable equipment target, or
        ``None`` when no visible equipment target can currently be executed.
    """
    candidates = find_equipment_candidates(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        allow_unreachable=allow_unreachable,
        now_ms=ctx.timestamp_ms,
    )
    if not candidates:
        return None

    container = candidates[0]
    command = walk_or_teleport(ctx, container["x"], container["y"], pickup_kind="equipment")
    if command is None:
        return None
    return (container, command)


def try_search_critical_equipment(ctx: DecideCtx) -> TickDecisionDict | None:
    """Search locally for equipment when emergency reserves are depleted.

    Args:
        ctx: Decision context.

    Returns:
        Equipment search decision, or ``None`` when emergency search should not
        run this tick.
    """
    if not needs_emergency_equipment(ctx):
        return None
    if ctx.fuel < ctx.config["fuel_low_threshold"]:
        emit_ai("skipping equipment search - fuel too low (%d)", ctx.fuel)
        return None
    search_decision = _plan_equipment_sense_or_search(
        ctx,
        925,
        (
            clear_resource_target(ctx.base)
            if ctx.base["resource_target_kind"] == "equipment"
            else ctx.base
        ),
    )
    if search_decision is None:
        raise ValueError("RECOVER_EQUIPMENT owner expected executable recovery search action")
    return search_decision


def decide_recover_equipment_mode(ctx: DecideCtx) -> TickDecisionDict:
    """Run the durable ``RECOVER_EQUIPMENT`` owner for this tick.

    The owner persists until the combat reserve exit threshold is restored,
    even after the emergency break threshold has already been crossed back
    upward. That prevents the mode from dropping out mid-restock.

    Args:
        ctx: Decision context.

    Returns:
        Mode-owned recovery decision.
    """
    target_reason = "equipment_critical" if needs_emergency_equipment(ctx) else "equipment_restock"
    target_decision = _plan_equipment_target(
        ctx,
        score=925,
        target_reason=target_reason,
        locked_reason="equipment_locked",
    )
    if target_decision is not None:
        return target_decision

    search_decision = _plan_equipment_sense_or_search(
        ctx,
        925,
        (
            clear_resource_target(ctx.base)
            if ctx.base["resource_target_kind"] == "equipment"
            else ctx.base
        ),
    )
    if search_decision is None:
        raise ValueError("RECOVER_EQUIPMENT owner expected executable recovery search action")
    return search_decision


def _plan_equipment_target(
    ctx: DecideCtx,
    *,
    score: int,
    target_reason: str,
    locked_reason: str,
) -> TickDecisionDict | None:
    """Plan an actionable equipment pickup or approach decision.

    Args:
        ctx: Decision context.
        score: Behavior score for actionable targets.
        target_reason: Behavior reason for a newly selected target.
        locked_reason: Behavior reason for a continued locked target.
    Returns:
        Recovery decision, or ``None`` when no actionable target exists.
    """
    base_state, locked_target = locked_resource_target(ctx, "equipment")
    if locked_target is not None:
        target_x = locked_target["x"]
        target_y = locked_target["y"]
        locked_command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="equipment")
        if locked_command is not None:
            emit_ai("continue locked equipment target at (%d,%d)", target_x, target_y)
            updated_state = AIStateDict(
                **{
                    **set_resource_target(base_state, "equipment", target_x, target_y),
                    "equipment_search_failures": 0,
                }
            )
            return make_decision(
                locked_command,
                "COLLECT_EQUIPMENT",
                score,
                target_x,
                target_y,
                locked_reason,
                updated_state,
                ctx.equip,
            )
        emit_ai("locked equipment target at (%d,%d) no longer executable", target_x, target_y)
        base_state = clear_resource_target(base_state)

    selection = select_equipment_target(ctx, allow_unreachable=True)
    if selection is None:
        emit_ai(
            "no executable RECOVER_EQUIPMENT target (%s)",
            describe_container_search(
                ctx.filtered,
                ctx.self_state,
                ctx.terrain,
                want_fuel=False,
                allow_unreachable=True,
            ),
        )
        return _plan_known_equipment_target(ctx, base_state, score=score)

    container, command = selection
    target_x = container["x"]
    target_y = container["y"]
    emit_ai("collect equipment at (%d,%d)", target_x, target_y)
    updated_state = AIStateDict(
        **{
            **set_resource_target(base_state, "equipment", target_x, target_y),
            "equipment_search_failures": 0,
        }
    )
    return make_decision(
        command,
        "COLLECT_EQUIPMENT",
        score,
        target_x,
        target_y,
        target_reason,
        updated_state,
        ctx.equip,
    )


def _plan_known_equipment_target(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    *,
    score: int,
) -> TickDecisionDict | None:
    """Approach a previously validated equipment target before blind search."""
    for container in find_known_equipment_candidates(
        ctx.filtered,
        ctx.self_state,
        now_ms=ctx.timestamp_ms,
    ):
        target_x = container["x"]
        target_y = container["y"]
        command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="equipment")
        if command is None:
            continue
        emit_ai("approach known equipment at (%d,%d)", target_x, target_y)
        updated_state = AIStateDict(
            **{
                **set_resource_target(ai_state, "equipment", target_x, target_y),
                "equipment_search_failures": 0,
            }
        )
        return make_decision(
            command,
            "COLLECT_EQUIPMENT",
            score,
            target_x,
            target_y,
            "known_equipment",
            updated_state,
            ctx.equip,
        )
    return None


def _plan_equipment_sense_or_search(
    ctx: DecideCtx,
    score: int,
    ai_state: AIStateDict,
) -> TickDecisionDict | None:
    """Sense the current viewport or hop to a fresh sector for equipment.

    Args:
        ctx: Decision context.
        score: Behavior score for the sensing/search action.
        ai_state: Base AI state to rewrite.

    Returns:
        Radar or teleport-search decision, or ``None`` when neither is legal.
    """
    if can_use_radar(ctx) and should_scan_resources_in_current_viewport(ctx):
        emit_ai(
            "radar to find equipment (dual=%d homing=%d radar=%d)",
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
        return make_decision(
            make_radar_command(),
            "COLLECT_EQUIPMENT",
            score,
            0,
            0,
            "radar_for_equipment",
            AIStateDict(
                **{
                    **ai_state,
                    "last_scan_ms": ctx.timestamp_ms,
                }
            ),
            ctx.equip,
        )
    return _plan_equipment_search(
        ctx,
        score=score,
        ai_state=ai_state,
        failure_count=ctx.ai_state["equipment_search_failures"],
    )


def _plan_equipment_search(
    ctx: DecideCtx,
    *,
    score: int,
    ai_state: AIStateDict,
    failure_count: int,
) -> TickDecisionDict | None:
    """Plan the next teleport-search hop for equipment recovery.

    Args:
        ctx: Decision context.
        score: Behavior score for the hop.
        ai_state: Base AI state to rewrite.
        failure_count: Current equipment-search failure count.

    Returns:
        Teleport-search decision, or ``None`` when search cannot proceed.
    """
    if failure_count >= ctx.config["equip_search_max_failures"]:
        emit_ai(
            "equipment search hit %d failures - continuing sweep (dual=%d homing=%d radar=%d)",
            failure_count,
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
        failure_count = 0

    return make_resource_search_hop(
        ctx,
        mode="COLLECT_EQUIPMENT",
        score=score,
        reason="search_equipment_local",
        failure_count=failure_count,
        ai_state=ai_state,
    )


__all__ = [
    "decide_recover_equipment_mode",
    "select_equipment_target",
    "try_search_critical_equipment",
]
