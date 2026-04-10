"""Shared resource-search helpers for recovery-oriented AI modes.

This module centralizes the local sector-hop logic used by both fuel and
equipment recovery so durable owners do not duplicate teleport-search
behavior.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport_search,
    clear_resource_target,
    teleport_fuel_cost_to,
)
from tankpit_bot.bot.ai.types import AIStateDict, BehaviorMode
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_teleport_command
from tankpit_bot.runtime_logging import emit_ai

_CARDINAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)


def local_resource_search_hop(ctx: DecideCtx) -> tuple[int, int, int]:
    """Compute the next local sector hop for resource recovery.

    Args:
        ctx: Decision context.

    Returns:
        Tuple of ``(target_x, target_y, next_patrol_index)``.
    """
    raw_index = ctx.ai_state["patrol_waypoint_index"]
    index = raw_index % len(_CARDINAL_DIRECTIONS)
    dx, dy = _CARDINAL_DIRECTIONS[index]
    ring = 1 + (raw_index // len(_CARDINAL_DIRECTIONS))
    distance = ctx.config["equip_search_hop_distance"] * ring
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    target_x = max(1, min(254, sx + dx * distance))
    target_y = max(1, min(254, sy + dy * distance))
    return (target_x, target_y, raw_index + 1)


def make_resource_search_hop(
    ctx: DecideCtx,
    *,
    mode: BehaviorMode,
    score: int,
    reason: str,
    failure_count: int | None = None,
    ai_state: AIStateDict | None = None,
) -> TickDecisionDict | None:
    """Create a teleport-search decision for recovery behavior.

    Args:
        ctx: Decision context.
        mode: Behavior mode label for the decision.
        score: Behavior score for the hop.
        reason: Behavior reason label.
        failure_count: Optional consecutive-failure count for recovery search.
        ai_state: Optional AI state base to rewrite before returning.

    Returns:
        Recovery teleport decision, or ``None`` when fuel is too low to hop.
    """
    target_x, target_y, next_index = local_resource_search_hop(ctx)
    if not can_afford_teleport_search(ctx, target_x, target_y):
        emit_ai(
            "cannot afford %s hop to (%d,%d) (fuel=%d cost=%d reserve=%d)",
            reason,
            target_x,
            target_y,
            ctx.fuel,
            teleport_fuel_cost_to(ctx, target_x, target_y),
            ctx.config["hunt_min_fuel"],
        )
        return None

    base_state = ctx.base if ai_state is None else ai_state
    cleared = clear_resource_target(base_state)

    if failure_count is None:
        emit_ai(
            "local resource hop to (%d,%d) (dual=%d homing=%d radar=%d)",
            target_x,
            target_y,
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
        updated_state = AIStateDict(
            **{
                **cleared,
                "patrol_waypoint_index": next_index,
            }
        )
    else:
        next_failures = failure_count + 1
        emit_ai(
            "local resource hop to (%d,%d) (dual=%d homing=%d radar=%d attempt=%d)",
            target_x,
            target_y,
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
            next_failures,
        )
        updated_state = AIStateDict(
            **{
                **cleared,
                "patrol_waypoint_index": next_index,
                "equipment_search_failures": next_failures,
            }
        )

    from tankpit_bot.bot.ai.context import make_decision

    return make_decision(
        make_teleport_command(target_x, target_y),
        mode,
        score,
        target_x,
        target_y,
        reason,
        updated_state,
        ctx.equip,
    )


__all__ = [
    "local_resource_search_hop",
    "make_resource_search_hop",
]
