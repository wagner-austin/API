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
    make_decision,
    teleport_fuel_cost_to,
)
from tankpit_bot.bot.ai.equipment import is_area_scanned
from tankpit_bot.bot.ai.types import AIStateDict, BehaviorMode
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_teleport_command
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.types import coord_key

_CARDINAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)

# Hop rings beyond this wrap back to ring 1. Without the cap the patrol
# index grows for the whole session and hop distances scale with it:
# live run 20260610-000x reached 90-tile hops costing 540 fuel, which
# the owner could no longer afford -- it then had no legal action left.
_MAX_SEARCH_RINGS = 3

# A hop whose clamped target moves the tank fewer tiles than this is a
# degenerate re-visit (the map-edge clamp collapsed it onto the current
# position) and never worth a teleport.
_MIN_HOP_DISPLACEMENT = 4


def _hop_target_for_cycle(ctx: DecideCtx, cycle: int) -> tuple[int, int]:
    """Return the clamped hop target for a position in the ring cycle.

    Args:
        ctx: Decision context.
        cycle: Position within the ``directions x rings`` cycle.

    Returns:
        Clamped ``(target_x, target_y)`` for the cycle position.
    """
    index = cycle % len(_CARDINAL_DIRECTIONS)
    dx, dy = _CARDINAL_DIRECTIONS[index]
    ring = 1 + (cycle // len(_CARDINAL_DIRECTIONS))
    distance = ctx.config["equip_search_hop_distance"] * ring
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    target_x = max(1, min(254, sx + dx * distance))
    target_y = max(1, min(254, sy + dy * distance))
    return (target_x, target_y)


def _is_worthwhile_hop(ctx: DecideCtx, target_x: int, target_y: int) -> bool:
    """Return True when a hop target reveals ground worth scanning.

    Rejected when: the hop barely moves (map-edge clamp), the landing
    viewport is already scanned, or terrain says the landing tile is
    impassable (water/rock — the server would displace us and the
    scan covers water, wasting an extra radar).

    Args:
        ctx: Decision context.
        target_x: Clamped hop target X coordinate.
        target_y: Clamped hop target Y coordinate.

    Returns:
        True when the hop moves far enough, lands on passable ground,
        and covers unscanned territory.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    if abs(target_x - sx) + abs(target_y - sy) < _MIN_HOP_DISPLACEMENT:
        return False
    if ctx.terrain is not None and not ctx.terrain.is_passable(target_x, target_y):
        return False
    viewport = ctx.world["viewport"]
    landing_left = target_x - viewport["width"] // 2
    landing_top = target_y - viewport["height"] // 2
    return not is_area_scanned(ctx.world, landing_left, landing_top, ctx.timestamp_ms)


_SHORT_HOP_DISTANCE = 8


def _short_hop_fallback(ctx: DecideCtx) -> tuple[int, int] | None:
    """Try a cheap short hop when the standard search hop is unaffordable.

    Cycles through the four cardinal directions at a fixed short distance
    and returns the first affordable, worthwhile target. Returns None only
    when even the shortest hop is unaffordable.

    Args:
        ctx: Decision context.

    Returns:
        Target coordinates for the short hop, or None.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    for dx, dy in _CARDINAL_DIRECTIONS:
        tx = max(1, min(254, sx + dx * _SHORT_HOP_DISTANCE))
        ty = max(1, min(254, sy + dy * _SHORT_HOP_DISTANCE))
        if abs(tx - sx) + abs(ty - sy) < _MIN_HOP_DISPLACEMENT:
            continue
        if not can_afford_teleport_search(ctx, tx, ty):
            continue
        return (tx, ty)
    return None


def local_resource_search_hop(ctx: DecideCtx) -> tuple[int, int, int]:
    """Compute the next local sector hop for resource recovery.

    Starting from the persisted patrol index, the full ring cycle is
    searched for the first hop that escapes the current position and
    lands on unscanned ground. When every cycle position is degenerate
    or covered, the raw indexed hop is returned so the owner always has
    a target -- coverage expires, so this self-heals within the scan
    TTL.

    Args:
        ctx: Decision context.

    Returns:
        Tuple of ``(target_x, target_y, next_patrol_index)``.
    """
    raw_index = ctx.ai_state["patrol_waypoint_index"]
    cycle_length = len(_CARDINAL_DIRECTIONS) * _MAX_SEARCH_RINGS
    for offset in range(cycle_length):
        cycle = (raw_index + offset) % cycle_length
        target_x, target_y = _hop_target_for_cycle(ctx, cycle)
        if _is_worthwhile_hop(ctx, target_x, target_y):
            return (target_x, target_y, raw_index + offset + 1)
    target_x, target_y = _hop_target_for_cycle(ctx, raw_index % cycle_length)
    return (target_x, target_y, raw_index + 1)


def is_recently_attempted(
    attempted: dict[str, int],
    x: int,
    y: int,
    now_ms: int,
    *,
    ttl_ms: int,
) -> bool:
    """Return True when a coordinate carries a live attempt mark.

    Args:
        attempted: Attempt marks keyed by "x,y" with dispatch timestamps.
        x: Target X coordinate.
        y: Target Y coordinate.
        now_ms: Current timestamp for TTL evaluation.
        ttl_ms: Mark lifetime in milliseconds.

    Returns:
        True if the coordinate was attempted within the TTL.
    """
    attempted_ms = attempted.get(coord_key(x, y))
    return attempted_ms is not None and now_ms - attempted_ms <= ttl_ms


def record_attempt_mark(
    attempted: dict[str, int],
    x: int,
    y: int,
    now_ms: int,
    *,
    ttl_ms: int,
) -> dict[str, int]:
    """Return attempt marks with expired entries pruned and (x, y) recorded.

    Args:
        attempted: Attempt marks keyed by "x,y" with dispatch timestamps.
        x: Target X coordinate to record.
        y: Target Y coordinate to record.
        now_ms: Dispatch timestamp recorded for the new mark.
        ttl_ms: Mark lifetime in milliseconds used for pruning.

    Returns:
        New attempt-mark mapping.
    """
    pruned = {
        key: marked_ms for key, marked_ms in attempted.items() if now_ms - marked_ms <= ttl_ms
    }
    pruned[coord_key(x, y)] = now_ms
    return pruned


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
        short = _short_hop_fallback(ctx)
        if short is not None:
            target_x, target_y = short
            next_index = ctx.ai_state["patrol_waypoint_index"]
            emit_ai(
                "standard hop unaffordable, short hop to (%d,%d) (fuel=%d cost=%d)",
                target_x,
                target_y,
                ctx.fuel,
                teleport_fuel_cost_to(ctx, target_x, target_y),
            )
        else:
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
    "is_recently_attempted",
    "local_resource_search_hop",
    "make_resource_search_hop",
    "record_attempt_mark",
]
