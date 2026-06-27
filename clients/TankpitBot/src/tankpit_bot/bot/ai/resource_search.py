"""Resource-search hop: teleport into the next fresh viewport.

A 16-tile cardinal hop lands the bot in a viewport whose left/top edge
is exactly one viewport-width away from the current one, so the new
viewport tiles cleanly with no overlap. Diagonals (16 in each axis)
land in a corner-adjacent viewport, also non-overlapping; they cost
more fuel but are still single-step fresh-viewport moves. Cardinals
are tried first because they are cheaper; diagonals are tried only
when no cardinal qualifies.

A direction qualifies when its landing tile is in bounds (not killed
by map-edge clamping), passable, fuel-affordable, and lands in a
viewport whose origin has no fresh scan coverage. When no direction
qualifies, the function returns ``None`` and the caller raises -- the
bot is genuinely stuck (every adjacent viewport scanned, blocked, or
unaffordable) and no second hopping mechanism papers over that.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport,
    clear_resource_target,
    make_decision,
)
from tankpit_bot.bot.ai.equipment import is_area_scanned
from tankpit_bot.bot.ai.types import AIStateDict, BehaviorMode
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_teleport_command
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.types import coord_key

_CARDINAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)

_DIAGONAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def _pick_fresh_viewport_hop(
    ctx: DecideCtx,
    directions: tuple[tuple[int, int], ...],
) -> tuple[int, int] | None:
    """Return the first direction whose landing is a usable fresh viewport.

    A direction qualifies when (a) the landing tile clears the map-edge
    clamp, (b) terrain is passable, (c) the teleport is fuel-affordable,
    and (d) the landing viewport has no fresh scan coverage.

    Args:
        ctx: Decision context.
        directions: Direction vectors to evaluate in order.

    Returns:
        ``(target_x, target_y)`` for the first qualifying direction, or
        ``None`` when none qualify.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    distance = ctx.config["equip_search_hop_distance"]
    viewport = ctx.world["viewport"]
    half_w = viewport["width"] // 2
    half_h = viewport["height"] // 2
    for dx, dy in directions:
        target_x = max(1, min(254, sx + dx * distance))
        target_y = max(1, min(254, sy + dy * distance))
        if abs(target_x - sx) != distance * abs(dx):
            continue
        if abs(target_y - sy) != distance * abs(dy):
            continue
        if ctx.terrain is not None and not ctx.terrain.is_passable(target_x, target_y):
            continue
        if not can_afford_teleport(ctx, target_x, target_y):
            continue
        if is_area_scanned(
            ctx.world,
            target_x - half_w,
            target_y - half_h,
            ctx.timestamp_ms,
        ):
            continue
        return (target_x, target_y)
    return None


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
    ai_state: AIStateDict | None = None,
) -> TickDecisionDict | None:
    """Create a teleport decision into the next fresh-viewport neighbor.

    Cardinals are tried first (cheapest fresh hop). If no cardinal
    qualifies, diagonals are tried (more expensive but still a
    non-overlapping viewport step). Returns ``None`` when no direction
    qualifies -- the caller raises rather than fall back to a shorter
    or smaller hop.

    Args:
        ctx: Decision context.
        mode: Behavior mode label for the decision.
        score: Behavior score for the hop.
        reason: Behavior reason label.
        ai_state: Optional AI state base to rewrite before returning.

    Returns:
        Teleport decision, or ``None`` when no fresh-viewport hop is
        possible from here.
    """
    target = _pick_fresh_viewport_hop(ctx, _CARDINAL_DIRECTIONS)
    if target is None:
        target = _pick_fresh_viewport_hop(ctx, _DIAGONAL_DIRECTIONS)
    if target is None:
        return None
    target_x, target_y = target
    base_state = ctx.base if ai_state is None else ai_state
    emit_ai(
        "fresh-viewport hop to (%d,%d) (dual=%d homing=%d radar=%d)",
        target_x,
        target_y,
        ctx.inventory["dual_shots"]["count"],
        ctx.inventory["homing_shots"]["count"],
        ctx.inventory["extra_radars"]["count"],
    )
    return make_decision(
        make_teleport_command(target_x, target_y),
        mode,
        score,
        target_x,
        target_y,
        reason,
        clear_resource_target(base_state),
        ctx.equip,
    )


__all__ = [
    "is_recently_attempted",
    "make_resource_search_hop",
    "record_attempt_mark",
]
