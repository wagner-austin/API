"""Resource-search hop: teleport into the cleanest fresh viewport nearby.

Candidate destinations are the eight compass neighbors at one and two
viewport-widths (16-tile cardinals tile cleanly with no overlap;
diagonals land corner-adjacent; the two-width ring reaches past a
scanned or dirty immediate neighborhood). A candidate qualifies when
its landing tile is in bounds (not killed by map-edge clamping),
passable, fuel-affordable, and lands in a viewport with no fresh scan
coverage.

Qualifying candidates are ranked by the **walkable fraction** of the
landing viewport from the static terrain map -- the recorded human
policy is to restock in clean, mostly-"." viewports because the
walk-only pickup contract makes rock/water ground uncollectable
(sessions 2026-07-01, see [[gameplay-loop]]). Candidates are iterated
cheapest-first (16 cardinal, 16 diagonal, 32 cardinal, 32 diagonal --
teleport cost grows with distance), so keeping the first best score
also keeps the cheapest hop among equally clean viewports.

When no candidate qualifies, the function returns ``None`` and the
caller raises -- the bot is genuinely stuck (every nearby viewport
scanned, blocked, or unaffordable) and no second hopping mechanism
papers over that.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport,
    clear_resource_target,
    make_decision,
)
from tankpit_bot.bot.ai.types import AIStateDict, BehaviorMode
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_teleport_command
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.scan_coverage import is_viewport_fully_covered
from tankpit_bot.state.types import coord_key

# Cheapest-first: within a ring, cardinal displacement costs less fuel
# than diagonal (16 vs ~22.6 tiles euclidean).
_HOP_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

# One viewport-width reaches the adjacent non-overlapping viewport; two
# widths reach past a fully scanned or water-heavy immediate ring.
_HOP_RING_MULTIPLIERS: tuple[int, int] = (1, 2)


def _viewport_walkable_fraction(
    ctx: DecideCtx,
    left: int,
    top: int,
    width: int,
    height: int,
) -> float:
    """Return the fraction of viewport tiles that are walkable ground.

    Off-map tiles (viewport clipped at the field border) count as
    unwalkable -- the border is rock. Without a terrain map every
    candidate scores 1.0, so selection degrades to cheapest-first.

    Args:
        ctx: Decision context.
        left: Viewport left X (inclusive).
        top: Viewport top Y (inclusive).
        width: Viewport width in tiles.
        height: Viewport height in tiles.

    Returns:
        Walkable tile count divided by the full viewport area.
    """
    terrain = ctx.terrain
    if terrain is None:
        return 1.0
    walkable = 0
    for y in range(max(0, top), min(255, top + height - 1) + 1):
        for x in range(max(0, left), min(255, left + width - 1) + 1):
            if terrain.is_passable(x, y):
                walkable += 1
    return walkable / (width * height)


def _pick_fresh_viewport_hop(ctx: DecideCtx) -> tuple[int, int] | None:
    """Return the qualifying hop whose landing viewport is most walkable.

    A candidate qualifies when (a) the landing tile clears the map-edge
    clamp, (b) terrain is passable, (c) the teleport is fuel-affordable,
    and (d) the landing viewport has no fresh scan coverage. Among
    qualifiers the highest walkable fraction wins; ties keep the
    earliest (cheapest) candidate.

    Args:
        ctx: Decision context.

    Returns:
        ``(target_x, target_y)`` of the best candidate, or ``None``
        when none qualify.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    distance = ctx.config["equip_search_hop_distance"]
    viewport = ctx.world["viewport"]
    half_w = viewport["width"] // 2
    half_h = viewport["height"] // 2
    best: tuple[int, int] | None = None
    best_score = 0.0
    for multiplier in _HOP_RING_MULTIPLIERS:
        step = distance * multiplier
        for dx, dy in _HOP_DIRECTIONS:
            target_x = max(1, min(254, sx + dx * step))
            target_y = max(1, min(254, sy + dy * step))
            if abs(target_x - sx) != step * abs(dx):
                continue
            if abs(target_y - sy) != step * abs(dy):
                continue
            if ctx.terrain is not None and not ctx.terrain.is_passable(target_x, target_y):
                continue
            if not can_afford_teleport(ctx, target_x, target_y):
                continue
            landing_left = target_x - half_w
            landing_top = target_y - half_h
            if is_viewport_fully_covered(
                ctx.world["scanned_tiles"],
                landing_left,
                landing_top,
                landing_left + viewport["width"] - 1,
                landing_top + viewport["height"] - 1,
                ctx.timestamp_ms,
            ):
                continue
            score = _viewport_walkable_fraction(
                ctx,
                landing_left,
                landing_top,
                viewport["width"],
                viewport["height"],
            )
            if score > best_score:
                best = (target_x, target_y)
                best_score = score
    return best


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
    """Create a teleport decision into the cleanest fresh viewport nearby.

    Candidates at one and two viewport-widths are ranked by the
    walkable fraction of the landing viewport (ties keep the cheapest
    hop). Returns ``None`` when no candidate qualifies -- the caller
    raises rather than fall back to a shorter or smaller hop.

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
    target = _pick_fresh_viewport_hop(ctx)
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
