"""Viewport exploration: where to go when nothing is targeted.

Ranks candidate viewport origins and turns the winner into a move.
Reads :mod:`tankpit_bot.bot.ai.movement` for the move itself; nothing
in that module reads back.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    local_actionable_bounds,
)
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.types import (
    BotCommand,
)
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.scan_coverage import is_viewport_fully_covered
from tankpit_bot.state.types import viewport_scan_key


def select_exploration_command(
    ctx: DecideCtx,
    *,
    candidate_offset: int = 0,
) -> tuple[int, int, BotCommand] | None:
    """Return the first executable exploration step inside the current viewport.

    Exploration is used when the bot wants fresh information but map/radar
    cannot be used immediately. The search stays inside the visible viewport
    and prefers tiles on the edges that are most likely to reveal a
    new viewport next.

    Args:
        ctx: Decision context with viewport, terrain, and fuel state.
        candidate_offset: Rotation offset into the ordered candidate list.

    Returns:
        Tuple of ``(x, y, command)`` for the first executable exploration
        target, or ``None`` when no exploration command can be executed.
    """
    for candidate_x, candidate_y in viewport_exploration_candidates(
        ctx,
        candidate_offset=candidate_offset,
    ):
        command = walk_or_teleport(ctx, candidate_x, candidate_y, pickup_kind=None)
        if command is None:
            continue
        return (candidate_x, candidate_y, command)
    emit_ai("no executable exploration target in current viewport")
    return None


def viewport_exploration_candidates(
    ctx: DecideCtx,
    *,
    candidate_offset: int = 0,
) -> list[tuple[int, int]]:
    """Return ordered exploration targets on the visible viewport boundary.

    Uses the actual viewport bounds rather than assuming the player is
    centered. The player moves freely inside the fixed viewport frame; it only
    recenters when the player reaches the edge. Exploration should therefore
    prefer the real visible edge while trying multiple edge-aligned candidates
    before giving up.

    Args:
        ctx: Decision context with viewport and self position.
        candidate_offset: Rotation offset into the ordered candidate list.

    Returns:
        Ordered unique candidate coordinates inside the visible viewport.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    left, top, right, bottom = local_actionable_bounds(ctx)
    preferred_x = right if sx < 128 else left
    preferred_y = bottom if sy < 128 else top
    alternate_x = left if preferred_x == right else right
    alternate_y = top if preferred_y == bottom else bottom
    clamped_x = min(max(sx, left), right)
    clamped_y = min(max(sy, top), bottom)
    middle_x = (left + right) // 2
    middle_y = (top + bottom) // 2

    ordered = [
        (preferred_x, preferred_y),
        (preferred_x, clamped_y),
        (clamped_x, preferred_y),
        (preferred_x, alternate_y),
        (alternate_x, preferred_y),
        (preferred_x, middle_y),
        (middle_x, preferred_y),
        (alternate_x, clamped_y),
        (clamped_x, alternate_y),
        (alternate_x, alternate_y),
        (alternate_x, middle_y),
        (middle_x, alternate_y),
    ]
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []
    for candidate_x, candidate_y in ordered:
        candidate = (candidate_x, candidate_y)
        if candidate == (sx, sy):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    if not candidates:
        return candidates
    ranked_candidates = [
        (_candidate_priority_for_ctx(ctx, candidate), candidate) for candidate in candidates
    ]
    ranked_candidates.sort(key=_ranked_exploration_key)
    candidates = [candidate for _, candidate in ranked_candidates]
    offset = candidate_offset % len(candidates)
    return candidates[offset:] + candidates[:offset]


def _exploration_priority(
    ctx: DecideCtx,
    target_x: int,
    target_y: int,
) -> tuple[bool, bool, bool, int]:
    """Rank exploration targets by how much fresh scan space they expose."""
    next_left, next_top = _predicted_viewport_origin_for_target(ctx, target_x, target_y)
    current_viewport = ctx.world["viewport"]
    current_key = viewport_scan_key(current_viewport["left"], current_viewport["top"])
    next_key = viewport_scan_key(next_left, next_top)
    is_current = next_key == current_key
    is_failed = ctx.ws.is_scan_viewport_failed(next_left, next_top, ctx.timestamp_ms)
    is_scanned = is_viewport_fully_covered(
        ctx.world["scanned_tiles"],
        next_left,
        next_top,
        next_left + current_viewport["width"] - 1,
        next_top + current_viewport["height"] - 1,
        ctx.forage_floor_ms,
    )
    reveal_distance = abs(target_x - ctx.self_state["x"]) + abs(target_y - ctx.self_state["y"])
    return (is_current, is_failed, is_scanned, -reveal_distance)


def _candidate_priority_for_ctx(
    ctx: DecideCtx,
    candidate: tuple[int, int],
) -> tuple[bool, bool, bool, int]:
    """Return a typed sort key wrapper for exploration candidates."""
    return _exploration_priority(ctx, candidate[0], candidate[1])


def _ranked_exploration_key(
    item: tuple[tuple[bool, bool, bool, int], tuple[int, int]],
) -> tuple[bool, bool, bool, int]:
    """Return the priority part of a ranked exploration candidate."""
    return item[0]


def _predicted_viewport_origin_for_target(
    ctx: DecideCtx,
    target_x: int,
    target_y: int,
) -> tuple[int, int]:
    """Return the viewport origin likely after reaching a target tile."""
    viewport = ctx.world["viewport"]
    max_left = max(0, 256 - viewport["width"])
    max_top = max(0, 256 - viewport["height"])
    return (
        min(max(target_x - (viewport["width"] // 2), 0), max_left),
        min(max(target_y - (viewport["height"] // 2), 0), max_top),
    )


__all__ = [
    "select_exploration_command",
    "viewport_exploration_candidates",
]
