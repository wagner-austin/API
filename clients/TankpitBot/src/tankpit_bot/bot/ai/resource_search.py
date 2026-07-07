"""Resource-search hop: teleport to the nearest clean-viewport fuel dot.

Candidate destinations come from the 0x4C MapData fuel-dot atlas — the
map's yellow-pixel fuel positions (server-cached per session; ~40% of
dots still hold fuel when visited, and every wire-verified dot held
high-volume fuel). Hopping dot-to-dot replaces the old blind compass
hop: each landing is in fuel-rich ground and the landing auto-pickup
makes the hop partially self-funding (user contract 2026-07-03: "hop
to nearest yellow dot with a 100% clean viewport").

A candidate dot qualifies when its landing tile is passable, the
teleport is fuel-affordable, the landing viewport has no fresh scan
coverage, and the landing viewport is 100% walkable ground from the
static terrain map (the walk-only pickup contract makes rock/water
tiles uncollectable, so a dirty viewport wastes the hop). Qualifiers
are taken nearest-first. Without a terrain map the walkable check
degrades to 1.0 and selection is purely nearest-affordable-unscanned.

When the atlas is empty (no map open yet this session) the hop
dispatches ``map_open`` — the dots arrive with the 0x4C response —
guarded by ``map_open_cooldown_ms`` so a dotless map cannot loop.
When no dot qualifies, the function returns ``None`` and the caller
raises — the bot is genuinely stuck and no second hopping mechanism
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
from tankpit_bot.bot.types import make_map_open_command, make_teleport_command
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.scan_coverage import is_viewport_fully_covered
from tankpit_bot.state.types import coord_key


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
    candidate scores 1.0, so selection degrades to nearest-first.

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


def _pick_fresh_dot_hop(ctx: DecideCtx) -> tuple[int, int] | None:
    """Return the nearest fuel dot whose landing viewport is 100% clean.

    A dot qualifies when (a) it is not the bot's own tile, (b) its
    landing tile is passable, (c) the teleport is fuel-affordable,
    (d) the landing viewport has no fresh scan coverage, and (e) the
    landing viewport is fully walkable ground. Candidates are tried
    nearest-first (euclidean, matching teleport cost).

    Args:
        ctx: Decision context.

    Returns:
        ``(target_x, target_y)`` of the nearest qualifying dot, or
        ``None`` when none qualify.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    viewport = ctx.world["viewport"]
    half_w = viewport["width"] // 2
    half_h = viewport["height"] // 2

    def _distance_sq(dot: tuple[int, int]) -> int:
        return (dot[0] - sx) ** 2 + (dot[1] - sy) ** 2

    for target_x, target_y in sorted(ctx.map_fuel_dots, key=_distance_sq):
        if (target_x, target_y) == (sx, sy):
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
        if (
            _viewport_walkable_fraction(
                ctx,
                landing_left,
                landing_top,
                viewport["width"],
                viewport["height"],
            )
            < 1.0
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


def _open_map_for_dots(
    ctx: DecideCtx,
    *,
    mode: BehaviorMode,
    score: int,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Dispatch ``map_open`` to populate the fuel-dot atlas.

    The atlas arrives with the 0x4C MapData response, so the first hop
    of a session may need one map open before any dot candidates
    exist. Guarded by ``map_open_cooldown_ms``: if a recent map open
    produced no dots there is nothing more to learn and the caller's
    exit path takes over.

    Args:
        ctx: Decision context.
        mode: Behavior mode label for the decision.
        score: Behavior score for the map open.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        Map-open decision, or ``None`` when a recent map open already
        failed to yield dots.
    """
    map_age_ms = ctx.timestamp_ms - ctx.ai_state["last_map_open_ms"]
    if ctx.ai_state["last_map_open_ms"] > 0 and map_age_ms <= ctx.config["map_open_cooldown_ms"]:
        return None
    emit_ai("opening map to load the fuel-dot atlas")
    return make_decision(
        make_map_open_command(),
        mode,
        score,
        0,
        0,
        "map_for_dots",
        AIStateDict(
            **{
                **base_state,
                "last_map_open_ms": ctx.timestamp_ms,
            }
        ),
        ctx.equip,
    )


def make_resource_search_hop(
    ctx: DecideCtx,
    *,
    mode: BehaviorMode,
    score: int,
    reason: str,
    ai_state: AIStateDict | None = None,
) -> TickDecisionDict | None:
    """Create a teleport decision to the nearest clean-viewport fuel dot.

    Landing on a dot auto-picks any fuel there, so each restock hop is
    partially self-funding. With an empty atlas the decision is a
    ``map_open`` (dots arrive with the 0x4C response). Returns
    ``None`` when no dot qualifies -- the caller raises rather than
    fall back to a blind hop.

    Args:
        ctx: Decision context.
        mode: Behavior mode label for the decision.
        score: Behavior score for the hop.
        reason: Behavior reason label.
        ai_state: Optional AI state base to rewrite before returning.

    Returns:
        Teleport (or atlas-loading map-open) decision, or ``None``
        when no fresh-viewport dot hop is possible from here.
    """
    base_state = ctx.base if ai_state is None else ai_state
    if not ctx.map_fuel_dots:
        return _open_map_for_dots(ctx, mode=mode, score=score, base_state=base_state)
    target = _pick_fresh_dot_hop(ctx)
    if target is None:
        return None
    target_x, target_y = target
    emit_ai(
        "fuel-dot hop to (%d,%d) (dual=%d homing=%d radar=%d)",
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
