"""Equipment foraging: tile-level sweep of the current viewport.

Equipment containers are radar-only-visible and the only source of
extra radars. The forager's job is to make sure every tile inside
the current viewport gets scanned, so any equipment / fuel container
on those tiles is discovered, picked up, and the tile-coverage map
filled.

The optimal play (user-confirmed 2026-06-22):

1. Teleport into a fresh viewport.
2. Fire one radar (extra reveals all 256 viewport tiles; free reveals
   the 5x5 around the tank intersected with the viewport bounds).
3. Walk to each container revealed and pick it up.
4. When extras are exhausted but the viewport is not fully scanned,
   walk ~5 tiles (matching the free-radar diameter) to a position
   whose next free radar covers the most uncovered ground, then radar
   again. Repeat until the viewport is scanned or extras are restored.
5. When every tile in the current viewport is scanned, teleport away
   -- the only way the viewport changes in this game configuration.

This module owns steps 1/2/4/5 of that loop: fire a radar when there
is still unscanned ground in the viewport, otherwise walk to the
viewport position that maximises the next free radar's coverage
gain, otherwise (whole viewport swept or no productive walk exists)
return ``None`` so the caller can teleport out. Step 3 lives in the
equipment / fuel recovery owners (each container hit is its own
``pickup_*`` decision).
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    clear_resource_target,
    make_decision,
)
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.types import AIStateDict, BehaviorMode
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_radar_command
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.scan_coverage import (
    free_radar_new_coverage,
    is_viewport_fully_covered,
    select_best_free_radar_position,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds


def select_forage_target(ctx: DecideCtx) -> tuple[int, int] | None:
    """Return the viewport tile whose next free radar reveals the most uncovered ground.

    Picks the destination that maximises next-radar coverage gain
    (5x5 footprint clipped to viewport, minus already-scanned tiles)
    rather than the nearest unscanned tile -- the optimal walk step
    for the free-radar tile-expansion strategy is ~5 tiles to match
    the radar diameter, not 1. Ties broken by Manhattan distance.

    Args:
        ctx: Decision context.

    Returns:
        ``(x, y)`` of the best destination tile, or ``None`` when no
        viewport position would reveal any new ground (viewport
        effectively scanned).
    """
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    return select_best_free_radar_position(
        ctx.world["scanned_tiles"],
        ctx.self_state["x"],
        ctx.self_state["y"],
        left,
        top,
        right,
        bottom,
        ctx.timestamp_ms,
        ctx.self_state["rank"],
    )


def plan_forage_search(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    *,
    score: int,
    behavior_mode: BehaviorMode,
    radar_affordable: bool,
) -> TickDecisionDict | None:
    """Plan the next foraging action inside the current viewport.

    The forager is mode-agnostic. Callers supply their own
    ``behavior_mode``, ``score``, and ``radar_affordable`` predicate;
    the function itself never branches on caller identity.

    Three branches in order:

    1. If any tile in the current viewport is unscanned AND
       ``radar_affordable`` is True → dispatch the radar. The wire
       handler records the revealed tile set into
       ``world.scanned_tiles`` when the server radar response
       arrives next tick.
    2. Else if an unscanned tile is reachable → walk toward it so the
       next tick's free radar (or a paid radar once affordable again)
       reveals it.
    3. Else the viewport is exhausted -- or the only viable approach is
       a teleport the caller cannot afford. Return ``None`` so the
       caller can teleport to a fresh viewport (this game configuration
       has viewport shifting OFF, so a teleport is the only way to
       reveal new ground).

    Args:
        ctx: Decision context.
        ai_state: Base AI state to rewrite for the produced command.
        score: Behavior score for the produced decision.
        behavior_mode: Behavior-mode label stamped on the decision.
        radar_affordable: Caller-evaluated predicate that says the
            radar's fuel cost is payable under the caller's reserve
            policy this tick.

    Returns:
        Foraging radar or move decision, or ``None`` when no productive
        in-viewport action exists this tick.
    """
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    viewport_fully_covered = is_viewport_fully_covered(
        ctx.world["scanned_tiles"],
        left,
        top,
        right,
        bottom,
        ctx.timestamp_ms,
    )
    # Extras reveal the whole viewport; free radar only reveals a
    # ``(2r+1)x(2r+1)`` footprint around the tank
    # (``r = free_radar_radius(rank)``). When no extras are stocked
    # AND the tank is inside the viewport, firing a free radar from a
    # spot whose footprint is already fully covered would mark zero
    # new tiles -- the tank must walk first so a later free radar
    # reaches new ground. Without this gate the forager loops firing
    # the same free radar from the same position whenever
    # ``can_use_radar`` is permissive (radar is fuel-free). The gate
    # intentionally only applies inside the viewport: if the tank is
    # somehow outside it (test setup, pre-synced wire state), there's
    # no walk that helps, so let radar fire and rely on the next wire
    # viewport update to converge state.
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    has_extras = ctx.inventory["extra_radars"]["count"] > 0
    tank_in_viewport = left <= sx <= right and top <= sy <= bottom
    if has_extras or not tank_in_viewport:
        radar_productive = True
    else:
        next_radar_gain = free_radar_new_coverage(
            ctx.world["scanned_tiles"],
            sx,
            sy,
            left,
            top,
            right,
            bottom,
            ctx.timestamp_ms,
            ctx.self_state["rank"],
        )
        radar_productive = next_radar_gain > 0
    if not viewport_fully_covered and radar_affordable and radar_productive:
        emit_ai(
            "forage radar (mode=%s, extras=%d, viewport=(%d,%d)-(%d,%d))",
            behavior_mode,
            ctx.inventory["extra_radars"]["count"],
            left,
            top,
            right,
            bottom,
        )
        return make_decision(
            make_radar_command(),
            behavior_mode,
            score,
            0,
            0,
            "forage_radar",
            clear_resource_target(ai_state),
            ctx.equip,
        )

    target = select_forage_target(ctx)
    if target is None:
        return None
    target_x, target_y = target
    command = walk_or_teleport(ctx, target_x, target_y, pickup_kind=None)
    if command is None:
        return None
    emit_ai("forage walk to unscanned tile (%d,%d) mode=%s", target_x, target_y, behavior_mode)
    return make_decision(
        command,
        behavior_mode,
        score,
        target_x,
        target_y,
        "forage_sweep",
        clear_resource_target(ai_state),
        ctx.equip,
    )


__all__ = [
    "plan_forage_search",
    "select_forage_target",
]
