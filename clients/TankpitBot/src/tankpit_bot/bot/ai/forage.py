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
    make_decision,
)
from tankpit_bot.bot.ai.movement import plan_viewport_walk
from tankpit_bot.bot.ai.radar_economics import (
    radar_spend_worthwhile,
)
from tankpit_bot.bot.ai.scoring_types import BehaviorMode
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_radar_command, make_scope_shift_command
from tankpit_bot.protocol.commands import (
    SCOPE_EAST,
    SCOPE_NORTH,
    SCOPE_SOUTH,
    SCOPE_WEST,
)
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.state.scan_coverage import (
    free_radar_new_coverage,
    is_tile_covered,
    is_viewport_fully_covered,
    select_best_free_radar_position,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

_FRONTIER_BAND_DEPTH = 8
"""How far past a viewport edge the frontier scorer looks.

Half a window: a free ``Rb`` pan anchors the window to the tank
([[viewport-shift-protocol]]), so a pan from the covered window's
interior nets about this much fresh ground beyond the old edge --
the band it scores is exactly what the next scan-walk-scan cycle
will work."""


def _frontier_walk_target(
    ctx: DecideCtx,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> tuple[tuple[int, int], tuple[int, int], int] | None:
    """Pick the frontier band facing the most unscanned ground.

    The zero-extras lawnmower's continuation (user doctrine
    2026-08-14: "move to the NEXT VIEWPORT OVER"): score the four
    bands just beyond the window's edges by uncovered-tile count and
    face the richest one. The window itself NEVER moves on a walk --
    autoscroll is pinned OFF, so only a teleport or a free ``Rb``
    pan shifts it ([[viewport-shift-protocol]] acceptance boundary).
    The caller walks to the returned edge tile first (a pan from the
    facing edge reveals a full 15 fresh tiles under the anchor law,
    against 8 from the window's centre), then pans toward the band.

    Args:
        ctx: Decision context.
        left: Viewport left bound.
        top: Viewport top bound.
        right: Viewport right bound.
        bottom: Viewport bottom bound.

    Returns:
        ``(edge, beyond, direction)`` -- the in-window edge tile to
        walk toward, the band's first tile past that edge, and the
        compass byte a scope pan toward the band takes -- or ``None``
        when every adjacent band is already covered (the search hop
        relocates instead). A winning band is non-empty, so its
        beyond tile is always on the map.
    """
    scanned = ctx.world["scanned_tiles"]
    floor_ms = ctx.forage_floor_ms
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    bands: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    east = range(right + 1, min(right + _FRONTIER_BAND_DEPTH, 255) + 1)
    west = range(max(left - _FRONTIER_BAND_DEPTH, 0), left)
    south = range(bottom + 1, min(bottom + _FRONTIER_BAND_DEPTH, 255) + 1)
    north = range(max(top - _FRONTIER_BAND_DEPTH, 0), top)
    bands.append(
        (
            sum(
                1
                for x in east
                for y in range(top, bottom + 1)
                if not is_tile_covered(scanned, x, y, floor_ms)
            ),
            (right, sy),
            (right + 1, sy),
            SCOPE_EAST,
        )
    )
    bands.append(
        (
            sum(
                1
                for x in west
                for y in range(top, bottom + 1)
                if not is_tile_covered(scanned, x, y, floor_ms)
            ),
            (left, sy),
            (left - 1, sy),
            SCOPE_WEST,
        )
    )
    bands.append(
        (
            sum(
                1
                for x in range(left, right + 1)
                for y in south
                if not is_tile_covered(scanned, x, y, floor_ms)
            ),
            (sx, bottom),
            (sx, bottom + 1),
            SCOPE_SOUTH,
        )
    )
    bands.append(
        (
            sum(
                1
                for x in range(left, right + 1)
                for y in north
                if not is_tile_covered(scanned, x, y, floor_ms)
            ),
            (sx, top),
            (sx, top - 1),
            SCOPE_NORTH,
        )
    )
    best_count, best_edge, best_beyond, best_direction = max(bands, key=_band_score)
    if best_count == 0:
        return None
    return best_edge, best_beyond, best_direction


def _band_score(band: tuple[int, tuple[int, int], tuple[int, int], int]) -> int:
    """Return the uncovered-tile count a frontier band carries."""
    return band[0]


def _frontier_pan(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    behavior_mode: BehaviorMode,
    score: int,
    beyond_x: int,
    beyond_y: int,
    direction: int,
) -> TickDecisionDict:
    """Pan the free viewport toward the richest uncovered band.

    Reached when the frontier's walk target IS the tank's own tile:
    the tank stands on the facing edge and the window has been walked
    to exhaustion. With autoscroll pinned OFF the window never moves
    on a walk, so re-dispatching the walk is a zero-length move the
    server rejects with 0x52 code 6 ("You are already there") --
    exactly the loop demo-1 span 2026-09-05 17:48-19:00, 395
    identical ``move -> (239,48)`` dispatches from (239,48), the NE
    corner of window (224,48). The ``Rb`` scope pan is the maroon
    pan-walk gait's cure applied to coverage: it costs nothing,
    anchors the window to the tank in the band's direction (the
    measured anchor law, [[viewport-shift-protocol]]), and a cardinal
    pan from the facing edge always shifts the window, revealing 15
    fresh tiles for the scan-walk-scan loop to resume on.

    Args:
        ctx: Decision context.
        ai_state: AI state to carry through the decision.
        behavior_mode: Owning behavior mode label.
        score: Priority score.
        beyond_x: First band tile past the exhausted edge.
        beyond_y: First band tile past the exhausted edge.
        direction: Compass byte for the scope pan.

    Returns:
        The scope-shift decision.
    """
    emit_ai(
        "forage frontier pan direction %d toward unscanned band at (%d,%d) mode=%s",
        direction,
        beyond_x,
        beyond_y,
        behavior_mode,
    )
    emit_diagnostic(
        diagnostic_kind="forage_frontier_pan",
        target_x=beyond_x,
        target_y=beyond_y,
        direction=direction,
    )
    return make_decision(
        make_scope_shift_command(direction),
        behavior_mode,
        score,
        beyond_x,
        beyond_y,
        "forage_frontier_pan",
        ai_state,
        ctx.equip,
        reason_context={"direction": direction},
    )


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
        ctx.forage_floor_ms,
        ctx.self_state["rank"],
    )


def plan_forage_search(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    *,
    score: int,
    behavior_mode: BehaviorMode,
) -> TickDecisionDict | None:
    """Plan the next foraging action inside the current viewport.

    The forager is mode-agnostic. Callers supply their own
    ``behavior_mode`` and ``score``; the function itself never branches
    on caller identity.

    Radar affordability is NOT a branch here. Radar dispatch can never
    kill the bot: the server accepts the command even at zero fuel
    (user-confirmed 2026-06-26), and the wire-reported 10-fuel
    deduction is a debit, not a precondition — which is what lets a
    stranded bot still see what is around it instead of looping
    silently. The old ``radar_affordable`` parameter carried a
    caller-supplied predicate back when two recovery modes each had
    their own; both modes are gone, the surviving caller passed a
    constant True, and the parameter went with them
    ([[session-state-deglobalisation]]).

    Three branches in order:

    1. If any tile in the current viewport is unscanned → dispatch the
       radar. The wire handler records the revealed tile set into
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
        ctx.forage_floor_ms,
    )
    # Extras reveal the whole viewport; free radar only reveals a
    # ``(2r+1)x(2r+1)`` footprint around the tank
    # (``r = free_radar_radius(rank)``). When no extras are stocked
    # AND the tank is inside the viewport, firing a free radar from a
    # spot whose footprint is already fully covered would mark zero
    # new tiles -- the tank must walk first so a later free radar
    # reaches new ground. Without this gate the forager loops firing
    # the same free radar from the same position, since radar is
    # fuel-free and nothing else would stop it. The gate
    # intentionally only applies inside the viewport: if the tank is
    # somehow outside it (test setup, pre-synced wire state), there's
    # no walk that helps, so let radar fire and rely on the next wire
    # viewport update to converge state.
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    has_extras = ctx.inventory["extra_radars"]["count"] > 0
    tank_in_viewport = left <= sx <= right and top <= sy <= bottom
    if has_extras:
        # An extra-radar scan consumes an item; the shared spend
        # economics decide, not mere non-full coverage (flag s9-5:
        # a stocked forage radar bought a sliver of tiles and the
        # tank hopped away the next tick).
        radar_productive = radar_spend_worthwhile(ctx)
        if not radar_productive:
            # With extras stocked, the game spends an extra on ANY
            # radar fire and an extra scans the whole viewport from
            # anywhere -- walking can never improve the scan, and the
            # free radar can never fire. Foraging here is FINISHED:
            # returning a decision anyway starved the collect hop one
            # rung below and produced a one-tile-per-tick edge crawl
            # (artax flags 1-3, 2026-08-06). Say so by yielding
            # nothing.
            return None
    elif not tank_in_viewport:
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
            ctx.forage_floor_ms,
            ctx.self_state["rank"],
        )
        radar_productive = next_radar_gain > 0
    if not viewport_fully_covered and radar_productive:
        emit_ai(
            "forage radar (mode=%s, extras=%d, viewport=(%d,%d)-(%d,%d))",
            behavior_mode,
            ctx.inventory["extra_radars"]["count"],
            left,
            top,
            right,
            bottom,
        )
        # Coverage decisions here (radar, frontier walk, sweep walk)
        # PRESERVE any held resource lock: a coverage tick is not a
        # pursuit, and a held lock survives non-pursuit ticks (the
        # s11-5 law in ``block_harvest``). Until 2026-09-02 these
        # three sites cleared the lock silently — the same violation
        # class as the quad sweep's livelock amplifier
        # ([[flag-triage-20260902]]); ``forage_frontier`` already
        # preserved.
        return make_decision(
            make_radar_command(),
            behavior_mode,
            score,
            0,
            0,
            "forage_radar",
            ai_state,
            ctx.equip,
        )

    target = select_forage_target(ctx)
    if target is None:
        # Only the zero-extras path reaches an exhausted selector: a
        # stocked forager either fired the radar above (spend worthy,
        # coverage incomplete) or ended foraging at the economics
        # gate -- worthy-but-fully-covered is a contradiction, since
        # the spend floor counts the same uncovered tiles coverage
        # does.
        #
        # Frontier continuation (user free-radar doctrine 2026-08-14:
        # "scan unique 5x5 areas until the viewport is fully scanned,
        # then move to the NEXT VIEWPORT OVER"): walk to the window
        # edge facing the least-scanned adjacent band, then spend a
        # free scope pan toward it. The window NEVER slides on a walk
        # -- autoscroll is pinned OFF, so only a teleport or an
        # ``Rb`` pan moves it ([[viewport-shift-protocol]]); the
        # pre-pan version of this branch believed otherwise and, once
        # the tank stood ON the facing edge tile, re-dispatched a
        # zero-length move every tick forever (demo-1 2026-09-05,
        # 395 rejected ``move -> (239,48)`` from (239,48)). No
        # teleport: relocation by hop is the search hop's job, and it
        # only gets the tick when every adjacent band is already
        # covered.
        frontier = _frontier_walk_target(ctx, left, top, right, bottom)
        if frontier is None:
            return None
        (frontier_x, frontier_y), (beyond_x, beyond_y), pan_direction = frontier
        if (frontier_x, frontier_y) == (sx, sy):
            return _frontier_pan(
                ctx,
                ai_state,
                behavior_mode,
                score,
                beyond_x,
                beyond_y,
                pan_direction,
            )
        command = plan_viewport_walk(ctx, frontier_x, frontier_y)
        if command is None:
            return None
        emit_ai(
            "forage frontier walk toward (%d,%d) mode=%s",
            frontier_x,
            frontier_y,
            behavior_mode,
        )
        return make_decision(
            command,
            behavior_mode,
            score,
            frontier_x,
            frontier_y,
            "forage_frontier_walk",
            ai_state,
            ctx.equip,
        )
    target_x, target_y = target
    # Coverage steps WALK, never teleport (user ruling 2026-08-14):
    # a free radar reveals ground for nothing, so no coverage step is
    # worth a 45+ fuel hop. An unwalkable best position means this
    # viewport is done for free-scan coverage -- yield to the search
    # hop, which relocates to a genuinely fresh viewport.
    command = plan_viewport_walk(ctx, target_x, target_y)
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
        ai_state,
        ctx.equip,
    )


__all__ = [
    "plan_forage_search",
    "select_forage_target",
]
