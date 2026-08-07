"""The quad sweep: stationary 4-shift/4-radar recon, then loop harvest.

User-derived doctrine ([[quad-sweep-doctrine]], 2026-08-06), built on
measured laws only. With autoscroll OFF, from one standing position:
shift the scope NW and fire an extra radar, then NE, SE, SW. The
ANCHOR law ([[viewport-shift-protocol]]) pins each shifted window to
the tank's own tile, so the four windows tile the 31x31 block around
the tank -- ~961 tiles for four extras, zero movement, zero fuel.
Movement between scans slides later windows off the grid, so the
sweep is ATOMIC: it continues only while the tank stands exactly on
its anchor tile, and any movement abandons it (the remaining
quadrants simply stop qualifying until a fresh block).

Harvest follows: revealed containers OUTSIDE the current window but
inside the block are served by a free scope shift that frames them
(the shifted window IS the acceptance window -- archive-proven
2026-08-06, twelve accepted shift-framed actions), after which the
ordinary in-window pickup branches take over. Targets beyond a single
shift's reach get a walk leg toward them first (shift-before-walk,
per leg). When neither recon nor harvest has work left, the cascade
falls through to the exit hop as before.
"""

from __future__ import annotations

from typing import Literal

from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE, is_container_blacklisted
from tankpit_bot.bot.ai.collect_pickups import pickup_not_worth_walk
from tankpit_bot.bot.ai.context import (
    RADAR_RESERVE_EXTRAS,
    RADAR_RESERVE_REVEAL_FLOOR_TILES,
    RADAR_SPEND_REVEAL_FLOOR_TILES,
    DecideCtx,
    clear_resource_target,
    make_decision,
)
from tankpit_bot.bot.ai.equipment import is_container_pursuable
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.reachability import is_collection_reachable_within_bounds
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    BotCommand,
    make_radar_command,
    make_scope_shift_command,
)
from tankpit_bot.inventory import inventory_counts
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.supervisor import equipment_pickup_refusal
from tankpit_bot.protocol.commands import (
    SCOPE_EAST,
    SCOPE_NORTH,
    SCOPE_NORTHEAST,
    SCOPE_NORTHWEST,
    SCOPE_SOUTH,
    SCOPE_SOUTHEAST,
    SCOPE_SOUTHWEST,
    SCOPE_WEST,
)
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.scan_coverage import viewport_uncovered_count
from tankpit_bot.state.types import ContainerStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

_WINDOW_LAST = 15
"""Offset of a window's far edge from its origin (16x16, inclusive)."""

_MAP_LAST_ORIGIN = 255 - _WINDOW_LAST
"""Largest window origin the 256x256 map allows on either axis."""

_QUADRANTS: tuple[tuple[int, int, int], ...] = (
    (SCOPE_NORTHWEST, -_WINDOW_LAST, -_WINDOW_LAST),
    (SCOPE_NORTHEAST, 0, -_WINDOW_LAST),
    (SCOPE_SOUTHEAST, 0, 0),
    (SCOPE_SOUTHWEST, -_WINDOW_LAST, 0),
)
"""Sweep order: scope direction and window-origin offset from the tank.

The offsets ARE the anchor law: a diagonal shift pins the tank to the
opposite corner of the shifted window, so NW parks the origin at
``(tank-15, tank-15)`` and SE at the tank's own tile.
"""

SWEEP_START_FLOOR_TILES = 480
"""Uncovered tiles the 31x31 block must hold to START a sweep.

Roughly half the block. A fresh landing (only the centered landing
scan spent) sits near ~700 uncovered and starts; harvest walking that
drags the block a handful of tiles exposes ~31 per tile and stays far
below the floor, so a sweep never re-fires mid-harvest -- the block
must be substantially virgin ground again before four more extras go
into it. Continuation is NOT gated by this floor: an in-progress
sweep (tank still on its anchor tile) keeps going on the per-quadrant
economics alone.
"""

SweepReason = Literal["quad_sweep_radar", "quad_sweep_shift"]
"""The two decision kinds a sweep tick can produce."""

BLOCK_REACH_TILES = 2 * _WINDOW_LAST + 1
"""Farthest Chebyshev distance a swept block's tile can sit from the
tank once harvest walking has dragged the tank to the block's far
edge (31: a full block diameter)."""

_LEG_PATH_MARGIN_TILES = 4
"""Slack added around the tank->target bounding box when checking
block reachability, so a path may bow around obstacles that sit
exactly on the straight corridor."""


def quadrant_bounds(sx: int, sy: int, offset_x: int, offset_y: int) -> tuple[int, int, int, int]:
    """Return one quadrant window's inclusive map-clamped bounds.

    Args:
        sx: Tank X (the sweep anchor).
        sy: Tank Y (the sweep anchor).
        offset_x: Window-origin X offset from the tank (anchor law).
        offset_y: Window-origin Y offset from the tank (anchor law).

    Returns:
        Inclusive ``(left, top, right, bottom)`` of the shifted window,
        clamped exactly like the server clamps every window origin.
    """
    left = min(max(sx + offset_x, 0), _MAP_LAST_ORIGIN)
    top = min(max(sy + offset_y, 0), _MAP_LAST_ORIGIN)
    return (left, top, left + _WINDOW_LAST, top + _WINDOW_LAST)


def _quadrant_spend_worthwhile(uncovered: int, extras: int) -> bool:
    """Return True when one quadrant scan is worth an extra radar.

    Mirrors the shared radar-spend economics
    (:func:`~tankpit_bot.bot.ai.context.radar_spend_worthwhile`)
    evaluated against the QUADRANT window instead of the current one.
    The sweep is an extras strategy by definition -- the free radar's
    5x5 cannot tile a block -- so zero extras never qualifies.

    Args:
        uncovered: Uncovered tile count inside the quadrant window.
        extras: Extra radars currently stocked.

    Returns:
        True when the spend clears the applicable reveal floor.
    """
    if extras > RADAR_RESERVE_EXTRAS:
        return uncovered >= RADAR_SPEND_REVEAL_FLOOR_TILES
    if extras > 0:
        return uncovered >= RADAR_RESERVE_REVEAL_FLOOR_TILES
    return False


def _anchored_state(base_state: AIStateDict, sx: int, sy: int) -> AIStateDict:
    """Return the AI state latched to the sweep anchor tile.

    Args:
        base_state: Base AI state to rewrite.
        sx: Anchor X (the tank's current tile).
        sy: Anchor Y (the tank's current tile).

    Returns:
        State with the anchor latched and any resource target cleared.
    """
    cleared = clear_resource_target(base_state)
    return AIStateDict(**{**cleared, "sweep_anchor_x": sx, "sweep_anchor_y": sy})


def plan_quad_sweep(ctx: DecideCtx, base_state: AIStateDict) -> TickDecisionDict | None:
    """Plan the next atomic sweep step: a quadrant shift or its radar.

    Runs BEFORE the pickup branches: the doctrine's atomicity rule is
    that pickups wait the ~8 ticks a full sweep takes, because any
    movement re-anchors later quadrants off the grid. Safety gates
    (landing scan, under-fire escape, desync rescan) and lock
    continuation still run first in the cascade.

    Anchor discipline: a sweep STARTS (latching ``sweep_anchor_*`` to
    the tank tile) only when the block holds at least
    :data:`SWEEP_START_FLOOR_TILES` uncovered tiles, and CONTINUES
    only while the tank still stands on its latched anchor. Any
    movement -- harvest legs, pickups, an escape -- silently abandons
    the remainder; the block gate keeps the next sweep from firing
    until the surroundings are substantially fresh again.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        The next sweep decision, or ``None`` when no quadrant scan is
        currently worth an extra (sweep complete, mid-harvest, or
        extras exhausted).
    """
    extras = ctx.inventory["extra_radars"]["count"]
    if extras == 0:
        return None
    if ctx.fuel <= ctx.config["fuel_low_threshold"]:
        # Recon is an economy move, never a survival move: at or below
        # the fuel-low break every tick belongs to fuel acquisition
        # (visible pickups, the dot hop, the walk rescue) -- an
        # 8-tick stationary sweep is exactly the exposure the
        # never-leave-the-tank-exposed contract forbids.
        return None
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    anchored = base_state["sweep_anchor_x"] == sx and base_state["sweep_anchor_y"] == sy
    if not anchored:
        block_left = max(sx - _WINDOW_LAST, 0)
        block_top = max(sy - _WINDOW_LAST, 0)
        block_uncovered = viewport_uncovered_count(
            ctx.world["scanned_tiles"],
            block_left,
            block_top,
            min(sx + _WINDOW_LAST, 255),
            min(sy + _WINDOW_LAST, 255),
            ctx.timestamp_ms,
        )
        if block_uncovered < SWEEP_START_FLOOR_TILES:
            return None
    window = viewport_visible_bounds(ctx.world["viewport"])
    pending = _pending_quadrants(ctx, sx, sy, extras)
    if not pending:
        return None
    for direction, bounds, uncovered in pending:
        if window != bounds:
            continue
        emit_ai(
            "quad sweep radar: quadrant dir=%d window=(%d,%d) uncovered=%d extras=%d",
            direction,
            bounds[0],
            bounds[1],
            uncovered,
            extras,
        )
        return _sweep_decision(ctx, base_state, sx, sy, "quad_sweep_radar", direction, uncovered)
    # No pending quadrant is framed yet. Scan the CURRENT window first
    # when it still clears the spend floor (self-correcting: fresh
    # ground under the window is worth the extra wherever the scope
    # actually parked), then steer toward the first pending quadrant.
    window_uncovered = viewport_uncovered_count(
        ctx.world["scanned_tiles"],
        window[0],
        window[1],
        window[2],
        window[3],
        ctx.timestamp_ms,
    )
    if _quadrant_spend_worthwhile(window_uncovered, extras):
        emit_ai(
            "quad sweep radar: current window (%d,%d) still fresh (uncovered=%d)",
            window[0],
            window[1],
            window_uncovered,
        )
        return _sweep_decision(ctx, base_state, sx, sy, "quad_sweep_radar", -1, window_uncovered)
    direction, bounds, uncovered = pending[0]
    emit_ai(
        "quad sweep shift: dir=%d toward window (%d,%d) uncovered=%d",
        direction,
        bounds[0],
        bounds[1],
        uncovered,
    )
    return _sweep_decision(ctx, base_state, sx, sy, "quad_sweep_shift", direction, uncovered)


def _pending_quadrants(
    ctx: DecideCtx,
    sx: int,
    sy: int,
    extras: int,
) -> list[tuple[int, tuple[int, int, int, int], int]]:
    """Return the quadrants still worth an extra, in sweep order.

    Args:
        ctx: Decision context.
        sx: Anchor X (the tank's current tile).
        sy: Anchor Y (the tank's current tile).
        extras: Extra radars currently stocked.

    Returns:
        ``(direction, bounds, uncovered)`` per qualifying quadrant.
    """
    pending: list[tuple[int, tuple[int, int, int, int], int]] = []
    for direction, offset_x, offset_y in _QUADRANTS:
        bounds = quadrant_bounds(sx, sy, offset_x, offset_y)
        uncovered = viewport_uncovered_count(
            ctx.world["scanned_tiles"],
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
            ctx.timestamp_ms,
        )
        if _quadrant_spend_worthwhile(uncovered, extras):
            pending.append((direction, bounds, uncovered))
    return pending


def _sweep_decision(
    ctx: DecideCtx,
    base_state: AIStateDict,
    sx: int,
    sy: int,
    reason: SweepReason,
    direction: int,
    uncovered: int,
) -> TickDecisionDict:
    """Build one sweep decision (a quadrant radar or a steering shift).

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite.
        sx: Anchor X.
        sy: Anchor Y.
        reason: The sweep reason kind.
        direction: Scope direction byte (-1 for the current-window
            radar, which needed no steer).
        uncovered: Uncovered tile count the spend is buying.

    Returns:
        The complete sweep decision, anchor latched.
    """
    command: BotCommand = (
        make_radar_command()
        if reason == "quad_sweep_radar"
        else make_scope_shift_command(direction)
    )
    return make_decision(
        command,
        "COLLECT",
        COLLECT_SCORE,
        sx,
        sy,
        reason,
        _anchored_state(base_state, sx, sy),
        ctx.equip,
        reason_context={"direction": direction, "uncovered": uncovered},
    )


def anchored_window_origin(
    window_left: int,
    window_top: int,
    sx: int,
    sy: int,
    direction: int,
) -> tuple[int, int]:
    """Return the window origin a scope shift will anchor to.

    The measured Rb anchor law ([[viewport-shift-protocol]]): an
    eastward component pins ``left = tank_x``, a westward one
    ``left = tank_x - 15``; south/north pin ``top`` the same way; an
    axis the direction does not name keeps its current origin.
    Direction 8 (Scope Center) recenters like a teleport landing.
    Map-clamped like every window origin.

    Args:
        window_left: Current stored window origin X.
        window_top: Current stored window origin Y.
        sx: Self X.
        sy: Self Y.
        direction: Compass byte, clockwise from north (0=N..7=NW),
            or 8 for center.

    Returns:
        The clamped ``(left, top)`` the server's 0x5A will state.
    """
    if direction == 8:
        left, top = sx - 8, sy - 8
    else:
        left, top = window_left, window_top
        if direction in (1, 2, 3):
            left = sx
        elif direction in (5, 6, 7):
            left = sx - _WINDOW_LAST
        if direction in (3, 4, 5):
            top = sy
        elif direction in (7, 0, 1):
            top = sy - _WINDOW_LAST
    return (
        min(max(left, 0), _MAP_LAST_ORIGIN),
        min(max(top, 0), _MAP_LAST_ORIGIN),
    )


def frame_direction(
    window: tuple[int, int, int, int],
    sx: int,
    sy: int,
    goal_x: int,
    goal_y: int,
) -> int | None:
    """Pick the scope direction whose anchored window moves toward a goal.

    The anchor-law compass without the ferry scout's single-pan reach
    cap: a goal beyond 15 tiles still deserves the shift (the window
    extends its full 15 toward the goal, opening walk room for the
    leg that follows). The no-op test applies the server's map clamp,
    so a shift that cannot change the window near a map edge answers
    ``None`` instead of dispatching forever.

    Args:
        window: Inclusive current window bounds (left, top, right,
            bottom) -- the stored 0x5A window.
        sx: Self X.
        sy: Self Y.
        goal_x: Goal tile X.
        goal_y: Goal tile Y.

    Returns:
        The ``SCOPE_*`` direction byte, or ``None`` when the goal
        shares the tank's tile or the anchored window IS the current
        window (the shift would reveal nothing).
    """
    east = goal_x > sx
    west = goal_x < sx
    south = goal_y > sy
    north = goal_y < sy
    if not (east or west or south or north):
        return None
    if north:
        direction = SCOPE_NORTHEAST if east else SCOPE_NORTHWEST if west else SCOPE_NORTH
    elif south:
        direction = SCOPE_SOUTHEAST if east else SCOPE_SOUTHWEST if west else SCOPE_SOUTH
    else:
        direction = SCOPE_EAST if east else SCOPE_WEST
    if anchored_window_origin(window[0], window[1], sx, sy, direction) == (
        window[0],
        window[1],
    ):
        return None
    return direction


def _harvest_candidates(ctx: DecideCtx) -> list[ContainerStateDict]:
    """Return block containers worth a framing shift, nearest first.

    A candidate is a pursuable, unblacklisted container OUTSIDE the
    current window but within :data:`BLOCK_REACH_TILES` Chebyshev of
    the tank, whose kind the tank can still absorb (equipment refused
    at full inventory, fuel skipped at cap and priced by the shared
    worth-the-walk rate), and that the composed terrain can serve
    (a collection path inside the tank->target bounding box, margin
    :data:`_LEG_PATH_MARGIN_TILES` -- window anchoring lets a
    leg-by-leg walk follow any such path). Without a terrain view the
    reachability gate is skipped -- the same assume-reachable stance
    the pickup candidate search takes, so a terrain-less session
    (seam boots, early live ticks) still harvests instead of hopping
    away from tracked block stock.

    Args:
        ctx: Decision context.

    Returns:
        Qualifying containers ordered by Manhattan distance.
    """
    terrain = ctx.terrain
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    equipment_wanted = (
        equipment_pickup_refusal(inventory_counts(ctx.inventory), ctx.self_state["rank"]) is None
    )
    fuel_wanted = ctx.fuel < fuel_capacity(ctx.self_state["rank"])
    ranked: list[tuple[int, ContainerStateDict]] = []
    for container in ctx.filtered["containers"].values():
        want_fuel = container["is_fuel"]
        if want_fuel and (not fuel_wanted or container["volume"] < 1):
            continue
        if not want_fuel and not equipment_wanted:
            continue
        if not is_container_pursuable(container, want_fuel=want_fuel):
            continue
        cx, cy = container["x"], container["y"]
        if is_container_blacklisted(cx, cy):
            continue
        if left <= cx <= right and top <= cy <= bottom:
            continue
        if max(abs(cx - sx), abs(cy - sy)) > BLOCK_REACH_TILES:
            continue
        if want_fuel and pickup_not_worth_walk(ctx, container):
            continue
        if terrain is not None and not is_collection_reachable_within_bounds(
            terrain,
            sx,
            sy,
            cx,
            cy,
            left=max(min(sx, cx) - _LEG_PATH_MARGIN_TILES, 0),
            top=max(min(sy, cy) - _LEG_PATH_MARGIN_TILES, 0),
            right=min(max(sx, cx) + _LEG_PATH_MARGIN_TILES, 255),
            bottom=min(max(sy, cy) + _LEG_PATH_MARGIN_TILES, 255),
        ):
            continue
        ranked.append((abs(cx - sx) + abs(cy - sy), container))
    ranked.sort(key=_candidate_distance)
    return [container for _, container in ranked]


def _candidate_distance(entry: tuple[int, ContainerStateDict]) -> int:
    """Return the Manhattan distance a ranked harvest entry carries."""
    return entry[0]


def plan_block_harvest_leg(ctx: DecideCtx, base_state: AIStateDict) -> TickDecisionDict | None:
    """Frame the nearest block container, or walk a leg toward it.

    Runs after the in-window pickup branches decline: whatever the
    current window held is taken (or refused), so the next harvest
    target sits outside it. Per leg, shift-before-walk: a scope shift
    that moves the window toward the target is free and comes first
    (once framed, next tick's ordinary pickup branch dispatches); only
    when the window is already anchored toward the target does the
    tick spend a walk (the movement layer's off-viewport approach --
    edge-tile walk, teleport only when no walkable edge serves).

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        The framing-shift or leg-walk decision, or ``None`` when no
        qualifying block container remains (harvest done -- the
        cascade proceeds to larder and the exit hop).
    """
    if ctx.fuel <= ctx.config["fuel_low_threshold"]:
        # Same rule as the sweep: at or below the fuel-low break the
        # desperation ladder (dot hop, walk-for-fuel rescue) owns the
        # tick -- its distance caps and cost ranking exist precisely
        # for that regime, and a framing shift toward far block stock
        # would preempt them with an unpriced walk.
        return None
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    window = viewport_visible_bounds(ctx.world["viewport"])
    for container in _harvest_candidates(ctx):
        cx, cy = container["x"], container["y"]
        direction = frame_direction(window, sx, sy, cx, cy)
        if direction is not None:
            emit_ai(
                "harvest frame shift dir=%d toward %s at (%d,%d)",
                direction,
                "fuel" if container["is_fuel"] else "equipment",
                cx,
                cy,
            )
            return make_decision(
                make_scope_shift_command(direction),
                "COLLECT",
                COLLECT_SCORE,
                cx,
                cy,
                "harvest_frame_shift",
                clear_resource_target(base_state),
                ctx.equip,
                reason_context={"direction": direction},
            )
        command = walk_or_teleport(ctx, cx, cy, pickup_kind=None)
        if command is None:
            continue
        emit_ai(
            "harvest leg toward %s at (%d,%d)",
            "fuel" if container["is_fuel"] else "equipment",
            cx,
            cy,
        )
        return make_decision(
            command,
            "COLLECT",
            COLLECT_SCORE,
            cx,
            cy,
            "harvest_leg_walk",
            clear_resource_target(base_state),
            ctx.equip,
        )
    return None


__all__ = [
    "BLOCK_REACH_TILES",
    "SWEEP_START_FLOOR_TILES",
    "frame_direction",
    "plan_block_harvest_leg",
    "plan_quad_sweep",
    "quadrant_bounds",
]
