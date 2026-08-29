"""The quad sweep: stationary 4-shift/4-radar recon over a virgin block.

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

Recon is need-driven (user ruling 2026-08-13, HUD flags 8/9/14):
known stock preempts scanning STRUCTURALLY -- the sweep sits BELOW
every collection branch in the COLLECT cascade, so it only wins a
tick when pickups, clearance, block harvest, the larder and the
ferry scout all declined. A mid-sweep reveal is therefore acted on
next tick (the movement aborts the sweep's remainder via the anchor
latch), which makes the sweep an INCREMENTAL scan-until-found rather
than an unconditional four-window ritual. The harvest half of the
doctrine lives in :mod:`~tankpit_bot.bot.ai.block_harvest`.
"""

from __future__ import annotations

from typing import Literal

from tankpit_bot.bot.ai.block_harvest import MAP_LAST_ORIGIN, WINDOW_LAST
from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE
from tankpit_bot.bot.ai.context import (
    RADAR_RESERVE_EXTRAS,
    RADAR_RESERVE_REVEAL_FLOOR_TILES,
    RADAR_SPEND_REVEAL_FLOOR_TILES,
    DecideCtx,
    make_decision,
)
from tankpit_bot.bot.ai.intent import clear_resource_target
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    BotCommand,
    make_radar_command,
    make_scope_shift_command,
)
from tankpit_bot.protocol.commands import (
    SCOPE_NORTHEAST,
    SCOPE_NORTHWEST,
    SCOPE_SOUTHEAST,
    SCOPE_SOUTHWEST,
)
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.scan_coverage import viewport_uncovered_count
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

_QUADRANTS: tuple[tuple[int, int, int], ...] = (
    (SCOPE_NORTHWEST, -WINDOW_LAST, -WINDOW_LAST),
    (SCOPE_SOUTHEAST, 0, 0),
    (SCOPE_NORTHEAST, 0, -WINDOW_LAST),
    (SCOPE_SOUTHWEST, -WINDOW_LAST, 0),
)
"""Sweep order: scope direction and window-origin offset from the tank.

The offsets ARE the anchor law: a diagonal shift pins the tank to the
opposite corner of the shifted window, so NW parks the origin at
``(tank-15, tank-15)`` and SE at the tank's own tile.

OPPOSITE corners first (NW then SE), not a compass circuit: two
adjacent quadrant windows share a 16-tile strip, while two opposite
ones overlap on a single tile -- so under the stop-on-found cascade
(a reveal is collected next tick and the sweep's remainder aborts)
the first two scans buy 511 unique tiles instead of 481. User-derived
2026-08-13 ("two corner scans, viewport all the way NW or SE").
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
    left = min(max(sx + offset_x, 0), MAP_LAST_ORIGIN)
    top = min(max(sy + offset_y, 0), MAP_LAST_ORIGIN)
    return (left, top, left + WINDOW_LAST, top + WINDOW_LAST)


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
    """Plan the next sweep step: a quadrant shift or its radar.

    Runs BELOW every collection branch in the cascade (user ruling
    2026-08-13, HUD flags 8/9/14 -- known stock preempts scanning),
    so a sweep tick means pickups, clearance, block harvest, larder
    and the ferry scout all declined: the block genuinely holds
    nothing known worth taking, and buying more information is the
    best remaining use of the tick.

    Anchor discipline: a sweep STARTS (latching ``sweep_anchor_*`` to
    the tank tile) only when the block holds at least
    :data:`SWEEP_START_FLOOR_TILES` uncovered tiles, and CONTINUES
    only while the tank still stands on its latched anchor. Any
    movement -- collecting a mid-sweep reveal, harvest legs, an
    escape -- silently abandons the remainder; the block gate keeps
    the next sweep from firing until the surroundings are
    substantially fresh again.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        The next sweep decision, or ``None`` when no quadrant scan is
        currently worth an extra (sweep complete, mid-harvest, or
        extras exhausted).
    """
    extras = ctx.inventory["extra_radars"]["count"]
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
        block_left = max(sx - WINDOW_LAST, 0)
        block_top = max(sy - WINDOW_LAST, 0)
        block_uncovered = viewport_uncovered_count(
            ctx.world["scanned_tiles"],
            block_left,
            block_top,
            min(sx + WINDOW_LAST, 255),
            min(sy + WINDOW_LAST, 255),
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
    # No pending quadrant is framed yet: steer toward the first one.
    # An earlier cut also radared the CURRENT window here whenever it
    # cleared the bare spend floor ("self-correcting"); measured live
    # (2026-08-13 run, HUD flags 4 and 5) that branch bought 32- and
    # 131-tile scraps of an already-87%-covered window for a radar +
    # 10 fuel each -- most of the session's 11 zero-yield scans. The
    # quadrant loop above already scans every window the sweep frames.
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
        direction: Scope direction byte.
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


__all__ = [
    "SWEEP_START_FLOOR_TILES",
    "plan_quad_sweep",
    "quadrant_bounds",
]
