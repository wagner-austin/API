"""Block harvest: anchor-law framing shifts and walk legs over known stock.

The harvest half of the quad-sweep doctrine ([[quad-sweep-doctrine]]):
revealed containers OUTSIDE the current window but inside the 31x31
block are served by a free scope shift that frames them (the shifted
window IS the acceptance window -- archive-proven 2026-08-06), after
which the ordinary in-window pickup branches take over. Targets beyond
a single shift's reach get a walk leg toward them first
(shift-before-walk, per leg). This module also owns the anchor-law
window geometry (:func:`anchored_window_origin`,
:func:`frame_direction`).
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE
from tankpit_bot.bot.ai.collect_pickups import pickup_not_worth_walk
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    make_decision,
)
from tankpit_bot.bot.ai.equipment import is_container_pursuable
from tankpit_bot.bot.ai.intent import set_resource_target
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.reachability import is_collection_reachable_within_bounds
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_scope_shift_command
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
from tankpit_bot.state.types import ContainerStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

WINDOW_LAST = 15
"""Offset of a window's far edge from its origin (16x16, inclusive)."""

MAP_LAST_ORIGIN = 255 - WINDOW_LAST
"""Largest window origin the 256x256 map allows on either axis."""

BLOCK_REACH_TILES = 2 * WINDOW_LAST + 1
"""Farthest Chebyshev distance a swept block's tile can sit from the
tank once harvest walking has dragged the tank to the block's far
edge (31: a full block diameter)."""

_LEG_PATH_MARGIN_TILES = 4
"""Slack added around the tank->target bounding box when checking
block reachability, so a path may bow around obstacles that sit
exactly on the straight corridor."""


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
            left = sx - WINDOW_LAST
        if direction in (3, 4, 5):
            top = sy
        elif direction in (7, 0, 1):
            top = sy - WINDOW_LAST
    return (
        min(max(left, 0), MAP_LAST_ORIGIN),
        min(max(top, 0), MAP_LAST_ORIGIN),
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


def _wanted_block_container(
    ctx: DecideCtx,
    container: ContainerStateDict,
    *,
    equipment_wanted: bool,
    fuel_wanted: bool,
    sx: int,
    sy: int,
) -> bool:
    """Return True when a container is wanted, pursuable block stock.

    The harvest ranking's qualification: the kind must still be
    absorbable (equipment refused at full inventory, fuel at cap),
    the container pursuable, within :data:`BLOCK_REACH_TILES` of the
    tank, and -- for fuel -- priced by the shared worth-the-walk rate
    so a near-cap sliver never counts as stock.

    Args:
        ctx: Decision context.
        container: Candidate container.
        equipment_wanted: Whether any equipment slot can still absorb.
        fuel_wanted: Whether the tank is below fuel capacity.
        sx: Self X.
        sy: Self Y.

    Returns:
        True when the container qualifies as block stock.
    """
    want_fuel = container["is_fuel"]
    if want_fuel and (not fuel_wanted or container["volume"] < 1):
        return False
    if not want_fuel and not equipment_wanted:
        return False
    if not is_container_pursuable(container, want_fuel=want_fuel):
        return False
    if max(abs(container["x"] - sx), abs(container["y"] - sy)) > BLOCK_REACH_TILES:
        return False
    if ctx.ws.is_move_target_failed(container["x"], container["y"], ctx.timestamp_ms):
        # The candidate filter must agree with the lock's release rule
        # (flag s11-5, 2026-08-13): the server refused movement to
        # (165,161), the structural move-failed release correctly
        # dropped the lock -- and this filter's absence re-latched the
        # same tile as "nearest block stock" one tick later, forever
        # (release -> re-latch -> release, one free scope shift per
        # cycle, window ping-ponging dir 3 <-> dir 7 at full fuel).
        return False
    return not (want_fuel and pickup_not_worth_walk(ctx, container))


def _harvest_candidates(ctx: DecideCtx) -> list[ContainerStateDict]:
    """Return block containers worth a framing shift, nearest first.

    A candidate is a container :func:`_wanted_block_container` accepts,
    OUTSIDE the current window, that the composed terrain can serve
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
        if not _wanted_block_container(
            ctx,
            container,
            equipment_wanted=equipment_wanted,
            fuel_wanted=fuel_wanted,
            sx=sx,
            sy=sy,
        ):
            continue
        cx, cy = container["x"], container["y"]
        if left <= cx <= right and top <= cy <= bottom:
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
    if ctx.fuel <= ctx.fuel_low_floor:
        # Same rule as the sweep: at or below the fuel-low break the
        # desperation ladder (dot hop, walk-for-fuel rescue) owns the
        # tick -- its distance caps and cost ranking exist precisely
        # for that regime, and a framing shift toward far block stock
        # would preempt them with an unpriced walk.
        return None
    if base_state["resource_target_kind"] != "":
        # Committed intent (flag s11-5, 2026-08-13): a HELD lock that
        # was merely not executable this tick ("holding plan") fell
        # through to here, and the latch below overwrote it with a
        # different target -- an un-enumerated re-target the lock
        # design forbids. While a lock is held, its continuation owns
        # the pursuit; harvest only chooses targets on a free slate.
        return None
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    window = viewport_visible_bounds(ctx.world["viewport"])
    for container in _harvest_candidates(ctx):
        cx, cy = container["x"], container["y"]
        kind = "fuel" if container["is_fuel"] else "equipment"
        # The chosen target is LATCHED as a resource lock ([[committed-intent]],
        # HUD flags 2/5/7 and the flag-4 two-shift oscillator of
        # 2026-08-13 20:50): an uncommitted leg re-derived its target
        # every tick, and because each frame shift changes the window
        # that feeds the next derivation, two out-of-window containers
        # on opposite sides oscillate the scope forever with zero
        # movement. The lock-continuation step now owns the pursuit
        # until an ENUMERATED release fires (served, tank at capacity,
        # markedly closer candidate, structural move-failed mark, or
        # the unservable verdict) -- transient churn cannot.
        locked = set_resource_target(base_state, kind, cx, cy)
        direction = frame_direction(window, sx, sy, cx, cy)
        if direction is not None:
            emit_ai(
                "harvest frame shift dir=%d toward %s at (%d,%d)",
                direction,
                kind,
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
                locked,
                ctx.equip,
                reason_context={"direction": direction},
            )
        command = walk_or_teleport(ctx, cx, cy, pickup_kind=None)
        if command is None:
            continue
        emit_ai(
            "harvest leg toward %s at (%d,%d)",
            kind,
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
            locked,
            ctx.equip,
        )
    return None


__all__ = [
    "BLOCK_REACH_TILES",
    "MAP_LAST_ORIGIN",
    "WINDOW_LAST",
    "anchored_window_origin",
    "frame_direction",
    "plan_block_harvest_leg",
]
