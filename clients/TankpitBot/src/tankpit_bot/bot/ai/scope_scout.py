"""Free viewport pan toward water-locked goals — the ferry scope scout.

User doctrine (2026-07-30, queued with the F5 larder work): "we want
ferries... technically we could just use a viewport shift" — look at
the water LIVE before boarding instead of teleporting on a stale
belief or writing the container off. The wire tool is the ``Rb``
scope-extend command ([[viewport-shift-protocol]]): free on every
axis (no fuel, no queue slot), answered by a ``0x5A`` whose patches
carry any ferry in the shifted window (wire terrain 5,
[[ferry-mechanics]]) straight into the terrain beliefs
``find_ferry_boarding_tile`` reads.

The measured anchor law bounds what a pan can show: the server pins
the tank to the trailing window edge, so the reachable view is the
31x31 area centered on the tank — a scout can vouch for water near
the CURRENT position, never across the map. Goals beyond
:data:`SCOPE_REACH_TILES` stay the discovery cascade's business.

A pan that reveals no ferry leaves no negative belief behind (0x5A
patches enumerate the dynamic layer present, never its absence), so
the scout latches ``last_scope_scout_ms`` and holds off for
:data:`SCOPE_SCOUT_COOLDOWN_MS` — without the latch the same declined
water-locked container would re-trigger the pan every tick.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE, is_container_blacklisted
from tankpit_bot.bot.ai.context import DecideCtx, make_decision
from tankpit_bot.bot.ai.equipment_search import (
    find_all_tracked_equipment,
    find_teleport_landing_tile,
)
from tankpit_bot.bot.ai.ferry_landing import find_ferry_boarding_tile
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_scope_shift_command
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
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

SCOPE_SCOUT_COOLDOWN_MS = 30000
"""Minimum quiet time between ferry scope scouts. Half the 60 s ferry
belief TTL (unbracketed, like the TTL itself): often enough that a
ferry drifting into the water near a stalled larder goal is noticed
within a belief lifetime, rare enough that a ferry-less lake costs
one free tick per half-minute instead of one per tick."""

SCOPE_REACH_TILES = 15
"""Farthest tile a single scope pan can bring into view: the anchor
law pins the tank to the trailing window edge, so the shifted 16x16
window extends exactly 15 tiles from the tank in the requested
direction (wire-measured 2026-08-01, [[viewport-shift-protocol]])."""


def scope_direction_toward(
    window: tuple[int, int, int, int],
    sx: int,
    sy: int,
    goal_x: int,
    goal_y: int,
) -> int | None:
    """Pick the compass byte that pans the window at a goal's water.

    The direction is the tank→goal compass sign, NOT a window test:
    the goal container is usually already in view (that is how it got
    radar-believed), and what the pan must reveal is the ferry search
    water AROUND and BEYOND it. Anchoring the window to the tank in
    the goal's direction shows the most of that water a single pan
    can ([[viewport-shift-protocol]] anchor law).

    Args:
        window: Inclusive current window bounds (left, top, right,
            bottom) — the stored 0x5A window.
        sx: Self X.
        sy: Self Y.
        goal_x: Goal tile X.
        goal_y: Goal tile Y.

    Returns:
        The ``SCOPE_*`` direction byte, or ``None`` when the goal is
        beyond :data:`SCOPE_REACH_TILES` (no single pan can serve it)
        or the anchored window IS the current window (the pan would
        reveal nothing — e.g. right after a previous scout the same
        way).
    """
    if max(abs(goal_x - sx), abs(goal_y - sy)) > SCOPE_REACH_TILES:
        return None
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
    left, top = window[0], window[1]
    shifted_left, shifted_top = left, top
    if east:
        shifted_left = sx
    elif west:
        shifted_left = sx - SCOPE_REACH_TILES
    if south:
        shifted_top = sy
    elif north:
        shifted_top = sy - SCOPE_REACH_TILES
    if (shifted_left, shifted_top) == (left, top):
        return None
    return direction


def _water_locked_goals(ctx: DecideCtx) -> list[tuple[int, int]]:
    """Collect goal tiles that only a ferry could serve right now.

    A goal qualifies when it has no legal teleport landing (open
    water on every side) AND no fresh believed ferry within boarding
    range — exactly the larder's ``no_landing`` decline, recomputed
    with the same helpers so the scout and the hop can never disagree
    about what is stuck.

    Args:
        ctx: Decision context (terrain narrowed non-None by caller).

    Returns:
        Water-locked goal tiles, unordered.
    """
    terrain = ctx.terrain
    assert terrain is not None  # caller guarantees this
    goals: list[tuple[int, int]] = []
    candidates: list[tuple[int, int]] = [
        (container["x"], container["y"])
        for container in ctx.world["containers"].values()
        if container["is_fuel"]
        and container["volume"] > 0
        and container["failed_pickups"] == 0
        and not is_container_blacklisted(container["x"], container["y"])
    ]
    candidates.extend(
        (equipment["x"], equipment["y"]) for equipment in find_all_tracked_equipment(ctx.world)
    )
    for goal_x, goal_y in candidates:
        if find_teleport_landing_tile(terrain, goal_x, goal_y) is not None:
            continue
        if find_ferry_boarding_tile(ctx.world, goal_x, goal_y, ctx.timestamp_ms) is not None:
            continue
        goals.append((goal_x, goal_y))
    return goals


def scope_scout_for_ferry(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Spend one free tick panning the viewport at a water-locked goal.

    Runs after the larder declines and before discovery: the pan
    costs nothing, so when a believed container is stuck behind the
    ``no_landing`` tally and sits inside scope reach, one look at the
    water is strictly cheaper than a discovery teleport. A ferry the
    pan reveals arrives as a 0x5A terrain-5 patch, and the NEXT
    tick's larder serves the same container ``ferry_served``.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        The scope-shift decision, or ``None`` when a combat lock is
        held (mid-fight restocks never sightsee), the cooldown is
        live, terrain is unknown, or no water-locked goal is inside
        a single pan's reach.
    """
    if base_state["combat_target_id"] != -1:
        return None
    if ctx.timestamp_ms - base_state["last_scope_scout_ms"] < SCOPE_SCOUT_COOLDOWN_MS:
        return None
    if ctx.terrain is None:
        return None
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    window = viewport_visible_bounds(ctx.world["viewport"])
    best: tuple[int, int, int, int] | None = None
    for candidate_x, candidate_y in _water_locked_goals(ctx):
        candidate_direction = scope_direction_toward(window, sx, sy, candidate_x, candidate_y)
        if candidate_direction is None:
            continue
        dist = max(abs(candidate_x - sx), abs(candidate_y - sy))
        if best is None or dist < best[0]:
            best = (dist, candidate_direction, candidate_x, candidate_y)
    if best is None:
        return None
    _, direction, goal_x, goal_y = best
    emit_ai(
        "scope scout toward water-locked goal (%d,%d): pan direction %d",
        goal_x,
        goal_y,
        direction,
    )
    emit_diagnostic(
        diagnostic_kind="ferry_scope_scout",
        goal_x=goal_x,
        goal_y=goal_y,
        direction=direction,
    )
    return make_decision(
        make_scope_shift_command(direction),
        "COLLECT",
        COLLECT_SCORE,
        goal_x,
        goal_y,
        "ferry_scope_scout",
        AIStateDict(**{**base_state, "last_scope_scout_ms": ctx.timestamp_ms}),
        ctx.equip,
        reason_context={"direction": direction},
    )


__all__ = [
    "SCOPE_REACH_TILES",
    "SCOPE_SCOUT_COOLDOWN_MS",
    "scope_direction_toward",
    "scope_scout_for_ferry",
]
