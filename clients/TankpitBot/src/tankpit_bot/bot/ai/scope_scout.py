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

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE
from tankpit_bot.bot.ai.context import DecideCtx, make_decision
from tankpit_bot.bot.ai.equipment_search import (
    find_all_tracked_equipment,
    find_teleport_landing_tile,
)
from tankpit_bot.bot.ai.ferry_landing import (
    FERRY_SEARCH_RADIUS,
    find_ferry_boarding_tile,
    goal_water_pond,
)
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
from tankpit_bot.runtime_logging import (
    emit_ai,
    emit_diagnostic,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

SCOPE_SCOUT_COOLDOWN_MS = 30000
"""Minimum quiet time between ferry scope scouts. Ferry memory is
positional, not clocked ([[ferry-mechanics]] no-drift law) — the pan
gambles only on water the bot has never looked at, or on a human
having ridden a ferry in since the last look. Both are rare, so the
scout looks at most once per half-minute instead of once per tick."""

SCOPE_REACH_TILES = 15
"""Farthest tile a single scope pan can bring into view: the anchor
law pins the tank to the trailing window edge, so the shifted 16x16
window extends exactly 15 tiles from the tank in the requested
direction (wire-measured 2026-08-01, [[viewport-shift-protocol]])."""


def pan_plan_toward(
    window: tuple[int, int, int, int],
    sx: int,
    sy: int,
    goal_x: int,
    goal_y: int,
) -> tuple[int, int, int] | None:
    """Compass byte plus anchored origin for a pan toward a goal.

    The direction is the tank→goal compass sign; the anchored origin
    is where the window's (left, top) lands under the measured anchor
    law ([[viewport-shift-protocol]]): the server pins the tank to the
    trailing edge, so the shifted window extends
    :data:`SCOPE_REACH_TILES` from the tank in the requested
    direction. No reach cap here — a goal beyond one pan still names
    the direction that reveals the next window of route toward it
    (the marooned walk-for-fuel gait's case; the ferry scout adds its
    own cap in :func:`scope_direction_toward`).

    Args:
        window: Inclusive current window bounds (left, top, right,
            bottom) — the stored 0x5A window.
        sx: Self X.
        sy: Self Y.
        goal_x: Goal tile X.
        goal_y: Goal tile Y.

    Returns:
        ``(direction, anchored_left, anchored_top)``, or ``None`` when
        the goal IS the tank's own tile (no axis to pan) or the
        anchored window IS the current window (the pan would reveal
        nothing — e.g. right after a previous pan the same way).
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
    return direction, shifted_left, shifted_top


def pan_reveals_new_goal_water(
    terrain: TerrainMapProtocol,
    window: tuple[int, int, int, int],
    shifted_left: int,
    shifted_top: int,
    goal_x: int,
    goal_y: int,
) -> bool:
    """Return True when the anchored pan uncovers unseen ferry water.

    The pan precheck (operator flag 2, run bot-20260831-152132
    15:22:52: pan direction 3, no ferry, frontier teleport one tick
    later — "it did a viewport shift then it teleported immediately").
    Water already inside the current window is DEFINITIVELY ferry-less
    right now — the live 0x5A stream would have delivered any ferry on
    it — so a pan is informative only when the anchored window covers
    at least one tile of the goal's own pond, within the ferry search
    radius, that the current window does not. Terrain is map-wide
    static, so this is decidable before spending the tick.

    Args:
        terrain: Static terrain of the current field.
        window: Inclusive current window bounds (left, top, right,
            bottom) — the stored 0x5A window.
        shifted_left: Anchored window left from :func:`pan_plan_toward`.
        shifted_top: Anchored window top from :func:`pan_plan_toward`.
        goal_x: Water-locked goal X.
        goal_y: Water-locked goal Y.

    Returns:
        True when the pan can show boarding-candidate water the bot is
        not already looking at.
    """
    left, top, right, bottom = window
    shifted_right = shifted_left + (right - left)
    shifted_bottom = shifted_top + (bottom - top)
    for water_x, water_y in goal_water_pond(terrain, goal_x, goal_y):
        if max(abs(water_x - goal_x), abs(water_y - goal_y)) > FERRY_SEARCH_RADIUS:
            continue
        in_shifted = (
            shifted_left <= water_x <= shifted_right and shifted_top <= water_y <= shifted_bottom
        )
        if not in_shifted:
            continue
        if left <= water_x <= right and top <= water_y <= bottom:
            continue
        return True
    return False


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
        if container["is_fuel"] and container["volume"] > 0 and container["failed_pickups"] == 0
    ]
    candidates.extend(
        (equipment["x"], equipment["y"]) for equipment in find_all_tracked_equipment(ctx.world)
    )
    for goal_x, goal_y in candidates:
        if find_teleport_landing_tile(terrain, goal_x, goal_y) is not None:
            continue
        boarding = find_ferry_boarding_tile(ctx.world, terrain, goal_x, goal_y)
        if boarding is not None:
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
    terrain = ctx.terrain
    if terrain is None:
        return None
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    window = viewport_visible_bounds(ctx.world["viewport"])
    best: tuple[int, int, int, int] | None = None
    for candidate_x, candidate_y in _water_locked_goals(ctx):
        dist = max(abs(candidate_x - sx), abs(candidate_y - sy))
        if dist > SCOPE_REACH_TILES:
            continue
        plan = pan_plan_toward(window, sx, sy, candidate_x, candidate_y)
        if plan is None:
            continue
        candidate_direction, shifted_left, shifted_top = plan
        if not pan_reveals_new_goal_water(
            terrain, window, shifted_left, shifted_top, candidate_x, candidate_y
        ):
            continue
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
    "pan_plan_toward",
    "pan_reveals_new_goal_water",
    "scope_scout_for_ferry",
]
