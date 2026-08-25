"""Marooned walk-for-fuel recovery — the last rung before ``out_of_fuel``.

A tank at critical fuel with every hop unaffordable is not stuck while
known fuel exists: walking is free at any fuel level, and so is the
``Rb`` viewport pan ([[viewport-shift-protocol]]). The recovery gait
this module owns is **walk the window, then pan it**: legs walk toward
the nearest known fuel as far as the stored 16x16 window allows, and
when the window is exhausted (the leg clamps onto the tank's own tile)
a free scope pan anchors the window 15 fresh tiles in the fuel's
direction and the next leg walks the revealed ground.

The pan half is the lesson of run bot-20260825-133452 (Artax, entry
fuel 0): with autoscroll pinned OFF the window never moves on its own,
so the pre-pan walker shuttled 331 s between two candidates' clamp
tiles — fuel three tiles past the west edge was never reached, 74
successful moves netted zero progress, and the timer expired mid-walk.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE
from tankpit_bot.bot.ai.context import DecideCtx, make_decision
from tankpit_bot.bot.ai.intent import release_collect_plan
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.scope_scout import SCOPE_REACH_TILES, pan_plan_toward
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_scope_shift_command
from tankpit_bot.runtime_logging import (
    emit_ai,
    emit_diagnostic,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

WALK_FOR_FUEL_MAX_TILES = 48
"""Farthest known fuel a marooned tank will walk toward (~96 s of
2 s-per-tile walking). Beyond this the out_of_fuel exit stands -- the
2026-07-25 exposure rule caps how long a broke tank crawls in the
open, even in the practice room where bots never initiate."""


def _maroon_pan_toward(
    ctx: DecideCtx,
    base_state: AIStateDict,
    target_x: int,
    target_y: int,
) -> TickDecisionDict | None:
    """Pan the free viewport toward known fuel the window has exhausted.

    Reached when a walking leg clamps onto the tank's own tile: the
    window has been walked to its edge and the fuel is still outside
    it. The ``Rb`` scope shift costs nothing at any fuel and anchors
    the window to the tank in the fuel's direction (the measured
    anchor law, [[viewport-shift-protocol]]), so the very next leg
    walks up to :data:`~tankpit_bot.bot.ai.scope_scout.SCOPE_REACH_TILES`
    of freshly revealed ground.

    Two guards bound the free pans:

    * **Movement law**: a pan must pay for itself in movement — the
      dispatch position is latched (``maroon_pan_x``/``maroon_pan_y``)
      and no second pan fires from the exact latched tile, so a pan
      whose revealed ground turns out unwalkable falls through to the
      exit instead of ping-ponging the window between two stuck
      candidates on opposite sides.
    * **Terrain veto**: when the post-pan clamp tile is known
      impassable, the pan is refused up front and the candidate loop
      moves on — no free action is spent proving what the terrain map
      already knows.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.
        target_x: Known-fuel target X (outside the current window).
        target_y: Known-fuel target Y.

    Returns:
        The scope-shift decision, or ``None`` when the movement law,
        the anchor no-op check, or the terrain veto refuses (the
        caller continues its candidate scan).
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    if (sx, sy) == (base_state["maroon_pan_x"], base_state["maroon_pan_y"]):
        return None
    window = viewport_visible_bounds(ctx.world["viewport"])
    plan = pan_plan_toward(window, sx, sy, target_x, target_y)
    if plan is None:
        return None
    direction, shifted_left, shifted_top = plan
    post_leg_x = min(max(target_x, shifted_left), shifted_left + SCOPE_REACH_TILES)
    post_leg_y = min(max(target_y, shifted_top), shifted_top + SCOPE_REACH_TILES)
    terrain = ctx.terrain
    if terrain is not None and not terrain.is_passable(post_leg_x, post_leg_y):
        return None
    emit_ai(
        "marooned at fuel %d: panning direction %d toward known fuel at (%d,%d)",
        ctx.fuel,
        direction,
        target_x,
        target_y,
    )
    emit_diagnostic(
        diagnostic_kind="walk_for_fuel_pan",
        target_x=target_x,
        target_y=target_y,
        direction=direction,
        fuel=ctx.fuel,
    )
    return make_decision(
        make_scope_shift_command(direction),
        "COLLECT",
        COLLECT_SCORE,
        target_x,
        target_y,
        "walk_for_fuel_pan",
        AIStateDict(
            **{
                **release_collect_plan(base_state, reason="walk_for_fuel_override"),
                "maroon_pan_x": sx,
                "maroon_pan_y": sy,
            }
        ),
        ctx.equip,
        reason_context={"direction": direction},
    )


def walk_for_fuel_last_resort(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Walk toward the nearest known fuel instead of exiting broke.

    The final rung before the ``out_of_fuel`` exit, reached only when
    every pickup, larder hop, forage step, and dot hop has declined at
    critical fuel. Walking is free at any fuel level (the density
    probe's marooned-recovery law, [[walk-mechanics]]), so a tank with
    known fuel in walking range is NOT actually stuck: runs
    bot-20260728-090813/-091209 exited at fuel 98/88 in a shore corner
    with the whole dot atlas 15+ unaffordable-teleport tiles away.
    Each tick walks one in-viewport leg toward the nearest candidate
    (map dot or believed container); arrival is handled by the normal
    cascade -- fresh ground re-enables forage, scans, and pickups.
    When the window itself is exhausted (the leg clamps onto the
    tank's own tile), :func:`_maroon_pan_toward` spends a free scope
    pan instead of skipping the candidate.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        A one-leg walk (or window pan) decision, or ``None`` when no
        known fuel is inside the walk cap or no leg is walkable (the
        exit stands). The caller guarantees critical fuel -- the
        healthy-fuel tick resolved via the hunt handoff before this
        rung.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    candidates: list[tuple[int, int, int]] = []
    for dot_x, dot_y in ctx.map_fuel_dots:
        candidates.append((abs(dot_x - sx) + abs(dot_y - sy), dot_x, dot_y))
    for container in ctx.world["containers"].values():
        if not container["is_fuel"] or container["volume"] <= 0:
            continue
        if container["failed_pickups"] > 0:
            continue
        candidates.append(
            (abs(container["x"] - sx) + abs(container["y"] - sy), container["x"], container["y"])
        )
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    # Nearest-first over EVERY candidate inside the cap: in a shore
    # corner the closest entries are water-locked containers whose leg
    # resolves to a teleport fallback, not a walk (run
    # bot-20260728-092357 gave up after trying only the nearest and
    # exited with dots in walking range further down the list).
    for _, target_x, target_y in sorted(
        c for c in candidates if 0 < c[0] <= WALK_FOR_FUEL_MAX_TILES
    ):
        terrain = ctx.terrain
        if terrain is not None and not terrain.is_passable(target_x, target_y):
            continue
        leg_x = min(max(target_x, left), right)
        leg_y = min(max(target_y, top), bottom)
        if (leg_x, leg_y) == (sx, sy):
            pan = _maroon_pan_toward(ctx, base_state, target_x, target_y)
            if pan is not None:
                return pan
            continue
        command = walk_or_teleport(ctx, leg_x, leg_y, pickup_kind=None)
        if command is None or command["cmd_type"] != "move":
            continue
        emit_ai(
            "marooned at fuel %d: walking leg (%d,%d) toward known fuel at (%d,%d)",
            ctx.fuel,
            leg_x,
            leg_y,
            target_x,
            target_y,
        )
        emit_diagnostic(
            diagnostic_kind="walk_for_fuel",
            target_x=target_x,
            target_y=target_y,
            leg_x=leg_x,
            leg_y=leg_y,
            fuel=ctx.fuel,
        )
        return make_decision(
            command,
            "COLLECT",
            COLLECT_SCORE,
            leg_x,
            leg_y,
            "walk_for_fuel",
            release_collect_plan(base_state, reason="walk_for_fuel_override"),
            ctx.equip,
        )
    return None


__all__ = [
    "WALK_FOR_FUEL_MAX_TILES",
    "walk_for_fuel_last_resort",
]
