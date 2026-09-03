"""The mine pin: salt an adjacent enemy's ring on the way into a fight.

Operator order (2026-09-01, verbatim): "when we get or teleport
adjacent to an enemy we should be able to use mines to pin them in."
The wire tool is ``CMD_MINE`` ([[mine-mechanics]]): one press lays a
3x3 pattern centered on the PLACER for a flat 10 fuel — mines are not
inventory — with tanks, water, rock, and out-of-window tiles skipped
and enemy mines in the pattern detonated 1:1. A pinned enemy pays 45
fuel per walk step INTO the field and their movement STOPS on the
mine ([[mine-mechanics]] walk-over law), so a salted ring taxes every
walking escape lane; teleports remain open to them (displacement law),
which keeps the pin harassment, not a cage.

The press spends the tick a shot would have used, so the doctrine is
ONE press per engagement: the first engage tick within reach drops,
every later tick shoots. The latch is ``mine_pin_presses`` — a
per-target map (the scalar it replaced re-armed whenever another
target intervened: the 2026-09-01 A→B→A→B lock shuttle bought four
presses on two tiles, flag-triage-20260902 row 7). Each entry also
records the placer tile, so a press that would re-lay an identical
3x3 from already-pressed ground is skipped even against a fresh
target. A re-engage of the same target (resume, pursuit return)
never pays a second press.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_target import _set_combat_target
from tankpit_bot.bot.ai.context import DecideCtx, make_decision
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_mine_drop_command
from tankpit_bot.physics.costs import MINE_PRESS_COST
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic

MINE_PIN_REACH_TILES = 2
"""Farthest Chebyshev distance at which the press still salts the
target's ring: the 3x3 covers self±1, the ring is target±1, and the
two overlap exactly while the distance is at most 2. Beyond it every
laid mine sits outside the ring and the press buys area denial the
fight will leave behind."""


def mine_pin_decision(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict | None:
    """Spend one engage tick pressing mines beside a close target.

    Args:
        ctx: Decision context.
        target: The engaged combat target.

    Returns:
        The mine-drop decision (recording the per-target press and its
        placer tile), or ``None`` when this target already got its
        press, an earlier press was laid from this exact tile (the
        identical 3x3 buys no new ground), the target is beyond
        :data:`MINE_PIN_REACH_TILES`, or fuel is too close to the
        survival floor for a 10-fuel press.
    """
    presses = ctx.ai_state["mine_pin_presses"]
    if str(target["tank_id"]) in presses:
        return None
    press_tile = f"{ctx.self_state['x']},{ctx.self_state['y']}"
    if press_tile in presses.values():
        return None
    reach = max(
        abs(ctx.self_state["x"] - target["x"]),
        abs(ctx.self_state["y"] - target["y"]),
    )
    if reach > MINE_PIN_REACH_TILES:
        return None
    if ctx.fuel <= ctx.fuel_low_floor + MINE_PRESS_COST:
        # Fuel is health: below the survival floor the 10-fuel press
        # and its tick both belong to the escape doctrine.
        return None
    emit_ai(
        "mine pin: salting %s's ring from (%d,%d) at reach %d",
        target["name"],
        ctx.self_state["x"],
        ctx.self_state["y"],
        reach,
    )
    emit_diagnostic(
        diagnostic_kind="mine_pin_pressed",
        target_id=target["tank_id"],
        target_name=target["name"],
        reach=reach,
        self_x=ctx.self_state["x"],
        self_y=ctx.self_state["y"],
    )
    return make_decision(
        make_mine_drop_command(),
        "HUNT",
        800,
        target["x"],
        target["y"],
        "mine_pin",
        AIStateDict(
            **{
                **_set_combat_target(ctx.base, target),
                "mine_pin_presses": {
                    **ctx.base["mine_pin_presses"],
                    str(target["tank_id"]): press_tile,
                },
            }
        ),
        ctx.equip,
        reason_context={"target_name": target["name"]},
    )


__all__ = [
    "MINE_PIN_REACH_TILES",
    "mine_pin_decision",
]
