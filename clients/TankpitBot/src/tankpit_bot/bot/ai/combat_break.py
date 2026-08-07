"""Damage-aware engagement break: leave a losing fight while escape is cheap.

The 2026-07-28 attrition run (`bot-20260728-075336`) proved the static
``fuel < fuel_low_threshold`` break blind to HOW a fight is going: two
attackers billed ~100 fuel/tick, the break tripped at 198, and the
2-action escape latency (map_open + teleport, ~4 s under fire) cost the
remaining budget before the teleport could leave -- exit ``out_of_fuel``
at 108 with the escape chosen but no longer affordable.

The gate here is a PROJECTION, not a threshold (user challenge
2026-07-28: "are you sure this wont cause premature fleeing from
killable targets?"):

    projected_fuel_at_kill = fuel
        - hits_to_kill * (own cost per shot tick)
        - hits_to_kill * (measured incoming fuel per tick)
    break only if projected_fuel_at_kill < escape floor

A near-death target makes ``hits_to_kill`` tiny, so even heavy fire
barely dents the projection and the kill is FINISHED first. A healthy
fleeing target under sustained fire projects to a strand and breaks
early. At zero measured incoming the projection reduces to own costs
only -- quiet fights keep today's behavior. Breaking delegates through
``refuel_for_hunt`` so the lock survives (never-drop,
[[bot-behavior-contract]]) and COLLECT's larder step supplies the
escape target.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.costs import DUAL_SHOT_COST
from tankpit_bot.physics.damage import DUAL_HIT_VICTIM_COST, HOMING_HIT_VICTIM_COST
from tankpit_bot.protocol.naming import is_human_name

INCOMING_RATE_WINDOW_MS = 10_000
"""Trailing window for the measured incoming-damage rate."""

_MIN_SUSTAIN_HITS = 3
"""Confirmed hits the window must hold before the rate counts at all.
One lucky enemy volley (1-2 hits) never triggers a break."""

_ESCAPE_LATENCY_TICKS = 2
"""Actions between deciding to leave and the teleport leaving:
map_open + teleport (the map-open precondition), ~2 s each --
measured live 2026-07-28 07:59:11-13."""

_TICK_MS = 2000
"""Nominal action-tick length used to convert the window to per-tick."""

_OWN_COST_PER_SHOT_TICK = DUAL_SHOT_COST + 10
"""Our spend per fighting tick: one shot debit plus ~10 position drain
(ledger-measured, run bot-20260728-075336: 40 duals at exactly -10
each, drain -10/tick)."""


class EngagementBreakDict(TypedDict):
    """One break assessment with its full arithmetic (diagnostic payload).

    Attributes:
        break_engagement: True when the projection says finishing
            strands the tank.
        hits_in_window: Fuel-confirmed incoming hits in the window.
        incoming_fuel_in_window: Fuel those hits cost.
        incoming_rate_per_tick: Windowed incoming fuel per action tick.
        hits_to_kill: Estimated shots left to finish the target.
        projected_fuel_at_kill: Fuel projected at kill completion.
        escape_floor: Reserve + escape latency exposure the projection
            must clear.
    """

    break_engagement: bool
    hits_in_window: int
    incoming_fuel_in_window: int
    incoming_rate_per_tick: int
    hits_to_kill: int
    projected_fuel_at_kill: int
    escape_floor: int


def estimate_hits_to_kill(ctx: DecideCtx, target: EnemyThreatDict) -> int:
    """Estimate remaining shots to finish the target from its quartile.

    ``damage_state`` is the fuel-quartile health tier (0=near death ..
    3=full); the estimate takes the quartile's UPPER bound of the
    target's rank-derived capacity (conservative: never underestimates
    the work left) against our per-hit damage -- dual (90) while duals
    are stocked, homing (45) otherwise.

    Args:
        ctx: Decision context (for our weapon stock).
        target: The engaged enemy.

    Returns:
        Estimated hits remaining, at least 1.
    """
    enemy_fuel_bound = (target["damage_state"] + 1) * fuel_capacity(target["rank"]) // 4
    per_hit = (
        DUAL_HIT_VICTIM_COST if ctx.inventory["dual_shots"]["count"] > 0 else HOMING_HIT_VICTIM_COST
    )
    return max(1, -(-enemy_fuel_bound // per_hit))


def assess_engagement_break(
    ctx: DecideCtx,
    target: EnemyThreatDict,
    hits_in_window: int,
    incoming_fuel_in_window: int,
) -> EngagementBreakDict:
    """Project the cost of finishing the kill against the escape fund.

    Human-fight break band (user ruling 2026-07-31: "the bot seems to
    run too much when fighting a human... it kinda just does damage
    then leaves"): against a human-classified target the projection is
    SUPPRESSED while fuel is at or above half the rank capacity. Human
    fights are attrition -- both sides refuel, sustain wins (2026-07-30
    ruling) -- and the one-kill projection vs a full-health human
    (hits_to_kill ~13) breaks near 900 fuel, which read as constant
    fleeing. The escape floor sits well below half capacity at every
    measured human damage rate (max 72/tick), so holding to the band
    never strands the escape. The ``sustained`` requirement is shared,
    so a quiet human fight still never breaks here.

    Args:
        ctx: Decision context.
        target: The engaged enemy.
        hits_in_window: Fuel-confirmed incoming hits in the trailing
            window (:data:`INCOMING_RATE_WINDOW_MS`).
        incoming_fuel_in_window: Fuel those hits cost.

    Returns:
        The assessment with its full arithmetic.
    """
    window_ticks = INCOMING_RATE_WINDOW_MS // _TICK_MS
    sustained = hits_in_window >= _MIN_SUSTAIN_HITS
    rate_per_tick = incoming_fuel_in_window // window_ticks if sustained else 0
    hits_to_kill = estimate_hits_to_kill(ctx, target)
    projected = ctx.fuel - hits_to_kill * (_OWN_COST_PER_SHOT_TICK + rate_per_tick)
    escape_floor = (
        ctx.config["fuel_low_threshold"]
        + ctx.config["hunt_min_fuel"]
        + _ESCAPE_LATENCY_TICKS * rate_per_tick
    )
    human_band_holds = (
        is_human_name(target["name"]) and ctx.fuel >= fuel_capacity(ctx.self_state["rank"]) // 2
    )
    return EngagementBreakDict(
        break_engagement=sustained and projected < escape_floor and not human_band_holds,
        hits_in_window=hits_in_window,
        incoming_fuel_in_window=incoming_fuel_in_window,
        incoming_rate_per_tick=rate_per_tick,
        hits_to_kill=hits_to_kill,
        projected_fuel_at_kill=projected,
        escape_floor=escape_floor,
    )


__all__ = [
    "INCOMING_RATE_WINDOW_MS",
    "EngagementBreakDict",
    "assess_engagement_break",
    "estimate_hits_to_kill",
]
