"""Shared retreat obligation and teleport eligibility before candidate ranking."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_break import INCOMING_RATE_WINDOW_MS
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.threat_primitives import WIRE_PRESENCE_TTL_MS
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.physics.supervisor import TELEPORT_RING1_COST_SLACK
from tankpit_bot.state.types import has_known_position
from tankpit_bot.types.constants import TankLiveness


def escape_required(ctx: DecideCtx, state: AIStateDict) -> bool:
    """Keep a committed break active even after its hit window decays.

    Args:
        ctx: Current decision snapshot and incoming damage instrument.
        state: State carrying the existing fuel-resume latch.

    Returns:
        Whether fuel recovery owns the tick; isolated hits without a
        losing-fight commitment retain ordinary collection policy.
    """
    if state["break_escape_until_fuel"] > ctx.fuel:
        return True
    hits, _ = ctx.ws.get_incoming_damage_window(
        ctx.timestamp_ms, INCOMING_RATE_WINDOW_MS, WIRE_PRESENCE_TTL_MS
    )
    return hits >= 3


def escape_hop_permitted(ctx: DecideCtx, x: int, y: int) -> bool:
    """Price transit before pickup credit and clear every fresh attacker.

    The two-action map/teleport sequence budgets two observed average
    hits, or two rate-derived ticks when that is higher, plus ring-1
    displacement. This is a risk budget, not a prediction of the server's
    affordability rule. It is re-evaluated when map opening completes.

    Args:
        ctx: Decision snapshot, including committed retreat and damage.
        x: Actual candidate landing X, after approach resolution.
        y: Actual candidate landing Y, after approach resolution.

    Returns:
        True outside retreat, or when transit is payable with life left
        and the landing clears the held target and fresh incoming shooters.
    """
    if not escape_required(ctx, ctx.base):
        return True
    hits, fuel_lost = ctx.ws.get_incoming_damage_window(
        ctx.timestamp_ms, INCOMING_RATE_WINDOW_MS, WIRE_PRESENCE_TTL_MS
    )
    exposure = 0
    if hits > 0:
        average_hit = -(-fuel_lost // hits)
        per_tick = -(-fuel_lost * 2000 // INCOMING_RATE_WINDOW_MS)
        exposure = 2 * max(average_hit, per_tick) + TELEPORT_RING1_COST_SLACK
    cost = teleport_cost(ctx.self_state["x"], ctx.self_state["y"], x, y)
    if cost + exposure >= ctx.fuel:
        return False
    state = ctx.base
    if state["combat_target_id"] != -1 and (
        max(abs(x - state["combat_target_x"]), abs(y - state["combat_target_y"])) < 16
    ):
        return False
    attackers = {
        hit["shooter_id"]
        for hit in ctx.ws.damage_book["confirmed_incoming"]
        if hit["timestamp_ms"] >= ctx.timestamp_ms - INCOMING_RATE_WINDOW_MS
    }
    for tank in ctx.world["tanks"].values():
        if (
            tank["tank_id"] in attackers
            and tank["liveness"] is not TankLiveness.DEACTIVATED
            and ctx.timestamp_ms - tank["last_wire_seen_ms"] <= WIRE_PRESENCE_TTL_MS
            and has_known_position(tank)
            and max(abs(x - tank["x"]), abs(y - tank["y"])) < 16
        ):
            return False
    return True


__all__ = ["escape_hop_permitted", "escape_required"]
