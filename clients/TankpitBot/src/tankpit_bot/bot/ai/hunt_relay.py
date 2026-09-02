"""Dot-relay travel toward out-of-range enemies (HUNT support).

The relay chain is the user's own play ("yellow dot teleporting while
en route to the opponent", 2026-07-03): hop fuel dots that close
distance to a target the bot cannot yet afford end-to-end, refuelling
on each landing, falling back to an any-direction refuel hop when the
deficit is fuel rather than distance (2026-07-19). Also home to the
human-pursuit travel selection (unlimited-distance doctrine,
2026-07-29) and the stale-human map-refresh predicate.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx, make_decision
from tankpit_bot.bot.ai.resource_search import make_resource_search_hop
from tankpit_bot.bot.ai.threat_acquisition import (
    find_relay_travel_targets,
    stale_human_exists,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_teleport_command
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.protocol.naming import is_human_name
from tankpit_bot.runtime_logging import emit_ai


def _pick_relay_dot(
    ctx: DecideCtx,
    travel: EnemyThreatDict,
) -> tuple[int, int] | None:
    """Return the fuel dot that best closes distance to ``travel``.

    A dot qualifies when it is strictly closer to the enemy than the
    bot's current tile (euclidean -- teleport cost geometry), the
    landing tile is passable, the hop leaves ``fuel_low_threshold``
    behind so a dry dot cannot strand the bot below the COLLECT entry
    reserve, and the leg costs at most ``engagement_fuel_budget`` --
    the per-leg deficit cap (2026-07-29). Among qualifiers the one
    nearest the enemy wins (maximum progress per hop); ties keep the
    cheaper hop. Strict progress makes the relay monotone -- it
    terminates at the enemy or runs out of qualifying dots.

    The leg cap closes the broke-arrival stall from the first live
    human pursuit (run bot-20260729-211458 window, 21:17:40): the
    uncapped max-progress rule picked a dot 131 tiles out, paid 787
    fuel in ONE leg (1100 -> 313), and landed four tiles from Yuppler
    unable to fight -- the between-kills restock then foraged AWAY
    from the target for minutes while the human watched. Capped legs
    keep every landing a short top-up instead of a full rebuild, so
    the chain stays "refuel per hop" as the 2026-07-03 contract
    intended.

    Args:
        ctx: Decision context.
        travel: Enemy the relay is travelling toward.

    Returns:
        ``(x, y)`` of the best relay dot, or ``None`` when no dot
        makes affordable progress.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    ex, ey = travel["x"], travel["y"]
    self_to_enemy_sq = (ex - sx) ** 2 + (ey - sy) ** 2
    best: tuple[int, int] | None = None
    best_remaining_sq = self_to_enemy_sq
    best_cost = 0
    for dot_x, dot_y in ctx.map_fuel_dots:
        remaining_sq = (ex - dot_x) ** 2 + (ey - dot_y) ** 2
        if remaining_sq >= self_to_enemy_sq:
            continue
        if ctx.terrain is not None and not ctx.terrain.is_passable(dot_x, dot_y):
            continue
        cost = teleport_cost(sx, sy, dot_x, dot_y)
        if cost > ctx.config["engagement_fuel_budget"]:
            continue
        if cost + ctx.config["fuel_low_threshold"] > ctx.fuel:
            continue
        if (
            best is None
            or remaining_sq < best_remaining_sq
            or (remaining_sq == best_remaining_sq and cost < best_cost)
        ):
            best = (dot_x, dot_y)
            best_remaining_sq = remaining_sq
            best_cost = cost
    return best


def relay_toward_unaffordable_enemy(
    ctx: DecideCtx,
    ai_state: AIStateDict,
) -> TickDecisionDict | None:
    """Hop toward the nearest out-of-range enemy via a fuel dot.

    User contract (2026-07-03): when no enemy is affordable end-to-end,
    do what a human does -- yellow-dot teleport while en route to the
    opponent, refuelling on each landing (teleporting onto a container
    tile auto-picks it up), instead of exiting the session. The relay
    fires only when a map-fresh enemy exists that fails ONLY the
    affordability gate; a map with no enemies at all still exits with
    ``no_viable_targets``.

    Each hop strictly reduces the distance to the enemy, so the relay
    terminates: either acquisition succeeds on a later tick (fuel
    recovered, distance shortened) or no qualifying dot remains.

    When no strict-progress dot exists the deficit may be FUEL, not
    distance: run 2026-07-19 14:49 rejoined at fuel 653 with orange-2
    only 26.6 tiles away (engage cost 159 -- unaffordable purely
    because 159+650 > 653) and only 6 of 628 dots strictly closer, all
    on water. The strict-progress rule starved the bot amid 622 usable
    dots. The fallback (:func:`_refuel_toward_engagement`) hops to the
    best fresh fuel dot in ANY direction -- getting richer instead of
    closer -- and only when that too is impossible does the caller
    exit the session.

    Args:
        ctx: Decision context.
        ai_state: Base AI state for the produced command.

    Returns:
        Relay teleport decision, refuel-hop decision, or ``None`` when
        there is no enemy worth relaying toward or no dot helps.
    """
    targets = find_relay_travel_targets(
        ctx.ws,
        ctx.filtered,
        ctx.self_state,
        ctx.blocked_targets,
        ctx.killed,
        ctx.terrain,
        ctx.timestamp_ms,
        ctx.config["map_intel_horizon_ms"],
        engagement_reserve_fuel=(
            ctx.config["engagement_fuel_budget"] + ctx.config["fuel_low_threshold"]
        ),
        priority_target_name=ctx.config["priority_target_name"],
        human_min_rank=ctx.config["human_target_min_rank"],
        human_max_rank=ctx.config["human_target_max_rank"],
        doctrine=ctx.config["doctrine"],
    )
    # EVERY candidate gets a dot search before any refuel fallback:
    # the nearest enemy can be dot-starved while a farther one has a
    # rich corridor (run bot-20260827-162050: red-5 at dist 139 had
    # ZERO qualifying dots behind the practice lake while red-2's
    # corridor held 24 — the single-candidate relay exited the
    # session at full fuel with 27 live enemies on the map).
    for travel in targets:
        dot = _pick_relay_dot(ctx, travel)
        if dot is not None:
            return _relay_leg_decision(ctx, ai_state, travel, dot)
    if targets:
        return _refuel_toward_engagement(ctx, ai_state, targets[0])
    return None


def relay_toward(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    travel: EnemyThreatDict,
) -> TickDecisionDict | None:
    """Produce one relay leg toward ``travel``: progress dot, else refuel.

    The shared core of every relay chain: hop to the fuel dot that
    best closes distance to the travel target, falling back to the
    any-direction refuel hop when no dot makes strict progress (the
    deficit is fuel, not distance). ``ai_state`` passes through to the
    decision unchanged, so a caller holding a combat lock keeps it
    across the leg (refuel-then-resume / human pursuit both ride
    this).

    Args:
        ctx: Decision context.
        ai_state: Base AI state for the produced command.
        travel: Enemy the relay is travelling toward.

    Returns:
        Relay teleport decision, refuel-hop decision, or ``None`` when
        no dot helps (caller decides the fallback).
    """
    dot = _pick_relay_dot(ctx, travel)
    if dot is None:
        return _refuel_toward_engagement(ctx, ai_state, travel)
    return _relay_leg_decision(ctx, ai_state, travel, dot)


def _relay_leg_decision(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    travel: EnemyThreatDict,
    dot: tuple[int, int],
) -> TickDecisionDict:
    """Build the one-leg dot-hop decision toward ``travel``.

    Args:
        ctx: Decision context.
        ai_state: Base AI state for the produced command.
        travel: Enemy the relay is travelling toward.
        dot: The chosen relay dot.

    Returns:
        The relay teleport decision.
    """
    dot_x, dot_y = dot
    emit_ai(
        "dot-relay toward %s (id=%d) at (%d,%d): hop to dot (%d,%d) (fuel=%d)",
        travel["name"],
        travel["tank_id"],
        travel["x"],
        travel["y"],
        dot_x,
        dot_y,
        ctx.fuel,
    )
    return make_decision(
        make_teleport_command(dot_x, dot_y),
        "HUNT",
        800,
        dot_x,
        dot_y,
        "dot_relay",
        ai_state,
        ctx.equip,
    )


def stale_human_needs_map_refresh(ctx: DecideCtx) -> bool:
    """Return whether a map refresh is owed to a stale known human.

    True only when BOTH hold: the map itself is older than the
    cooldown (a fresh snapshot that still shows the human stale means
    they left the game -- no refresh can cure that), and a rank-window
    human in the registry is rejected purely for stale map data.

    Args:
        ctx: Decision context.

    Returns:
        True when the acquire path should refresh the map instead of
        settling for a practice bot.
    """
    map_age_ms = ctx.timestamp_ms - ctx.ai_state["last_map_open_ms"]
    if map_age_ms <= ctx.config["map_open_cooldown_ms"]:
        return False
    return stale_human_exists(
        ctx.ws,
        ctx.filtered,
        ctx.self_state,
        ctx.blocked_targets,
        ctx.killed,
        ctx.terrain,
        ctx.timestamp_ms,
        ctx.config["map_intel_horizon_ms"],
        engagement_reserve_fuel=(
            ctx.config["engagement_fuel_budget"] + ctx.config["fuel_low_threshold"]
        ),
        human_min_rank=ctx.config["human_target_min_rank"],
        human_max_rank=ctx.config["human_target_max_rank"],
        doctrine=ctx.config["doctrine"],
    )


def human_pursuit_travel_target(ctx: DecideCtx) -> EnemyThreatDict | None:
    """Return the human the relay should pursue, or ``None``.

    User ruling 2026-07-29 ("unlimited distance for humans... this is
    the real deal"): a rank-window human on the map outranks every
    affordable practice bot at TARGET SELECTION -- even when reaching
    them takes a fuel-dot relay chain across the whole field. Locks
    are still never switched mid-fight (same-day follow-up: "finish
    the kill then the human player will be the next target"); this
    helper only fires during fresh acquisition.

    :func:`find_relay_travel_targets` already sorts human-tier first,
    so its winner being a bot proves no pursuit-worthy human exists.
    Born from the Yuppler encounter (run bot-20260729-204708 window:
    dist 95 rejected ``unaffordable`` while the bot farmed red-3 at
    dist 19 -- the doctrine said prioritize humans, the code only
    consulted the relay when NOTHING was affordable).

    Args:
        ctx: Decision context.

    Returns:
        The unaffordable rank-window human worth relaying toward, or
        ``None`` when no such human is on the fresh map.
    """
    targets = find_relay_travel_targets(
        ctx.ws,
        ctx.filtered,
        ctx.self_state,
        ctx.blocked_targets,
        ctx.killed,
        ctx.terrain,
        ctx.timestamp_ms,
        ctx.config["map_intel_horizon_ms"],
        engagement_reserve_fuel=(
            ctx.config["engagement_fuel_budget"] + ctx.config["fuel_low_threshold"]
        ),
        priority_target_name=ctx.config["priority_target_name"],
        human_min_rank=ctx.config["human_target_min_rank"],
        human_max_rank=ctx.config["human_target_max_rank"],
        doctrine=ctx.config["doctrine"],
    )
    if targets and is_human_name(targets[0]["name"]):
        return targets[0]
    return None


def _refuel_toward_engagement(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    travel: EnemyThreatDict,
) -> TickDecisionDict | None:
    """Hop to the best fresh fuel dot in ANY direction to fund the fight.

    Fired when an otherwise-viable enemy exists but no dot makes
    strict progress toward it: the deficit is fuel, not distance (run
    2026-07-19 14:49 -- a human grabs the yellow dot next door, tops
    up, and pounces; the bot exited the session instead). Reuses the
    COLLECT restock picker, so the hop inherits its freshness,
    affordability, and value-ranking gates.

    Refueling only helps below the fuel cap: at capacity a still-
    unaffordable enemy is genuinely out of range and the caller's
    session exit is correct. Termination: a fuel-bearing hop raises
    fuel toward the cap or affordability; a dry hop lowers it toward
    the picker's own affordability floor -- either way the fallback
    cannot loop forever, and the ledger's retry-loop audit surfaces
    repeated same-target hops.

    Args:
        ctx: Decision context.
        ai_state: Base AI state for the produced command.
        travel: The enemy the refuel is funding an engagement with.

    Returns:
        Refuel teleport decision, or ``None`` when the tank is already
        at fuel capacity or no fresh dot qualifies.
    """
    capacity = fuel_capacity(ctx.self_state["rank"])
    if ctx.fuel >= capacity:
        return None
    engage_cost = teleport_cost(
        ctx.self_state["x"],
        ctx.self_state["y"],
        travel["x"],
        travel["y"],
    )
    needed = engage_cost + ctx.config["engagement_fuel_budget"] + ctx.config["fuel_low_threshold"]
    emit_ai(
        "no progress dot toward %s (id=%d) -- refueling in place "
        "(fuel=%d, engagement needs ~%d, capacity=%d)",
        travel["name"],
        travel["tank_id"],
        ctx.fuel,
        needed,
        capacity,
    )
    return make_resource_search_hop(
        ctx,
        mode="HUNT",
        score=800,
        reason="hunt_refuel",
        ai_state=ai_state,
    )


__all__ = [
    "human_pursuit_travel_target",
    "relay_toward",
    "relay_toward_unaffordable_enemy",
    "stale_human_needs_map_refresh",
]
