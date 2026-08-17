"""Combat approach: teleport to the target and close the last tiles.

The outermost engagement stage -- it decides how to reach a locked
target and hands off to :func:`~tankpit_bot.bot.ai.combat_strategy.engage_target`
once in range. Sits above every other combat module; nothing imports
it back.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_opportunity import opportunity_shot_decision
from tankpit_bot.bot.ai.combat_prep import refuel_for_hunt
from tankpit_bot.bot.ai.combat_strategy import (
    engage_target,
    has_cardinal_combat_shot,
    has_combat_shot,
)
from tankpit_bot.bot.ai.combat_target import (
    _combat_landing_candidates,
    _set_combat_target,
    block_combat_target_and_replan,
    combat_landing_tile,
    has_clear_shot_line,
    is_already_engaged,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport,
    make_decision,
    teleport_fuel_cost_to,
)
from tankpit_bot.bot.ai.hunt_relay import relay_toward
from tankpit_bot.bot.ai.mine_clearance import find_corridor_clearance_shot
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    make_move_command,
    make_shoot_command,
    make_teleport_command,
)
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

WALK_CLOSE_TILES = 8
"""Manhattan distance under which closing on a target WALKS.

Re-priced 2026-08-13 (HUD flag 16: "why didn't it just walk back to
the enemy? a teleport is 2 ticks and more fuel per tile") from the
MEASURED walking speed -- 15 cardinal tiles in 3.30 s (2026-08-06),
~0.22 s per tile at ~1 fuel per tile -- replacing the falsified
~2 s-per-tile premise that set the old bound of 3. A walk is one
move dispatch (~2 s tick) plus the server's leg, so at 8 tiles it
costs ~3.8 s and 8 fuel against the teleport's ~4 s (map-open
precondition plus the hop) and ``floor(6 x euclid)`` ~= 34 fuel:
the walk wins on fuel everywhere and on time out to ~9 tiles. The
bound stays inside the corridor-clearance guard's reach (known
mines are shot before the first step); hidden-mine arrest risk is
what keeps it at 8 rather than the full time-breakeven."""


def teleport_to_target(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase 1: Teleport to enemy.


    Args:
        ctx: Decision context.
        target: Combat target to close on.

    Returns:
        Teleport decision, or a blocked-target replanning decision when the
        landing tile is unusable or the teleport is unaffordable.
    """
    # In view means no teleport is needed at all (user law 2026-07-29:
    # "as long as theyre on the viewport and its a clear dual shot then
    # id just hit them from my new location"; flag s2-13 tightened it:
    # purple-9 at Manhattan 9 -- in view, beyond the old 8-tile bound --
    # still got a paid teleport). The server fires at stationary
    # in-view targets from any range ([[weapon-selection]]): water
    # never blocks, rock clips to a billed single that resolves as a
    # miss through the stationary-miss block. This also IS the
    # mine-ring/ferry counterplay -- a ringed or shore-locked target in
    # view is shot regardless of landing state -- and it covers every
    # acquire path (fresh viewport, map-known, locked resume) because
    # they all funnel through here and engage_target latches the lock
    # itself.
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    if left <= target["x"] <= right and top <= target["y"] <= bottom:
        if has_clear_shot_line(ctx, target):
            emit_ai(
                "%s already in view at (%d,%d) - engaging without teleport",
                target["name"],
                target["x"],
                target["y"],
            )
            return engage_target(ctx, target)
        # In view but the dual line is occluded (flag s3-16: "we're
        # shooting over terrain when we should have teleported back
        # adjacent") -- fall through to the close teleport; adjacency
        # has no intermediate tiles, so the landing buys a clear shot.
        emit_ai(
            "%s in view at (%d,%d) but the shot line is blocked - closing for a clear shot",
            target["name"],
            target["x"],
            target["y"],
        )
    close_distance = abs(ctx.self_state["x"] - target["x"]) + abs(ctx.self_state["y"] - target["y"])
    if close_distance <= WALK_CLOSE_TILES:
        # Short closes WALK (user rulings: flag 1 of run
        # bot-20260730-011x "i think it should ahve waked back
        # instead of teleporting", re-priced by HUD flag 16
        # 2026-08-13): at the measured ~0.22 s/tile the walk beats
        # the teleport's map-open + hop on time out to ~9 tiles and
        # costs a sixth of the fuel -- see WALK_CLOSE_TILES.
        walk_candidates = _combat_landing_candidates(ctx, target)
        if walk_candidates:
            walk_x, walk_y = walk_candidates[0]
            corridor_mine = find_corridor_clearance_shot(
                ctx.filtered,
                ctx.self_state,
                ctx.terrain,
                walk_x,
                walk_y,
            )
            if corridor_mine is not None:
                # Flags s6-8/9: six 45-fuel walk-ins against KNOWN
                # mines because no walk ever consulted the mine layer.
                # Mine shots are free singles ([[mine-mechanics]]), so
                # the corridor is drained before the first step.
                mine_x, mine_y = corridor_mine
                emit_ai(
                    "walk corridor to (%d,%d) is mined - clearing (%d,%d) first",
                    walk_x,
                    walk_y,
                    mine_x,
                    mine_y,
                )
                return make_decision(
                    make_shoot_command(mine_x, mine_y),
                    "HUNT",
                    800,
                    mine_x,
                    mine_y,
                    "mine_clearance_shot",
                    _set_combat_target(ctx.base, target),
                    ctx.equip,
                    reason_context={"target_name": target["name"]},
                )
            emit_ai(
                "walking %d tiles to close on %s at (%d,%d)",
                close_distance,
                target["name"],
                target["x"],
                target["y"],
            )
            return make_decision(
                make_move_command(walk_x, walk_y),
                "HUNT",
                800,
                walk_x,
                walk_y,
                "walk_to_target",
                _set_combat_target(ctx.base, target),
                ctx.equip,
                reason_context={"target_name": target["name"]},
            )
    landing_x, landing_y = combat_landing_tile(ctx, target)
    if ctx.ws.is_move_target_failed(landing_x, landing_y, ctx.timestamp_ms):
        # The target is off-view here (in-view targets shot above), so
        # with no legal landing and no legal shot, blocking and
        # replanning is all that is left.
        emit_ai(
            "combat landing (%d,%d) for %s already failed, blocking target",
            landing_x,
            landing_y,
            target["name"],
        )
        return block_combat_target_and_replan(ctx, target)
    # Engaging must leave fuel to FIGHT, not merely to exist: the
    # reserve is fuel_low_threshold (the line where COLLECT outranks
    # HUNT) plus the full engagement_fuel_budget -- the same
    # end-to-end funding the acquisition gate demands, so a chase
    # re-teleport can never slip through cheaper than the fight it
    # funds. Two priced incidents behind the sum: run 20260611-004505
    # (a teleport gated only on hunt_min_fuel landed at 224, the fuel
    # mode hijacked the next tick, and the ~190 spent reaching
    # purple-8 bought a forbidden fight) and run 20260729-105325 (the
    # fuel_low_threshold-only reserve passed a 158-cost chase at 372
    # by 14 fuel, landed at 214, LOW_FUEL hijacked one shot later,
    # session min 140 -- user ruling 2026-07-29: "we cant kill anyone
    # if we die... we should fuel before chasing"). Refuel-for-hunt
    # keeps the lock and prefers in-viewport walk pickups, then the
    # larder, so the detour is usually one container away.
    engagement_reserve = ctx.config["fuel_low_threshold"] + ctx.config["engagement_fuel_budget"]
    if not can_afford_teleport(
        ctx,
        landing_x,
        landing_y,
        reserve_fuel=engagement_reserve,
    ):
        cost = teleport_fuel_cost_to(ctx, landing_x, landing_y)
        max_affordable = fuel_capacity(ctx.self_state["rank"]) - engagement_reserve
        if cost > max_affordable:
            # Beyond refuel reach (flag s10-2, 2026-07-30): a 504-cost
            # chase at fuel 1097/1100 hit this branch and "refueled"
            # a 3-point deficit with a 121-fuel dot teleport -- no
            # amount of fuel makes a cost above cap-minus-reserve
            # affordable. Distance problems take the RELAY lane
            # (hunt_relay.relay_toward): the strict-progress dot
            # selector, monotone by construction. The first cut here
            # reached for the COLLECT dot-ranker instead, whose /cost
            # denominator makes the dot under the tank's feet beat any
            # dot toward the prey -- live 2026-08-04 23:24-23:29: a
            # locked target 98 tiles out produced a two-tile teleport
            # ping-pong ((206,254)<->(207,254), then
            # (225,253)<->(225,254)) at one hop per 2 ticks, forever.
            emit_ai(
                "chase to %s costs %d > max affordable %d - relaying via dots "
                "(fuel=%d, refuel cannot fix distance)",
                target["name"],
                cost,
                max_affordable,
                ctx.fuel,
            )
            relay = relay_toward(
                ctx,
                _set_combat_target(ctx.base, target),
                target,
            )
            if relay is not None:
                return relay
            # No strict-progress dot and no below-cap refuel: the
            # target is unreachable by relay from here. Blocking it
            # beats treadmilling -- the next acquisition pass finds
            # closer prey or exits honestly.
            emit_ai(
                "no relay progress toward %s - blocking unreachable target",
                target["name"],
            )
            return block_combat_target_and_replan(ctx, target)
        emit_ai(
            "cannot afford combat teleport for %s to (%d,%d) (fuel=%d cost=%d reserve=%d)"
            " - refueling before hunt",
            target["name"],
            landing_x,
            landing_y,
            ctx.fuel,
            cost,
            engagement_reserve,
        )
        return refuel_for_hunt(ctx, target)
    emit_ai(
        "teleport near %s to (%d,%d) (target at %d,%d)",
        target["name"],
        landing_x,
        landing_y,
        target["x"],
        target["y"],
    )
    return make_decision(
        make_teleport_command(landing_x, landing_y),
        "HUNT",
        800,
        landing_x,
        landing_y,
        "teleport_target",
        _set_combat_target(ctx.base, target),
        ctx.equip,
        reason_context={"target_name": target["name"]},
    )


def close_target(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase closing: shoot when in range or already engaged; teleport on first close.

    User-contract gameplay loop (2026-06-26): open map, teleport
    cardinally adjacent to the target, dual until the target teleports
    away, then stay in place and fire homing until the target is
    deactivated. Enemies don't move *within* the viewport -- when they
    leave cardinal adjacency, they teleported -- so chasing them with
    another teleport burns fuel without changing the firing geometry.

    Three branches:

    1. **Cardinally adjacent** -- shoot (server picks ``dual`` for the
       guaranteed point-blank hit).
    2. **Already engaged** (:func:`is_already_engaged`) -- shoot from
       the current tile. The server picks ``homing`` when the target
       is not adjacent and homing tracks, so a stay-put fire continues
       to land hits without spending fuel on another teleport.
    3. **Fresh acquire, not adjacent** -- teleport directly to the
       target. This is the one-time close the engagement contract
       allows; subsequent ticks fall through branch (2).


    Args:
        ctx: Decision context.
        target: Combat target to approach.

    Returns:
        Close-range combat decision: a shot when already in cardinal position,
        a teleport when affordable, or a blocked-target replanning decision.
    """
    # Firefight doctrine (user ruling 2026-08-14: "you have a main
    # target ofc, but you should also return fire to anyone else
    # engaging and take kill shots ... when someone is in the lowest
    # or second lowest damage state"): before serving the main
    # target, one shot may divert to a visible finisher or an active
    # attacker. The lock is untouched -- next tick the main fight
    # resumes (in view, the re-entry is a free same-tile shot).
    opportunity = opportunity_shot_decision(ctx, target["tank_id"])
    if opportunity is not None:
        return opportunity
    if has_cardinal_combat_shot(ctx.self_state, target) or (
        has_combat_shot(ctx, target) and has_clear_shot_line(ctx, target)
    ):
        # In-view, in range, and the dual line is clean: fire from the
        # current tile (user ruling 2026-07-29). The purple-8 receipt
        # this closes: after a break-driven pickup the bot stood at
        # dist 2 and paid a teleport to regain cardinal adjacency
        # instead of shooting. Cardinal adjacency is trivially clear
        # (no intermediate tiles); a ranged shot must pass the lifted
        # LOS test or the server serves a half-damage over-terrain
        # homing (flag s3-16 / Artax death: the losing trade).
        return engage_target(ctx, target)
    dist = abs(ctx.self_state["x"] - target["x"]) + abs(ctx.self_state["y"] - target["y"])
    if is_already_engaged(ctx):
        if not has_clear_shot_line(ctx, target):
            # Flag s3-16 ("we're shooting over terrain when we should
            # have teleported back adjacent"): a stay-put shot here is
            # a half-damage homing arcing over the occluder while the
            # enemy duals back for 90 -- the trade that killed Artax.
            # Re-close instead; the landing is adjacent, and adjacency
            # always has a clear line.
            emit_ai(
                "engaged %s at (%d,%d) is behind terrain from (%d,%d) - "
                "re-closing for a clear shot",
                target["name"],
                target["x"],
                target["y"],
                ctx.self_state["x"],
                ctx.self_state["y"],
            )
            return teleport_to_target(ctx, target)
        emit_ai(
            "engaged %s moved off cardinal from (%d,%d) target=(%d,%d) dist=%d; staying put",
            target["name"],
            ctx.self_state["x"],
            ctx.self_state["y"],
            target["x"],
            target["y"],
            dist,
        )
        return engage_target(ctx, target)
    emit_ai(
        "fresh acquire of %s from (%d,%d) target=(%d,%d) dist=%d; teleporting to close",
        target["name"],
        ctx.self_state["x"],
        ctx.self_state["y"],
        target["x"],
        target["y"],
        dist,
    )
    return teleport_to_target(ctx, target)


__all__ = [
    "WALK_CLOSE_TILES",
    "close_target",
    "teleport_to_target",
]
