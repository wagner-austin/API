"""Fresh target acquisition for HUNT: search, resume, greet, acquire.

The resume-or-acquire cascade: continue a held lock (in view or via
pursuit), then viewport-confirmed threats, then map-known candidates,
then the dot relay / refuel-in-place fallbacks, ending with the
``no_viable_targets`` session exit on a fresh empty map. Also home to
the human-consent greeting approach (2026-07-30). Held-lock mechanics
live in :mod:`hunt_lock`; relay travel in :mod:`hunt_relay`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_landing import choose_greeting_landing_tile
from tankpit_bot.bot.ai.combat_strategy import (
    clear_combat_target,
    close_target,
    engage_target,
    get_locked_target,
    has_cardinal_combat_shot,
    open_map_for_target,
    refuel_for_hunt,
    select_new_combat_target,
    teleport_to_target,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport,
    make_decision,
    target_position_is_fresh,
)
from tankpit_bot.bot.ai.humans import (
    is_human_name,
    is_human_rank_protected,
    is_practice_bot_name,
)
from tankpit_bot.bot.ai.hunt_lock import (
    locked_target_pursuit,
    visible_threats,
)
from tankpit_bot.bot.ai.hunt_relay import (
    human_pursuit_travel_target,
    relay_toward,
    relay_toward_unaffordable_enemy,
    stale_human_needs_map_refresh,
)
from tankpit_bot.bot.ai.threats import (
    find_acquisition_target,
    human_combat_consented,
    make_enemy_threat_from_tank,
    manhattan_distance,
)
from tankpit_bot.bot.ai.types import AIStateDict, EnemyThreatDict, ReasonKind
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_map_open_command, make_teleport_command
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic


def search_for_enemies(
    ctx: DecideCtx,
    *,
    ai_state: AIStateDict,
    map_reason: ReasonKind,
) -> TickDecisionDict:
    """Open the map for a global enemy snapshot.

    HUNT never fires radar to look for enemies (radar reveals only
    hidden entities -- fuel / equipment containers and mines) and the
    viewport-edge walk was dead weight under this game configuration
    (viewport shifting is OFF, so walking to an edge reveals no new
    ground -- only a teleport opens a new viewport, and a directionless
    edge-teleport burns fuel without aiming at a known enemy). The
    only useful enemy-search action when no target is in
    ``analyze_threats`` is to refresh the global map snapshot.

    The dispatch is always issued. The bot's in-flight-action machinery
    short-circuits a second dispatch while one is already pending, and
    every fresh ``map_data_processed`` event hands the acquire path a
    new set of enemy positions to chase.

    Args:
        ctx: Decision context.
        ai_state: Base AI state to rewrite for the produced command.
        map_reason: Behavior reason for the map-open refresh.

    Returns:
        Map-open decision tagged with ``map_reason``.
    """
    emit_ai("opening map to search for enemies")
    return make_decision(
        make_map_open_command(),
        "HUNT",
        0,
        0,
        0,
        map_reason,
        AIStateDict(
            **{
                **ai_state,
                "last_map_open_ms": ctx.timestamp_ms,
            }
        ),
        ctx.equip,
    )


def decide_hunt_acquire(ctx: DecideCtx) -> TickDecisionDict:
    """Resume a viewport-confirmed lock or acquire a fresh combat target.

    Resume-or-acquire cascade:

    1. **Resume held lock.** If ``combat_target_id != -1`` and the
       locked target is in the current threat list, engage or close on
       it. If the lock is set but the target is off-viewport (a mode
       interrupt relocated the bot, or the target fled past the
       reroute window), the pursuit helper chases it: teleport back on
       a trustworthy position, refresh the map on a stale one, and
       release the lock ONLY when the target is genuinely gone -- dead
       or vanished from the registry (user ruling 2026-07-26: live
       targets are never dropped). Resuming means going to the target,
       never firing from stand-off range (user contract 2026-07-02;
       live run 2026-07-01 20:48 fired at a target 92 tiles away and
       looped on server rejections).
    2. **Strict (viewport-confirmed) threats.** ``analyze_threats``
       returns only enemies with recent ``last_viewport_observation_ms``;
       these are immediately fireable. If a viable one exists, pick it
       and teleport (or open the map first if the wire position is
       stale).
    3. **Loose (map-fresh) acquisition.** When no viewport-confirmed
       threat exists, look at every enemy whose ``timestamp_ms`` is
       within ``map_open_cooldown_ms`` (i.e. seen in a recent map
       snapshot), gated on end-to-end affordability (teleport cost +
       kill budget + fuel-low reserve). Teleport at the nearest
       affordable one. ``SCAN_ON_LANDING`` handles viewport
       confirmation before any shot.

    When the map snapshot is fresh and nothing is viable the session
    exits with ``no_viable_targets``; a stale or absent snapshot
    dispatches another ``map_open``.
    """
    threats = visible_threats(ctx)
    locked = get_locked_target(ctx, threats)
    if locked is not None:
        emit_ai("resuming held lock on %s (id=%d)", locked["name"], locked["tank_id"])
        if has_cardinal_combat_shot(ctx.self_state, locked):
            return engage_target(ctx, locked)
        return close_target(ctx, locked)
    if ctx.ai_state["combat_target_id"] != -1:
        # A lock that reaches ACQUIRE with its target off-viewport is
        # an engagement resumed after a mode interrupt (COLLECT may
        # have relocated the bot arbitrarily far). User contract
        # (2026-07-25): the restock cycle does NOT abandon the target
        # -- damage persists, so return to the same tank and finish
        # it. Resuming means GOING to the target, never firing from
        # stand-off range (user contract 2026-07-02; live run
        # 2026-07-01 20:48 fired at a target 92 tiles away and looped
        # on server rejections): teleport back on a trustworthy
        # position, refresh the map on a stale one, and release the
        # lock only when the target is genuinely gone (dead or
        # vanished from the registry).
        pursuit = locked_target_pursuit(ctx)
        if pursuit is None:
            # The ONLY place a live-looking lock is released (user
            # ruling 2026-07-26: live targets are never dropped) --
            # reached exclusively when the registry says the target
            # is dead or vanished entirely.
            emit_diagnostic(
                diagnostic_kind="target_departed",
                target_id=ctx.ai_state["combat_target_id"],
                reason="gone_from_registry",
            )
            emit_ai(
                "locked target id=%d is gone - re-acquiring fresh",
                ctx.ai_state["combat_target_id"],
            )
            return _decide_hunt_acquire_fresh(ctx, threats, clear_combat_target(ctx.base))
        return_cost = teleport_cost(
            ctx.self_state["x"],
            ctx.self_state["y"],
            pursuit["x"],
            pursuit["y"],
        )
        engagement_floor = ctx.config["engagement_fuel_budget"] + ctx.config["fuel_low_threshold"]
        if ctx.fuel < return_cost + engagement_floor:
            if is_human_name(pursuit["name"]):
                # Unlimited-distance human pursuit (user ruling
                # 2026-07-29): a locked human who teleported beyond
                # the fundable range is CHASED via the relay -- each
                # leg closes distance AND refuels on the landing
                # pickup, where the plain refuel detour only gets
                # richer in place. ``ctx.base`` carries the lock, so
                # never-drop rides through every leg.
                emit_ai(
                    "locked human %s beyond fundable range (fuel=%d, needs ~%d) - "
                    "trying a relay leg with the lock held",
                    pursuit["name"],
                    ctx.fuel,
                    return_cost + engagement_floor,
                )
                relay = relay_toward(ctx, ctx.base, pursuit)
                if relay is not None:
                    return relay
            emit_ai(
                "cannot fund return to locked target %s (fuel=%d, needs ~%d) - "
                "refueling with the lock held (resume follows)",
                pursuit["name"],
                ctx.fuel,
                return_cost + engagement_floor,
            )
            # Refuel-then-RESUME (user ruling 2026-07-27): the fuel
            # detour keeps the lock; the resume machinery returns here
            # once the trip is fundable. Run 183703's red-1 was lost
            # to the old release-and-reacquire at this exact branch.
            return refuel_for_hunt(ctx, pursuit)
        if not target_position_is_fresh(ctx, pursuit):
            emit_ai(
                "returning to locked target %s - refreshing stale position via map",
                pursuit["name"],
            )
            return open_map_for_target(ctx, pursuit)
        emit_ai(
            "returning to locked target %s at (%d,%d) after mode interrupt",
            pursuit["name"],
            pursuit["x"],
            pursuit["y"],
        )
        return teleport_to_target(ctx, pursuit)

    return _decide_hunt_acquire_fresh(ctx, threats, ctx.base)


def _unvisited_unconsented_human(
    ctx: DecideCtx,
    ai_state: AIStateDict,
) -> EnemyThreatDict | None:
    """Return the nearest human owed the stand-off visit, or ``None``.

    A candidate is an alive, position-synced, map-fresh enemy human
    inside the configured rank window who has neither consented to
    combat nor been VISITED yet. The visit latch is deliberately
    separate from the HELLO latch (user ruling 2026-07-31: the hello
    can fire from anywhere on the map; the visit is what puts the bot
    in their sight so consent can follow).

    Args:
        ctx: Decision context.
        ai_state: Base AI state (the ``visited_tank_ids`` map).

    Returns:
        The nearest qualifying human as a threat record.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    candidate: EnemyThreatDict | None = None
    best_dist = 0
    for tank in ctx.filtered["tanks"].values():
        if tank["is_self"] or tank["team"] == ctx.self_state["team"]:
            continue
        if tank["liveness"] != "alive" or (tank["x"] == 0 and tank["y"] == 0):
            continue
        if not is_human_name(tank["name"]):
            continue
        if is_human_rank_protected(
            tank["name"],
            tank["rank"],
            min_rank=ctx.config["human_target_min_rank"],
            max_rank=ctx.config["human_target_max_rank"],
        ):
            continue
        if str(tank["tank_id"]) in ai_state["visited_tank_ids"]:
            continue
        if human_combat_consented(tank["tank_id"]):
            continue
        if ctx.timestamp_ms - tank["timestamp_ms"] > ctx.config["map_open_cooldown_ms"]:
            continue
        dist = manhattan_distance(sx, sy, tank["x"], tank["y"])
        if candidate is None or dist < best_dist:
            candidate = make_enemy_threat_from_tank(tank, dist)
            best_dist = dist
    return candidate


def _greeting_approach(
    ctx: DecideCtx,
    ai_state: AIStateDict,
) -> TickDecisionDict | None:
    """Teleport a few tiles off an unconsented human so they see us.

    The human-consent contract's approach move (user ruling
    2026-07-30: "make sure we teleport to them. and that we've said
    hello first, then that is when the consent contract must be
    returned. but we want to see them. and not an adjacent teleport.
    a few tiles off"): an unconsented human is never ACQUIRED, but
    the bot still comes to them once -- landing in the greeting
    stand-off band so both sides see each other. The HELLO itself is
    independent and may already have fired from anywhere on the map
    (user ruling 2026-07-31; ``bot/ai/greeting.py``). The
    ``visited_tank_ids`` map makes the visit a one-shot per human,
    stamped on DISPATCH so an unresponsive human cannot draw visit
    after visit: after it, the bot resumes farming until they respond
    (chat) or strike first, either of which consents them into normal
    acquisition.

    Args:
        ctx: Decision context.
        ai_state: Base AI state for the produced command.

    Returns:
        The greeting-approach teleport, or ``None`` when no map-fresh,
        rank-window, unconsented, unvisited human exists or no legal
        stand-off landing is affordable.
    """
    candidate = _unvisited_unconsented_human(ctx, ai_state)
    if candidate is None:
        return None
    landing = choose_greeting_landing_tile(ctx.filtered, ctx.self_state, candidate, ctx.terrain)
    if landing is None:
        return None
    landing_x, landing_y = landing
    if not can_afford_teleport(ctx, landing_x, landing_y):
        return None
    emit_ai(
        "greeting approach: teleporting %d tiles off %s (id=%d) so they see us",
        abs(landing_x - candidate["x"]) + abs(landing_y - candidate["y"]),
        candidate["name"],
        candidate["tank_id"],
    )
    emit_diagnostic(
        diagnostic_kind="greeting_approach",
        target_id=candidate["tank_id"],
        target_name=candidate["name"],
        landing_x=landing_x,
        landing_y=landing_y,
    )
    return make_decision(
        make_teleport_command(landing_x, landing_y),
        "HUNT",
        800,
        landing_x,
        landing_y,
        "greet_approach",
        AIStateDict(
            **{
                **ai_state,
                "visited_tank_ids": {
                    **ai_state["visited_tank_ids"],
                    str(candidate["tank_id"]): ctx.timestamp_ms,
                },
            }
        ),
        ctx.equip,
        reason_context={"target_name": candidate["name"]},
    )


def _decide_hunt_acquire_fresh(
    ctx: DecideCtx,
    threats: list[EnemyThreatDict],
    ai_state: AIStateDict,
) -> TickDecisionDict:
    """Acquire a new target from viewport threats or fresh map intel.

    When the map snapshot is fresh (opened within the cooldown) and no
    enemy passes the acquisition gates -- including affordability --
    the bot first tries a **dot relay**: hop to the fuel dot that best
    closes distance to the nearest otherwise-viable enemy, refuelling
    on landing (user contract 2026-07-03). When no dot makes progress
    it falls back to **refuel-in-place** -- the best fresh fuel dot in
    any direction, funding a future engagement instead of approaching
    it (user ruling 2026-07-19). Only when there is no enemy worth
    relaying toward, or the tank is at fuel capacity, or no fresh dot
    qualifies, does the session end with ``no_viable_targets`` (user
    contract 2026-07-02).

    Args:
        ctx: Decision context.
        threats: Viewport-confirmed threat list for this tick.
        ai_state: Base AI state for the produced command (lock already
            cleared when arriving from a stale-lock release).

    Returns:
        Teleport, map-open, or engage decision.

    Raises:
        SessionExitError: When fresh map intel shows no viable
            target anywhere.
    """
    greet = _greeting_approach(ctx, ai_state)
    if greet is not None:
        return greet
    target = select_new_combat_target(ctx, threats)
    if target is not None:
        emit_ai("new target %s (id=%d)", target["name"], target["tank_id"])
        if target_position_is_fresh(ctx, target):
            emit_ai("fresh wire position - teleporting to %s", target["name"])
            return teleport_to_target(ctx, target)
        return open_map_for_target(ctx, target)

    map_target = find_acquisition_target(
        ctx.filtered,
        ctx.self_state,
        ctx.blocked_targets,
        ctx.killed,
        ctx.terrain,
        ctx.timestamp_ms,
        ctx.config["map_open_cooldown_ms"],
        engagement_reserve_fuel=(
            ctx.config["engagement_fuel_budget"] + ctx.config["fuel_low_threshold"]
        ),
        priority_target_name=ctx.config["priority_target_name"],
        human_min_rank=ctx.config["human_target_min_rank"],
        human_max_rank=ctx.config["human_target_max_rank"],
    )
    if map_target is not None:
        return _acquire_map_target(ctx, ai_state, map_target)

    map_age_ms = ctx.timestamp_ms - ctx.ai_state["last_map_open_ms"]
    if ctx.ai_state["last_map_open_ms"] > 0 and map_age_ms <= ctx.config["map_open_cooldown_ms"]:
        relay = relay_toward_unaffordable_enemy(ctx, ai_state)
        if relay is not None:
            return relay
        raise SessionExitError(
            "no_viable_targets",
            f"fresh map snapshot ({map_age_ms}ms old) has no affordable enemy "
            f"and no relay dot at ({ctx.self_state['x']},{ctx.self_state['y']}) "
            f"fuel={ctx.fuel}",
        )

    return search_for_enemies(
        ctx,
        ai_state=ai_state,
        map_reason="find_enemies",
    )


def _acquire_map_target(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    map_target: EnemyThreatDict,
) -> TickDecisionDict:
    """Resolve a map-known acquisition winner into a decision.

    A practice-bot winner first defers to the human-pursuit overrides:
    the unlimited-distance relay toward a rank-window human (user
    ruling 2026-07-29) and the stale-human map refresh (the freshness
    asymmetry -- a quiet human goes map-stale in 5 s while bots stay
    wire-fresh by moving). Otherwise the winner is teleport-acquired.

    Args:
        ctx: Decision context.
        ai_state: Base AI state for the produced command.
        map_target: The acquisition winner.

    Returns:
        Relay, map-refresh, or teleport-acquire decision.
    """
    if is_practice_bot_name(map_target["name"]):
        human_travel = human_pursuit_travel_target(ctx)
        if human_travel is not None:
            emit_ai(
                "human %s (id=%d) at dist %d outranks affordable bot %s - "
                "relaying toward them (unlimited-distance human pursuit)",
                human_travel["name"],
                human_travel["tank_id"],
                human_travel["distance"],
                map_target["name"],
            )
            relay = relay_toward(ctx, ai_state, human_travel)
            if relay is not None:
                return relay
            emit_ai(
                "no dot or refuel leg helps toward %s right now - farming %s while the map evolves",
                human_travel["name"],
                map_target["name"],
            )
        elif stale_human_needs_map_refresh(ctx):
            emit_ai(
                "known human is map-stale - refreshing map before settling for %s",
                map_target["name"],
            )
            return search_for_enemies(ctx, ai_state=ai_state, map_reason="find_target")
    emit_ai(
        "map-known target %s (id=%d) at (%d,%d) - teleport-acquiring",
        map_target["name"],
        map_target["tank_id"],
        map_target["x"],
        map_target["y"],
    )
    return teleport_to_target(ctx, map_target)


__all__ = [
    "decide_hunt_acquire",
    "search_for_enemies",
]
