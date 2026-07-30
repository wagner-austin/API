"""Durable HUNT owner and shared enemy-search helpers."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_break import (
    INCOMING_RATE_WINDOW_MS,
    assess_engagement_break,
)
from tankpit_bot.bot.ai.combat_strategy import (
    clear_combat_target,
    close_target,
    engage_target,
    get_locked_target,
    has_cardinal_combat_shot,
    is_already_engaged,
    open_map_for_target,
    refuel_for_hunt,
    select_new_combat_target,
    teleport_to_target,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_use_radar,
    make_decision,
    target_position_is_fresh,
)
from tankpit_bot.bot.ai.humans import is_human_name, is_practice_bot_name
from tankpit_bot.bot.ai.resource_search import make_resource_search_hop
from tankpit_bot.bot.ai.threats import (
    analyze_threats,
    find_acquisition_target,
    find_locked_target_pursuit,
    find_relay_travel_target,
    stale_human_exists,
)
from tankpit_bot.bot.ai.types import AIStateDict, EnemyThreatDict, ReasonKind
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    make_map_open_command,
    make_radar_command,
    make_teleport_command,
)
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.sniffer.world_state import get_incoming_damage_window
from tankpit_bot.state.scan_coverage import is_viewport_fully_covered
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds


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


def decide_hunt_mode(ctx: DecideCtx) -> TickDecisionDict:
    """Run the durable ``HUNT`` owner for this tick.

    Args:
        ctx: Decision context.

    Returns:
        Mode-owned hunt decision.
    """
    if ctx.mode_state == "CONFIRM_KILL":
        return _decide_hunt_confirm_kill(ctx)
    if ctx.mode_state == "SCAN_ON_LANDING":
        return _decide_hunt_scan_on_landing(ctx)
    if ctx.mode_state == "ENGAGE":
        return _decide_hunt_engage(ctx)
    if ctx.mode_state == "CLOSE":
        return _decide_hunt_close(ctx)
    if ctx.mode_state == "REFRESH":
        return _decide_hunt_refresh(ctx)
    return _decide_hunt_acquire(ctx)


def _decide_hunt_acquire(ctx: DecideCtx) -> TickDecisionDict:
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
    threats = _visible_threats(ctx)
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
        pursuit = _locked_target_pursuit(ctx)
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
                relay = _relay_toward(ctx, ctx.base, pursuit)
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
        if is_practice_bot_name(map_target["name"]):
            # Unlimited-distance human pursuit (user ruling 2026-07-29):
            # before farming an affordable bot, check whether a
            # rank-window human is on the fresh map beyond the
            # affordability horizon -- if so, the relay chain toward
            # THEM outranks the bot. (An affordable human would already
            # be the acquisition winner via the tier sort.)
            human_travel = _human_pursuit_travel_target(ctx)
            if human_travel is not None:
                emit_ai(
                    "human %s (id=%d) at dist %d outranks affordable bot %s - "
                    "relaying toward them (unlimited-distance human pursuit)",
                    human_travel["name"],
                    human_travel["tank_id"],
                    human_travel["distance"],
                    map_target["name"],
                )
                relay = _relay_toward(ctx, ai_state, human_travel)
                if relay is not None:
                    return relay
                emit_ai(
                    "no dot or refuel leg helps toward %s right now - "
                    "farming %s while the map evolves",
                    human_travel["name"],
                    map_target["name"],
                )
            elif _stale_human_needs_map_refresh(ctx):
                # Freshness asymmetry (Yuppler, 2026-07-29 21:19): bots
                # stay wire-fresh by moving; a quiet human goes stale
                # 5 s after each map open, and with a fresh bot always
                # available the map never got reopened -- the human
                # was invisible outside 5-second windows. A known
                # rank-window human whose only curable defect is stale
                # map data forces a refresh BEFORE the bot may settle
                # for farming; the cooldown gate bounds the cadence.
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

    map_age_ms = ctx.timestamp_ms - ctx.ai_state["last_map_open_ms"]
    if ctx.ai_state["last_map_open_ms"] > 0 and map_age_ms <= ctx.config["map_open_cooldown_ms"]:
        relay = _relay_toward_unaffordable_enemy(ctx, ai_state)
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


def _relay_toward_unaffordable_enemy(
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
    travel = find_relay_travel_target(
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
    if travel is None:
        return None
    return _relay_toward(ctx, ai_state, travel)


def _relay_toward(
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


def _stale_human_needs_map_refresh(ctx: DecideCtx) -> bool:
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
        human_min_rank=ctx.config["human_target_min_rank"],
        human_max_rank=ctx.config["human_target_max_rank"],
    )


def _human_pursuit_travel_target(ctx: DecideCtx) -> EnemyThreatDict | None:
    """Return the human the relay should pursue, or ``None``.

    User ruling 2026-07-29 ("unlimited distance for humans... this is
    the real deal"): a rank-window human on the map outranks every
    affordable practice bot at TARGET SELECTION -- even when reaching
    them takes a fuel-dot relay chain across the whole field. Locks
    are still never switched mid-fight (same-day follow-up: "finish
    the kill then the human player will be the next target"); this
    helper only fires during fresh acquisition.

    :func:`find_relay_travel_target` already sorts human-tier first,
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
    travel = find_relay_travel_target(
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
    if travel is not None and is_human_name(travel["name"]):
        return travel
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


def _resume_locked_target_off_viewport(
    ctx: DecideCtx, pursuit: EnemyThreatDict
) -> TickDecisionDict:
    """Decide pursuit-fire vs re-teleport when the locked target is off-viewport.

    Distinguishes two physically different situations that the HUNT
    cascade otherwise collapses into the same state:

    * **Engaged-then-gone** — the bot teleported to the target, fired
      at least one shot, the target then moved or teleported away.
      Pursuit homing toward the last wire position is correct; the
      server picks ``homing`` when out of point-blank range and
      homing tracks (user contract 2026-06-22).
    * **Never-engaged** — the planner set the lock and emitted a
      teleport, but the executor swapped the teleport for a
      pre-teleport ``map_open`` (see ``executor.py:158-171``). The
      teleport never dispatched; the planner re-ran in HUNT/CLOSE
      and fell straight to pursuit at the still-distant last wire
      position. Live run 2026-06-23 21:36:31: lock formed at
      bot=(46,100), red-4 at (56,177); ``map_open`` took 6116ms;
      next decision fired ``shoot`` at dist=87 and looped 19+ times.

    The discriminating signal is :func:`is_already_engaged` --
    ``last_shot_target_id`` is set only when the bot has actually
    dispatched a ``shoot`` at this target id, so a mismatch with
    ``combat_target_id`` proves the bot never engaged this target
    and the lock is the pre-engagement intent, not a mid-fight
    chase. Re-issue the teleport in that case so the original
    intent isn't silently dropped.

    Args:
        ctx: Decision context.
        pursuit: Pursuit threat synthesized from the world registry.

    Returns:
        ``teleport`` decision when the bot has not yet engaged this
        target (re-issues the deferred intent), otherwise an
        ``engage`` decision that fires homing at the last wire
        position.
    """
    if is_already_engaged(ctx):
        emit_ai(
            "locked target %s left viewport - firing toward last wire position",
            pursuit["name"],
        )
        return engage_target(ctx, pursuit)
    emit_ai(
        "locked target %s never engaged - re-teleporting to (%d,%d)",
        pursuit["name"],
        pursuit["x"],
        pursuit["y"],
    )
    return teleport_to_target(ctx, pursuit)


def _locked_target_pursuit(ctx: DecideCtx) -> EnemyThreatDict | None:
    """Return the locked target as a pursuit threat when they left the viewport.

    Behavior contract (user-confirmed 2026-06-22): when a locked
    target teleports out of view, the bot does NOT chase and does
    NOT enter CONFIRM_KILL on first viewport-miss. Instead it stays
    put and keeps firing at the target's last known wire position --
    the server picks ``homing`` when the target is mid-move or out
    of point-blank range, and homing tracks. The lock holds until
    an actual deactivation signal arrives (liveness flips, or the
    tank lands in ``killed_tank_ids``).

    Args:
        ctx: Decision context.

    Returns:
        Pursuit ``EnemyThreatDict`` synthesised from the wire
        registry, or ``None`` when the locked target is truly gone
        (id cleared, dead, or position too stale).
    """
    return find_locked_target_pursuit(
        ctx.filtered,
        ctx.self_state,
        ctx.ai_state["combat_target_id"],
        ctx.killed,
    )


def _decide_hunt_scan_on_landing(ctx: DecideCtx) -> TickDecisionDict:
    """Engage the target after the combat-landing scan completed."""
    threats = _visible_threats(ctx)
    target = get_locked_target(ctx, threats)
    if target is not None:
        if has_cardinal_combat_shot(ctx.self_state, target):
            return engage_target(ctx, target)
        return close_target(ctx, target)
    pursuit = _locked_target_pursuit(ctx)
    if pursuit is not None:
        emit_ai(
            "locked target %s left viewport - firing toward last wire position",
            pursuit["name"],
        )
        return engage_target(ctx, pursuit)
    return _decide_hunt_acquire(ctx)


def _decide_hunt_refresh(ctx: DecideCtx) -> TickDecisionDict:
    """Refresh target information before closing or engaging."""
    threats = _visible_threats(ctx)
    target = get_locked_target(ctx, threats)
    if target is not None:
        if has_cardinal_combat_shot(ctx.self_state, target):
            return engage_target(ctx, target)
        return close_target(ctx, target)
    pursuit = _locked_target_pursuit(ctx)
    if pursuit is not None:
        return _resume_locked_target_off_viewport(ctx, pursuit)
    return _decide_hunt_acquire(ctx)


def _decide_hunt_close(ctx: DecideCtx) -> TickDecisionDict:
    """Close distance on the locked combat target."""
    threats = _visible_threats(ctx)
    target = get_locked_target(ctx, threats)
    if target is not None:
        viewport_left, viewport_top, viewport_right, viewport_bottom = viewport_visible_bounds(
            ctx.filtered["viewport"],
        )
        if (
            has_cardinal_combat_shot(ctx.self_state, target)
            and can_use_radar(ctx)
            and not is_viewport_fully_covered(
                ctx.filtered["scanned_tiles"],
                viewport_left,
                viewport_top,
                viewport_right,
                viewport_bottom,
                ctx.timestamp_ms,
            )
        ):
            emit_ai("landed adjacent to %s, scanning viewport first", target["name"])
            return make_decision(
                make_radar_command(),
                "HUNT",
                800,
                target["x"],
                target["y"],
                "scan_on_landing",
                AIStateDict(
                    **{
                        **ctx.base,
                        "last_scan_ms": ctx.timestamp_ms,
                        # Record the per-viewport landing-scan latch so
                        # a later COLLECT entry in the same viewport
                        # does not fire a second landing radar.
                        "last_landing_scan_viewport": f"{viewport_left},{viewport_top}",
                    }
                ),
                ctx.equip,
            )
        return close_target(ctx, target)
    pursuit = _locked_target_pursuit(ctx)
    if pursuit is not None:
        return _resume_locked_target_off_viewport(ctx, pursuit)
    return _enter_confirm_kill(ctx)


def _decide_hunt_engage(ctx: DecideCtx) -> TickDecisionDict:
    """Engage the locked combat target or confirm its disappearance."""
    threats = _visible_threats(ctx)
    target = get_locked_target(ctx, threats)
    if target is not None:
        broken = _break_losing_engagement(ctx, target)
        if broken is not None:
            return broken
        if has_cardinal_combat_shot(ctx.self_state, target):
            return engage_target(ctx, target)
        return close_target(ctx, target)
    pursuit = _locked_target_pursuit(ctx)
    if pursuit is not None:
        broken = _break_losing_engagement(ctx, pursuit)
        if broken is not None:
            return broken
        return _resume_locked_target_off_viewport(ctx, pursuit)
    return _enter_confirm_kill(ctx)


def _break_losing_engagement(
    ctx: DecideCtx,
    target: EnemyThreatDict,
) -> TickDecisionDict | None:
    """Break the fight when finishing it is projected to strand the tank.

    The damage-aware break ([[bot-behavior-contract]] §3.3, built after
    run bot-20260728-075336 lost 1050→108 fuel in 30 s of two-attacker
    fire): the measured incoming rate and the target's remaining health
    project the fuel at kill completion; when that projection falls
    below the escape floor the tick delegates to ``refuel_for_hunt``
    -- the lock survives (never-drop) and COLLECT's larder step aims
    the escape at known fuel. A near-death target projects cheap and
    is finished first; a quiet fight (rate 0) never breaks here.

    Args:
        ctx: Decision context.
        target: The engaged (or pursued) enemy.

    Returns:
        The lock-held refuel delegation, or ``None`` to keep fighting.
    """
    hits, fuel_lost = get_incoming_damage_window(ctx.timestamp_ms, INCOMING_RATE_WINDOW_MS)
    assessment = assess_engagement_break(ctx, target, hits, fuel_lost)
    if not assessment["break_engagement"]:
        return None
    emit_ai(
        "breaking engagement with %s: projected fuel %d at kill < floor %d "
        "(incoming %d/tick over %d hits, %d hits to kill, fuel=%d)",
        target["name"],
        assessment["projected_fuel_at_kill"],
        assessment["escape_floor"],
        assessment["incoming_rate_per_tick"],
        assessment["hits_in_window"],
        assessment["hits_to_kill"],
        ctx.fuel,
    )
    emit_diagnostic(
        diagnostic_kind="engagement_break",
        target_id=target["tank_id"],
        target_name=target["name"],
        fuel=ctx.fuel,
        hits_in_window=assessment["hits_in_window"],
        incoming_fuel_in_window=assessment["incoming_fuel_in_window"],
        incoming_rate_per_tick=assessment["incoming_rate_per_tick"],
        hits_to_kill=assessment["hits_to_kill"],
        projected_fuel_at_kill=assessment["projected_fuel_at_kill"],
        escape_floor=assessment["escape_floor"],
    )
    return refuel_for_hunt(ctx, target)


def _decide_hunt_confirm_kill(ctx: DecideCtx) -> TickDecisionDict:
    """Leave confirm-kill by reacquiring with cleared combat state."""
    cleared = clear_combat_target(ctx.base)
    cleared_ctx = DecideCtx(
        ctx.world,
        ctx.self_state,
        cleared,
        ctx.inventory,
        ctx.timestamp_ms,
        ctx.terrain,
        ctx.combat_feedback,
        ctx.map_fuel_dots,
    )
    return _decide_hunt_acquire(cleared_ctx)


def _enter_confirm_kill(ctx: DecideCtx) -> TickDecisionDict:
    """Emit an explicit confirm-kill transition and clear combat target state."""
    target_id = ctx.ai_state["combat_target_id"]
    if target_id != -1 and str(target_id) in ctx.killed:
        emit_ai("target %d entered kill cooldown; confirming kill", target_id)
    elif target_id != -1:
        emit_ai("locked target %d disappeared; confirming kill before reacquire", target_id)
    cleared = clear_combat_target(ctx.base)
    return search_for_enemies(
        ctx,
        ai_state=cleared,
        map_reason="confirm_kill",
    )


def _visible_threats(ctx: DecideCtx) -> list[EnemyThreatDict]:
    """Return visible threats as a typed list for local routing.

    Args:
        ctx: Decision context.

    Returns:
        Visible enemy threats ordered by the threat analyzer.
    """
    return analyze_threats(
        ctx.filtered,
        ctx.self_state,
        ctx.timestamp_ms,
        human_min_rank=ctx.config["human_target_min_rank"],
        human_max_rank=ctx.config["human_target_max_rank"],
    )


__all__ = [
    "decide_hunt_mode",
    "search_for_enemies",
]
