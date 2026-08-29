"""Held-lock machinery for HUNT: pursuit fire, resume, break escape.

Everything here operates on an EXISTING combat lock: firing at a
departed target's cached position (with the human homing cap), the
engaged-vs-never-engaged resume split, and the damage-aware
engagement break with its escape latch ([[bot-behavior-contract]]
§3.3). Fresh acquisition lives in :mod:`hunt_acquire`; relay travel
in :mod:`hunt_relay`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_break import (
    INCOMING_RATE_WINDOW_MS,
    assess_engagement_break,
)
from tankpit_bot.bot.ai.combat_close import teleport_to_target
from tankpit_bot.bot.ai.combat_opportunity import opportunity_shot_decision
from tankpit_bot.bot.ai.combat_prep import (
    open_map_for_target,
    refuel_for_hunt,
)
from tankpit_bot.bot.ai.combat_strategy import engage_target, frame_target_shift
from tankpit_bot.bot.ai.combat_target import (
    block_combat_target_and_replan,
    get_locked_target,
    is_already_engaged,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.mode_gates import human_fight_resume_fuel_floor
from tankpit_bot.bot.ai.threat_primitives import (
    pursuit_homing_budget_spent,
    pursuit_trace_is_live,
)
from tankpit_bot.bot.ai.threats import (
    analyze_threats,
    find_locked_target_pursuit,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.protocol.naming import is_human_name, is_practice_bot_name
from tankpit_bot.runtime_logging import (
    emit_ai,
    emit_diagnostic,
)


def visible_threats(ctx: DecideCtx) -> list[EnemyThreatDict]:
    """Return visible threats as a typed list for local routing.

    Args:
        ctx: Decision context.

    Returns:
        Visible enemy threats ordered by the threat analyzer.
    """
    return analyze_threats(
        ctx.ws,
        ctx.filtered,
        ctx.self_state,
        ctx.timestamp_ms,
        human_min_rank=ctx.config["human_target_min_rank"],
        human_max_rank=ctx.config["human_target_max_rank"],
    )


def locked_target_pursuit(ctx: DecideCtx) -> EnemyThreatDict | None:
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


def resume_locked_target_off_viewport(ctx: DecideCtx, pursuit: EnemyThreatDict) -> TickDecisionDict:
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
        return pursuit_fire(ctx, pursuit)
    emit_ai(
        "locked target %s never engaged - re-teleporting to (%d,%d)",
        pursuit["name"],
        pursuit["x"],
        pursuit["y"],
    )
    return teleport_to_target(ctx, pursuit)


def pursuit_fire(ctx: DecideCtx, pursuit: EnemyThreatDict) -> TickDecisionDict:
    """Fire at a departed target's cached position, or chase instead.

    Two gates before the shot:

    * **Trace wall** — the server's homing reroute dies ~12 s after
      the target left the viewport; past it a shot is a booked miss,
      so skip straight to the map chase the miss would have triggered
      anyway.
    * **Human homing cap (user ruling 2026-07-31)** — against a
      human-classified target, ONE pursuit shot per departure window:
      milking the reroute wall for all ~7 tracked hits "is cheating."
      Once the window's shot is stamped, further pursuit ticks chase
      via the map (refresh position, teleport, fight in person).
      Practice bots keep unlimited pursuit fire.

    Args:
        ctx: Decision context.
        pursuit: Pursuit threat synthesized from the world registry.

    Returns:
        Engage decision (stamping the pursuit-shot window), or the
        map-chase decision when either gate blocks the shot.
    """
    opportunity = opportunity_shot_decision(ctx, pursuit["tank_id"], ctx.base)
    if opportunity is not None:
        # Firefight doctrine (2026-08-14): a visible finisher or an
        # active attacker outranks a shot at a DEPARTED target's
        # cached position -- return fire to whoever is actually here,
        # then resume the chase.
        return opportunity
    shift = frame_target_shift(ctx, pursuit)
    if shift is not None:
        # Visibility law (flag s11-2): a cached position within one
        # scope shift's reach is exactly the zone where the server
        # refuses to homing-track ("close enough that a viewport
        # shift would reveal them"), so a pursuit shot here is a dead
        # weapon=0 miss AND would wrongly spend the human homing
        # budget. The free shift frames the target instead — checked
        # BEFORE the trace wall and the budget stamp.
        return shift
    if not pursuit_trace_is_live(ctx.filtered, pursuit["tank_id"], ctx.timestamp_ms):
        emit_ai(
            "homing trace on %s expired - chasing via map instead of a dead shot",
            pursuit["name"],
        )
        return open_map_for_target(ctx, pursuit)
    if is_human_name(pursuit["name"]) and pursuit_homing_budget_spent(
        ctx.filtered,
        pursuit["tank_id"],
        ctx.ai_state["pursuit_shot_target_id"],
        ctx.ai_state["pursuit_shot_ms"],
    ):
        emit_ai(
            "one pursuit homing already spent on %s this departure - chasing via map",
            pursuit["name"],
        )
        emit_diagnostic(
            diagnostic_kind="pursuit_homing_capped",
            target_id=pursuit["tank_id"],
            target_name=pursuit["name"],
        )
        return open_map_for_target(ctx, pursuit)
    emit_ai(
        "locked target %s left viewport - firing toward last wire position",
        pursuit["name"],
    )
    stamped_ctx = ctx.derive(
        AIStateDict(
            **{
                **ctx.ai_state,
                "pursuit_shot_target_id": pursuit["tank_id"],
                "pursuit_shot_ms": ctx.timestamp_ms,
            }
        )
    )
    return engage_target(stamped_ctx, pursuit)


def release_break_latch(ctx: DecideCtx) -> DecideCtx:
    """Return ctx with the break latch cleared once fuel recovers.

    Args:
        ctx: Decision context.

    Returns:
        The same context, or one whose AI state has the latch zeroed.
    """
    latch = ctx.ai_state["break_escape_until_fuel"]
    if latch <= 0 or ctx.fuel < latch:
        return ctx
    emit_ai("break latch released (fuel %d >= floor %d)", ctx.fuel, latch)
    return ctx.derive(AIStateDict(**{**ctx.ai_state, "break_escape_until_fuel": 0}))


def continue_break_escape(ctx: DecideCtx) -> TickDecisionDict | None:
    """Return the escape decision while the break latch is holding.

    Args:
        ctx: Decision context.

    Returns:
        Lock-held refuel decision, or ``None`` when no latch holds
        (or the escaped target is gone entirely).
    """
    latch = ctx.ai_state["break_escape_until_fuel"]
    if latch > 0:
        # A holding latch gates EVERY combat phase, not just ENGAGE.
        # Flag s2-7 (run bot-20260730-000030, 00:11:18-29): the latch
        # check lived only inside _break_losing_engagement, so CLOSE
        # ticks kept shooting orange-8 while the escape's larder hop
        # deferred for a map open -- four shoot/map_open cycles, fuel
        # 572->462 under fire, the exact oscillation the latch exists
        # to prevent. Gating at entry means no phase can trade shots
        # mid-escape and the deferred hop re-dispatches on the very
        # next tick against the opened map.
        escape_target = get_locked_target(ctx, visible_threats(ctx))
        if escape_target is None:
            escape_target = locked_target_pursuit(ctx)
        if escape_target is not None:
            emit_ai(
                "break latch holding for %s (fuel %d < floor %d) - continuing escape",
                escape_target["name"],
                ctx.fuel,
                latch,
            )
            return refuel_for_hunt(ctx, escape_target)
    return None


def assess_locked_engagement(ctx: DecideCtx) -> TickDecisionDict | None:
    """Run the break assessment for a held combat lock at phase entry.

    Args:
        ctx: Decision context.

    Returns:
        Break decision when the projection says the fight is lost,
        otherwise ``None``.
    """
    if ctx.ai_state["combat_target_id"] != -1:
        # The break ASSESSMENT gates every combat phase at entry, like
        # the latch above. It used to live only on the ENGAGE path, and
        # the Artax death (run bot-20260730-004144, 01:06:55, killed by
        # Yuppler) ran entirely in CLOSE: point-blank shot ticks looped
        # through _decide_hunt_close, no assessment ever ran, and the
        # first break of the whole fight fired at fuel 216 -- four
        # seconds before deactivation ("he just stood there and tanked
        # like 4 shots"). Assessed here, the first 3-hit window breaks
        # the fight near full fuel, while escape is still cheap.
        break_target = get_locked_target(ctx, visible_threats(ctx))
        if break_target is None:
            break_target = locked_target_pursuit(ctx)
        if break_target is not None:
            broken = _break_losing_engagement(ctx, break_target)
            if broken is not None:
                return broken
    return None


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
    hits, fuel_lost = ctx.ws.get_incoming_damage_window(ctx.timestamp_ms, INCOMING_RATE_WINDOW_MS)
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
    # Release level = the fuel at which the SAME projection clears the
    # floor: fuel_at_break + shortfall. Latching at the bare floor was
    # a zero-width band in disguise -- current fuel at break time is
    # usually far ABOVE the floor (the break trips on the PROJECTION),
    # so a floor-release cleared next tick and the latch was a no-op
    # (live receipts 23:23:45-55: three releases in ten seconds at
    # falling fuel 755/697/626, every one instantly re-broken).
    shortfall = assessment["escape_floor"] - assessment["projected_fuel_at_kill"]
    release_at = ctx.fuel + shortfall
    capacity = fuel_capacity(ctx.self_state["rank"])
    if release_at >= capacity:
        if not is_practice_bot_name(target["name"]):
            # A HUMAN fight is never "unwinnable" (user ruling
            # 2026-07-30, after the Artax death: "the bot can fight
            # against a human and win... it should have collected fuel
            # and then kept fighting and collected as necessary").
            # Human fights are attrition -- both sides refuel -- so
            # the one-kill projection that condemns a practice-bot
            # fight does not apply; blocking Yuppler at fuel 216 and
            # standing on the map was the death. Latch at the RESUME
            # floor, not capacity (user ruling 2026-07-31: "teleports
            # far away to restock and then comes back which isnt very
            # fun"): refuel just enough to fund the re-entry -- one
            # good container away -- and get back in the fight. The
            # floor stays above the half-capacity break band so the
            # hysteresis never inverts.
            resume_floor = human_fight_resume_fuel_floor(ctx)
            emit_ai(
                "human fight with %s projects past capacity (needs %d) - "
                "refuel to %d nearby and resume",
                target["name"],
                release_at,
                resume_floor,
            )
            human_latched_ctx = ctx.derive(
                AIStateDict(
                    **{
                        **ctx.ai_state,
                        "break_escape_until_fuel": resume_floor,
                    }
                )
            )
            return refuel_for_hunt(human_latched_ctx, target)
        # Even a full tank cannot fund this fight -- the projection
        # fails at every reachable fuel level, so latching would pin
        # the bot in COLLECT forever. Block with the standard TTL and
        # replan; the cooldown retries once the fight geometry may
        # have changed.
        emit_ai(
            "engagement with %s unwinnable at any fuel (needs %d, capacity %d) - blocking",
            target["name"],
            release_at,
            capacity,
        )
        return block_combat_target_and_replan(ctx, target)
    latched_ctx = ctx.derive(
        AIStateDict(
            **{
                **ctx.ai_state,
                "break_escape_until_fuel": release_at,
            }
        )
    )
    return refuel_for_hunt(latched_ctx, target)


__all__ = [
    "assess_locked_engagement",
    "continue_break_escape",
    "locked_target_pursuit",
    "pursuit_fire",
    "release_break_latch",
    "resume_locked_target_off_viewport",
    "visible_threats",
]
