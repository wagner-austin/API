"""Durable HUNT owner and shared enemy-search helpers."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import (
    clear_combat_target,
    close_target,
    engage_target,
    get_locked_target,
    has_cardinal_combat_shot,
    is_already_engaged,
    open_map_for_target,
    select_new_combat_target,
    teleport_to_target,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_use_radar,
    make_decision,
    target_position_is_fresh,
)
from tankpit_bot.bot.ai.equipment import is_current_viewport_scanned
from tankpit_bot.bot.ai.threats import (
    analyze_threats,
    find_acquisition_target,
    find_locked_target_pursuit,
)
from tankpit_bot.bot.ai.types import AIStateDict, EnemyThreatDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_map_open_command, make_radar_command
from tankpit_bot.runtime_logging import emit_ai


def search_for_enemies(
    ctx: DecideCtx,
    *,
    ai_state: AIStateDict,
    map_reason: str,
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
    """Resume a held lock or acquire a fresh combat target.

    Resume-or-acquire cascade:

    1. **Resume held lock.** If ``combat_target_id != -1`` from a prior
       engagement that survived a fuel/equipment recovery cycle, prefer
       continuing that fight. If the locked target is in the current
       threat list, engage or close on it. If the lock is set but the
       target is off-viewport, fire homing toward the last wire
       position via ``_locked_target_pursuit`` -- same staying-put
       pursuit the in-fight substates use. This is what makes
       "restock-then-finish-the-kill" work: fuel and equipment recovery
       preserve ``combat_target_id`` rather than clearing it, and HUNT
       resumes the same engagement instead of re-acquiring fresh.
    2. **Strict (viewport-confirmed) threats.** ``analyze_threats``
       returns only enemies with recent ``last_viewport_observation_ms``;
       these are immediately fireable. If a viable one exists, pick it
       and teleport (or open the map first if the wire position is
       stale).
    3. **Loose (map-fresh) acquisition.** When no viewport-confirmed
       threat exists, look at every enemy whose ``timestamp_ms`` is
       within ``map_open_cooldown_ms`` (i.e. seen in a recent map
       snapshot). Teleport at the nearest viable one. ``SCAN_ON_LANDING``
       handles viewport confirmation before any shot, so map-only
       intel never produces a phantom firing.

    Only when all stages produce nothing does the bot dispatch
    another ``map_open``.
    """
    threats = _visible_threats(ctx)
    locked = get_locked_target(ctx, threats)
    if locked is not None:
        emit_ai("resuming held lock on %s (id=%d)", locked["name"], locked["tank_id"])
        if has_cardinal_combat_shot(ctx.self_state, locked):
            return engage_target(ctx, locked)
        return close_target(ctx, locked)
    pursuit = _locked_target_pursuit(ctx)
    if pursuit is not None:
        emit_ai(
            "resuming held lock on %s (id=%d) -- off viewport, firing toward last wire position",
            pursuit["name"],
            pursuit["tank_id"],
        )
        return engage_target(ctx, pursuit)

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
    )
    if map_target is not None:
        emit_ai(
            "map-known target %s (id=%d) at (%d,%d) - teleport-acquiring",
            map_target["name"],
            map_target["tank_id"],
            map_target["x"],
            map_target["y"],
        )
        return teleport_to_target(ctx, map_target)

    return search_for_enemies(
        ctx,
        ai_state=ctx.base,
        map_reason="find_enemies",
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
        if (
            has_cardinal_combat_shot(ctx.self_state, target)
            and can_use_radar(ctx)
            and not is_current_viewport_scanned(ctx.filtered)
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
        if has_cardinal_combat_shot(ctx.self_state, target):
            return engage_target(ctx, target)
        return close_target(ctx, target)
    pursuit = _locked_target_pursuit(ctx)
    if pursuit is not None:
        return _resume_locked_target_off_viewport(ctx, pursuit)
    return _enter_confirm_kill(ctx)


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
    return analyze_threats(ctx.filtered, ctx.self_state, ctx.timestamp_ms)


__all__ = [
    "decide_hunt_mode",
    "search_for_enemies",
]
