"""Durable HUNT owner: phase routing over the lock/acquire machinery.

``decide_hunt_mode`` gates every tick with the break latch and break
assessment (both in :mod:`hunt_lock` -- the Artax death proved
per-phase checks always leave a phase uncovered), then routes to the
phase deciders below. Fresh acquisition lives in :mod:`hunt_acquire`;
held-lock pursuit and escape in :mod:`hunt_lock`; relay travel in
:mod:`hunt_relay`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_close import close_target
from tankpit_bot.bot.ai.combat_strategy import (
    engage_target,
    has_cardinal_combat_shot,
)
from tankpit_bot.bot.ai.combat_target import (
    clear_combat_target,
    get_locked_target,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    make_decision,
    radar_spend_worthwhile,
)
from tankpit_bot.bot.ai.hunt_acquire import decide_hunt_acquire, search_for_enemies
from tankpit_bot.bot.ai.hunt_lock import (
    assess_locked_engagement,
    continue_break_escape,
    locked_target_pursuit,
    pursuit_fire,
    release_break_latch,
    resume_locked_target_off_viewport,
    visible_threats,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_radar_command
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds


def decide_hunt_mode(ctx: DecideCtx) -> TickDecisionDict:
    """Run the durable ``HUNT`` owner for this tick.

    Args:
        ctx: Decision context.

    Returns:
        Mode-owned hunt decision.
    """
    ctx = release_break_latch(ctx)
    if ctx.mode_state == "CONFIRM_KILL":
        return _decide_hunt_confirm_kill(ctx)
    escape = continue_break_escape(ctx)
    if escape is not None:
        return escape
    broken = assess_locked_engagement(ctx)
    if broken is not None:
        return broken
    if ctx.mode_state == "SCAN_ON_LANDING":
        return _decide_hunt_scan_on_landing(ctx)
    if ctx.mode_state == "ENGAGE":
        return _decide_hunt_engage(ctx)
    if ctx.mode_state == "CLOSE":
        return _decide_hunt_close(ctx)
    if ctx.mode_state == "REFRESH":
        return _decide_hunt_refresh(ctx)
    return decide_hunt_acquire(ctx)


def _decide_hunt_scan_on_landing(ctx: DecideCtx) -> TickDecisionDict:
    """Engage the target after the combat-landing scan completed."""
    threats = visible_threats(ctx)
    target = get_locked_target(ctx, threats)
    if target is not None:
        if has_cardinal_combat_shot(ctx.self_state, target):
            return engage_target(ctx, target)
        return close_target(ctx, target)
    pursuit = locked_target_pursuit(ctx)
    if pursuit is not None:
        return pursuit_fire(ctx, pursuit)
    return decide_hunt_acquire(ctx)


def _decide_hunt_refresh(ctx: DecideCtx) -> TickDecisionDict:
    """Refresh target information before closing or engaging."""
    threats = visible_threats(ctx)
    target = get_locked_target(ctx, threats)
    if target is not None:
        if has_cardinal_combat_shot(ctx.self_state, target):
            return engage_target(ctx, target)
        return close_target(ctx, target)
    pursuit = locked_target_pursuit(ctx)
    if pursuit is not None:
        return resume_locked_target_off_viewport(ctx, pursuit)
    return decide_hunt_acquire(ctx)


def _decide_hunt_close(ctx: DecideCtx) -> TickDecisionDict:
    """Close distance on the locked combat target."""
    threats = visible_threats(ctx)
    target = get_locked_target(ctx, threats)
    if target is not None:
        viewport_left, viewport_top, _, _ = viewport_visible_bounds(ctx.filtered["viewport"])
        if (
            has_cardinal_combat_shot(ctx.self_state, target)
            # Shared radar-spend economics (s9 flags): a stocked scan
            # must buy a real reveal, not a coverage sliver -- the
            # fully-covered check alone let 1-tile holes spend extras
            # mid-combat.
            and radar_spend_worthwhile(ctx)
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
    pursuit = locked_target_pursuit(ctx)
    if pursuit is not None:
        return resume_locked_target_off_viewport(ctx, pursuit)
    return _enter_confirm_kill(ctx)


def _decide_hunt_engage(ctx: DecideCtx) -> TickDecisionDict:
    """Engage the locked combat target or confirm its disappearance.

    The break assessment and escape latch both gate at
    :func:`decide_hunt_mode` entry (the Artax death proved per-phase
    checks always leave a phase uncovered), so this path engages
    without re-assessing.
    """
    threats = visible_threats(ctx)
    target = get_locked_target(ctx, threats)
    if target is not None:
        if has_cardinal_combat_shot(ctx.self_state, target):
            return engage_target(ctx, target)
        return close_target(ctx, target)
    pursuit = locked_target_pursuit(ctx)
    if pursuit is not None:
        return resume_locked_target_off_viewport(ctx, pursuit)
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
        ctx.map_fuel_dots,
        ws=ctx.ws,
    )
    return decide_hunt_acquire(cleared_ctx)


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


__all__ = [
    "decide_hunt_mode",
]
