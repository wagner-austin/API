"""Durable HUNT owner and shared enemy-search helpers."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import (
    clear_combat_target,
    close_target,
    engage_target,
    get_locked_target,
    has_cardinal_combat_shot,
    open_map_for_target,
    select_new_combat_target,
    teleport_to_target,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_use_radar,
    has_recent_map_snapshot,
    is_current_viewport_scan_failed,
    make_decision,
)
from tankpit_bot.bot.ai.equipment import is_current_viewport_scanned
from tankpit_bot.bot.ai.movement import select_exploration_command
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.types import AIStateDict, EnemyThreatDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_map_open_command, make_radar_command
from tankpit_bot.runtime_logging import emit_ai


def search_for_enemies(
    ctx: DecideCtx,
    *,
    ai_state: AIStateDict,
    map_reason: str,
    radar_reason: str,
    edge_reason: str,
) -> TickDecisionDict:
    """Search for enemies with explicit search-stage reasons.

    Args:
        ctx: Decision context.
        ai_state: Base AI state to rewrite for the produced command.
        map_reason: Behavior reason to use for a map-open refresh.
        radar_reason: Behavior reason to use for a radar scan.
        edge_reason: Behavior reason to use for viewport-edge repositioning.

    Returns:
        Enemy-search decision using map-open, radar, or edge movement.
    """
    map_age = ctx.timestamp_ms - ctx.ai_state["last_map_open_ms"]
    if map_age < ctx.config["map_open_cooldown_ms"]:
        scan_age = ctx.timestamp_ms - ctx.ai_state["last_scan_ms"]
        if (
            scan_age >= ctx.config["scan_cooldown_ms"]
            and can_use_radar(ctx)
            and not is_current_viewport_scanned(ctx.filtered)
            and not is_current_viewport_scan_failed(ctx)
        ):
            emit_ai("radar to search for enemies")
            return make_decision(
                make_radar_command(),
                "HUNT",
                0,
                0,
                0,
                radar_reason,
                AIStateDict(
                    **{
                        **ai_state,
                        "last_scan_ms": ctx.timestamp_ms,
                    }
                ),
                ctx.equip,
            )

        emit_ai("walk to viewport edge while searching for enemies")
        exploration = select_exploration_command(ctx)
        if exploration is not None:
            edge_x, edge_y, edge_command = exploration
            return make_decision(
                edge_command,
                "HUNT",
                0,
                edge_x,
                edge_y,
                edge_reason,
                ai_state,
                ctx.equip,
            )

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
    if ctx.mode_state == "ENGAGE":
        return _decide_hunt_engage(ctx)
    if ctx.mode_state == "CLOSE":
        return _decide_hunt_close(ctx)
    if ctx.mode_state == "REFRESH":
        return _decide_hunt_refresh(ctx)
    return _decide_hunt_acquire(ctx)


def _decide_hunt_acquire(ctx: DecideCtx) -> TickDecisionDict:
    """Acquire a fresh combat target or fall back to enemy search."""
    threats = _visible_threats(ctx)
    target = select_new_combat_target(ctx, threats)
    if target is not None:
        emit_ai("new target %s (id=%d)", target["name"], target["tank_id"])
        if _has_recent_map_snapshot(ctx):
            emit_ai("fresh map intel available - teleporting to %s", target["name"])
            return teleport_to_target(ctx, target)
        return open_map_for_target(ctx, target)
    return search_for_enemies(
        ctx,
        ai_state=ctx.base,
        map_reason="find_enemies",
        radar_reason="radar_for_enemies",
        edge_reason="edge_for_enemies",
    )


def _decide_hunt_refresh(ctx: DecideCtx) -> TickDecisionDict:
    """Refresh target information before closing or engaging."""
    threats = _visible_threats(ctx)
    target = get_locked_target(ctx, threats)
    if target is None:
        return _decide_hunt_acquire(ctx)
    if has_cardinal_combat_shot(ctx.self_state, target):
        return engage_target(ctx, target)
    return close_target(ctx, target)


def _decide_hunt_close(ctx: DecideCtx) -> TickDecisionDict:
    """Close distance on the locked combat target."""
    threats = _visible_threats(ctx)
    target = get_locked_target(ctx, threats)
    if target is None:
        return _enter_confirm_kill(ctx)
    return close_target(ctx, target)


def _decide_hunt_engage(ctx: DecideCtx) -> TickDecisionDict:
    """Engage the locked combat target or confirm its disappearance."""
    threats = _visible_threats(ctx)
    target = get_locked_target(ctx, threats)
    if target is None:
        return _enter_confirm_kill(ctx)
    return engage_target(ctx, target)


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
        radar_reason="confirm_kill",
        edge_reason="confirm_kill",
    )


def _visible_threats(ctx: DecideCtx) -> list[EnemyThreatDict]:
    """Return visible threats as a typed list for local routing.

    Args:
        ctx: Decision context.

    Returns:
        Visible enemy threats ordered by the threat analyzer.
    """
    return analyze_threats(ctx.filtered, ctx.self_state, ctx.timestamp_ms)


def _has_recent_map_snapshot(ctx: DecideCtx) -> bool:
    """Return True when the map-open snapshot is still fresh for hunt routing."""
    return has_recent_map_snapshot(ctx)


__all__ = [
    "decide_hunt_mode",
    "search_for_enemies",
]
