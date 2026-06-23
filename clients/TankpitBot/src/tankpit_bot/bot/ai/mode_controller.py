"""Durable top-level AI mode helpers and migration rules."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    combat_reserve_restored,
)
from tankpit_bot.bot.ai.modes import AIMode, AIModeState, is_valid_ai_mode_state
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision


def clear_ai_mode(ai_state: AIStateDict) -> AIStateDict:
    """Return AI state with durable mode ownership cleared.

    Args:
        ai_state: Current AI state.

    Returns:
        AI state with mode ownership reset to ``UNSET``.
    """
    return AIStateDict(
        **{
            **ai_state,
            "mode": "UNSET",
            "mode_state": "",
            "mode_started_ms": 0,
        }
    )


def set_ai_mode(
    ai_state: AIStateDict,
    mode: AIMode,
    mode_state: AIModeState,
    timestamp_ms: int,
) -> AIStateDict:
    """Return AI state with a validated durable mode assignment.

    Args:
        ai_state: Current AI state.
        mode: Durable top-level mode.
        mode_state: Substate within the durable mode.
        timestamp_ms: Current tick timestamp in milliseconds.

    Returns:
        AI state with updated mode ownership.

    Raises:
        ValueError: If the mode/substate pair is invalid.
    """
    if not is_valid_ai_mode_state(mode, mode_state):
        raise ValueError(f"Invalid AI mode/state pair: {mode}/{mode_state}")
    started_ms = 0
    if mode != "UNSET":
        started_ms = ai_state["mode_started_ms"] if ai_state["mode"] == mode else timestamp_ms
    return AIStateDict(
        **{
            **ai_state,
            "mode": mode,
            "mode_state": mode_state,
            "mode_started_ms": started_ms,
        }
    )


def clear_mode_on_decision(decision: TickDecisionDict) -> TickDecisionDict:
    """Return a decision whose updated AI state has no durable owner.

    Args:
        decision: Planner decision to rewrite.

    Returns:
        Decision with durable mode ownership cleared.
    """
    return make_tick_decision(
        command=decision["command"],
        behavior=decision["behavior"],
        updated_ai_state=clear_ai_mode(decision["updated_ai_state"]),
        desired_equipment=decision["desired_equipment"],
    )


def apply_mode_to_decision(
    decision: TickDecisionDict,
    mode: AIMode,
    mode_state: AIModeState,
    timestamp_ms: int,
) -> TickDecisionDict:
    """Return a decision whose updated AI state carries durable mode ownership.

    Args:
        decision: Planner decision to rewrite.
        mode: Durable top-level mode.
        mode_state: Durable substate within the mode.
        timestamp_ms: Current tick timestamp in milliseconds.

    Returns:
        Decision with durable mode ownership applied.
    """
    return make_tick_decision(
        command=decision["command"],
        behavior=decision["behavior"],
        updated_ai_state=set_ai_mode(decision["updated_ai_state"], mode, mode_state, timestamp_ms),
        desired_equipment=decision["desired_equipment"],
    )


def should_enter_recover_fuel(ctx: DecideCtx) -> bool:
    """Return True when fuel recovery should own planning.

    Args:
        ctx: Decision context.

    Returns:
        True when fuel is at or below the low-threshold entry rule.
    """
    return ctx.fuel <= ctx.config["fuel_low_threshold"]


def should_exit_recover_fuel(ctx: DecideCtx) -> bool:
    """Return True when fuel recovery can release control.

    Args:
        ctx: Decision context.

    Returns:
        True when fuel has recovered to the full-threshold exit rule.
    """
    return ctx.fuel >= ctx.config["fuel_full_threshold"]


def needs_radar_restock(ctx: DecideCtx) -> bool:
    """Return True while extra radars sit below the healthy buffer.

    Radars find both enemies and equipment, so the bot rebuilds a
    healthy stock before returning to the hunt rather than fighting
    blind at one or two. This is the exit-side (resume) predicate; it
    deliberately ignores visible threats, because rebuilding the kit
    outranks chasing wanderers the bot cannot find or beat without
    radar (live run 20260613-011044: looped 0->3->2->1, never built a
    buffer, because it ran off to fight at the first radar).

    Args:
        ctx: Decision context.

    Returns:
        True when the extra-radar count is below the resume threshold.
    """
    return ctx.inventory["extra_radars"]["count"] < ctx.config["radar_resume_threshold"]


def should_enter_recover_equipment(ctx: DecideCtx) -> bool:
    """Return True when equipment recovery should own planning.

    Per the user-defined gameplay loop ("restock -> find enemy ->
    fight -> kill -> restock -> hunt"), restock-mode entry has two
    tiers:

    * **Emergency** -- any reserve below its *break* threshold
      (4 duals / 4 homings / 5 radars). The bot can't fight without
      ammo; restock interrupts even an active combat target.
    * **Between kills** -- any reserve below its *resume* threshold
      (25 / 25 / 20) AND no active combat target. The bot finishes
      the current kill first, then restocks before the next hunt.

    The "active combat target" gate is the difference between
    "restock, fight, restock" and "restock, shoot ONCE, restock":
    a single missed-or-hit shot drops dual from 25 to 24, which
    trips the resume threshold but should NOT abandon the in-flight
    kill -- the bot finishes the engagement, THEN restocks for the
    next one.

    Args:
        ctx: Decision context.

    Returns:
        True when the bot is below the emergency break threshold,
        or when between kills with reserves below the resume
        threshold.
    """
    if (
        ctx.inventory["dual_shots"]["count"] < ctx.config["dual_break_threshold"]
        or ctx.inventory["homing_shots"]["count"] < ctx.config["dual_break_threshold"]
        or ctx.inventory["extra_radars"]["count"] < ctx.config["radar_break_threshold"]
    ):
        return True
    if ctx.ai_state["combat_target_id"] != -1:
        return False
    return (
        ctx.inventory["dual_shots"]["count"] < ctx.config["dual_resume_threshold"]
        or ctx.inventory["homing_shots"]["count"] < ctx.config["dual_resume_threshold"]
        or ctx.inventory["extra_radars"]["count"] < ctx.config["radar_resume_threshold"]
    )


def should_exit_recover_equipment(ctx: DecideCtx) -> bool:
    """Return True when equipment recovery can release control.

    The mode holds until BOTH reserves are healthy: weapons back above
    the resume threshold AND radars rebuilt to their resume buffer.
    The break/resume gap gives hysteresis -- entry at the low break,
    exit only at the higher resume -- so the bot rebuilds to a full
    stock instead of leaving the moment it scrapes together one radar.

    Args:
        ctx: Decision context.

    Returns:
        True when weapon reserves are restored and radars are no longer
        below the resume buffer.
    """
    return combat_reserve_restored(ctx) and not needs_radar_restock(ctx)


def should_enter_hunt(ctx: DecideCtx) -> bool:
    """Return True when HUNT is the valid top-level owner.

    The bot only hunts when properly stocked — weapon reserves above the
    resume threshold and no fuel/equipment recovery needed. Starting a
    fight with low ammo leads to abandoned kills when the break
    threshold pulls the bot away mid-fight.

    Args:
        ctx: Decision context.

    Returns:
        True when combat-ready and no recovery mode has stronger entry.
    """
    if should_enter_recover_fuel(ctx):
        return False
    if should_enter_recover_equipment(ctx):
        return False
    return combat_reserve_restored(ctx)


def should_exit_hunt(ctx: DecideCtx) -> bool:
    """Return True when HUNT should release control.

    Args:
        ctx: Decision context.

    Returns:
        True when a recovery mode now has stronger entry conditions.
    """
    return not should_enter_hunt(ctx)


def derive_hunt_mode_state(decision: TickDecisionDict) -> AIModeState:
    """Derive the HUNT substate from planner output.

    Args:
        decision: Decision produced by the hunt-owner path.

    Returns:
        Derived HUNT substate for the updated AI state.
    """
    command_type = decision["command"]["cmd_type"]
    reason = decision["behavior"]["reason"]
    has_locked_target = decision["updated_ai_state"]["combat_target_id"] != -1
    if reason == "confirm_kill":
        return "CONFIRM_KILL"
    if reason == "scan_on_landing":
        return "SCAN_ON_LANDING"
    if command_type == "shoot":
        return "ENGAGE"
    if command_type in ("teleport", "move"):
        if has_locked_target:
            return "CLOSE"
        return "ACQUIRE"
    if command_type in ("map_open", "radar"):
        if reason == "find_enemies":
            return "ACQUIRE"
        if has_locked_target:
            return "REFRESH"
        return "ACQUIRE"
    return "ACQUIRE"


def derive_recover_equipment_mode_state(decision: TickDecisionDict) -> AIModeState:
    """Derive the ``RECOVER_EQUIPMENT`` substate from planner output.

    Args:
        decision: Decision produced by the equipment recovery owner path.

    Returns:
        Derived equipment recovery substate for the updated AI state.
    """
    reason = decision["behavior"]["reason"]
    command_type = decision["command"]["cmd_type"]
    if reason in ("forage_radar", "forage_sweep"):
        return "SENSE"
    if reason == "search_equipment_local":
        return "SEARCH"
    if command_type == "pickup_equipment":
        return "PICKUP"
    return "APPROACH"


def derive_recover_fuel_mode_state(decision: TickDecisionDict) -> AIModeState:
    """Derive the ``RECOVER_FUEL`` substate from planner output.

    Args:
        decision: Decision produced by the fuel recovery owner path.

    Returns:
        Derived fuel recovery substate for the updated AI state.
    """
    reason = decision["behavior"]["reason"]
    command_type = decision["command"]["cmd_type"]
    if reason in ("forage_radar", "forage_sweep"):
        return "SENSE"
    if reason == "search_fuel_local":
        return "SEARCH"
    if command_type == "pickup_fuel":
        return "PICKUP"
    return "APPROACH"


__all__ = [
    "apply_mode_to_decision",
    "clear_ai_mode",
    "clear_mode_on_decision",
    "derive_hunt_mode_state",
    "derive_recover_equipment_mode_state",
    "derive_recover_fuel_mode_state",
    "set_ai_mode",
    "should_enter_hunt",
    "should_enter_recover_equipment",
    "should_enter_recover_fuel",
    "should_exit_hunt",
    "should_exit_recover_equipment",
    "should_exit_recover_fuel",
]
