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


def should_enter_collect(ctx: DecideCtx) -> bool:
    """Return True when the unified COLLECT mode should own planning.

    Entry triggers across fuel and equipment:

    * **Fuel low** -- at or below the fuel-low threshold.
    * **Insufficient engagement budget** -- no active combat target AND
      fuel cannot cover ``fuel_low_threshold + engagement_fuel_budget``.
      Refusing the engagement up front beats committing to a kill the
      bot will drop mid-fight (live run 2026-06-26 19:55: spawned at
      400 fuel, engaged adjacent enemy, dropped LOW_FUEL at 152, lost
      both the kill and the survival hop).
    * **Weapon emergency** -- any weapon reserve below its break
      threshold, or extra radars at or below the radar break threshold.
      Interrupts even an active combat target.
    * **Between kills** -- any weapon reserve below its resume
      threshold, or extra radars below the resume buffer, AND no active
      combat target. Finishes the current kill first, then restocks
      before the next hunt.

    Args:
        ctx: Decision context.

    Returns:
        True when fuel or equipment reserves require collection.
    """
    if ctx.fuel <= ctx.config["fuel_low_threshold"]:
        return True
    if ctx.ai_state["combat_target_id"] == -1:
        engagement_floor = ctx.config["fuel_low_threshold"] + ctx.config["engagement_fuel_budget"]
        if ctx.fuel < engagement_floor:
            return True
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


def should_exit_collect(ctx: DecideCtx) -> bool:
    """Return True when COLLECT can release control.

    The mode holds until BOTH reserves are healthy: fuel back above the
    full threshold AND weapons + radars back above their resume buffers.
    The break/resume gap gives hysteresis -- entry at the low break,
    exit only at the higher resume -- so the bot rebuilds to a full
    stock instead of leaving the moment it scrapes together one radar.

    Args:
        ctx: Decision context.

    Returns:
        True when fuel and combat reserves are restored.
    """
    if ctx.fuel < ctx.config["fuel_full_threshold"]:
        return False
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
        True when combat-ready and COLLECT has no stronger entry condition.
    """
    if should_enter_collect(ctx):
        return False
    return combat_reserve_restored(ctx)


def should_exit_hunt(ctx: DecideCtx) -> bool:
    """Return True when HUNT should release control.

    Args:
        ctx: Decision context.

    Returns:
        True when COLLECT now has a stronger entry condition.
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


def derive_collect_mode_state(decision: TickDecisionDict) -> AIModeState:
    """Derive the ``COLLECT`` substate from planner output.

    Args:
        decision: Decision produced by the collect-owner path.

    Returns:
        Derived collect substate for the updated AI state.
    """
    reason = decision["behavior"]["reason"]
    command_type = decision["command"]["cmd_type"]
    if reason in ("forage_radar", "forage_sweep", "scan_on_landing"):
        return "SENSE"
    if reason == "search_collect_local":
        return "SEARCH"
    if command_type in ("pickup_fuel", "pickup_equipment"):
        return "PICKUP"
    return "APPROACH"


__all__ = [
    "apply_mode_to_decision",
    "clear_ai_mode",
    "clear_mode_on_decision",
    "derive_collect_mode_state",
    "derive_hunt_mode_state",
    "set_ai_mode",
    "should_enter_collect",
    "should_enter_hunt",
    "should_exit_collect",
    "should_exit_hunt",
]
