"""Durable top-level AI mode helpers and migration rules."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    combat_reserve_restored,
)
from tankpit_bot.bot.ai.modes import AIMode, AIModeState, is_valid_ai_mode_state
from tankpit_bot.bot.ai.types import AIStateDict, make_behavior_score
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import BotCommand, make_hold_command
from tankpit_bot.state.rank_formulas import combat_radar_min, inventory_capacity


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


def resolve_owner_from_manual(ai_state: AIStateDict) -> AIMode | None:
    """Return the durable owner pinned by the SPA, or ``None`` for auto.

    The bot service surface writes :attr:`AIStateDict.manual_mode` when
    the phone user picks a mode. Auto-arbitration only runs when the
    field is ``None``; otherwise the pinned mode wins.

    Args:
        ai_state: Current AI state (already normalised by the caller).

    Returns:
        The pinned :data:`AIMode` when the SPA has selected one;
        ``None`` when auto-arbitration should run.
    """
    return ai_state["manual_mode"]


def _bump_dispatch_counter(state: AIStateDict, command: BotCommand) -> AIStateDict:
    """Return state with the live dispatch counter for ``command`` advanced.

    Args:
        state: AI state to update.
        command: Command being dispatched this tick.

    Returns:
        AI state whose ``live_radars_used`` or ``live_teleports`` is
        incremented when ``command`` targets that path. Other command
        types return ``state`` unchanged — those commands do not feed
        the SPA stats panel.
    """
    if command["cmd_type"] == "radar":
        return AIStateDict(**{**state, "live_radars_used": state["live_radars_used"] + 1})
    if command["cmd_type"] == "teleport":
        return AIStateDict(**{**state, "live_teleports": state["live_teleports"] + 1})
    return state


def apply_dispatch_counters(decision: TickDecisionDict) -> TickDecisionDict:
    """Advance live dispatch counters for the commands in ``decision``.

    Called by :func:`tankpit_bot.bot.ai_strategy.decide` on every
    non-hold path just before the arbitrator returns. Each command in
    the decision that maps to a tracked counter — radar or teleport —
    contributes one increment; other command types leave the counter
    untouched. Both the primary and (if present) secondary commands
    contribute, because the executor dispatches both under the tick's
    success gate.

    The tick loop persists ``decision["updated_ai_state"]`` only after
    :func:`tankpit_bot.bot.executor.execute` returns True, so a
    validation-side rejection never leaks a counter increment into
    the SPA stats panel — the wire and the panel stay aligned.

    Args:
        decision: Planner decision about to leave the arbitrator.

    Returns:
        Decision whose ``updated_ai_state`` reflects the dispatch
        counters the executor will produce for the primary + secondary
        commands.
    """
    state = _bump_dispatch_counter(decision["updated_ai_state"], decision["command"])
    secondary = decision["secondary_command"]
    if secondary is not None:
        state = _bump_dispatch_counter(state, secondary)
    return make_tick_decision(
        command=decision["command"],
        behavior=decision["behavior"],
        updated_ai_state=state,
        desired_equipment=decision["desired_equipment"],
        secondary_command=secondary,
    )


def make_hold_decision(ai_state: AIStateDict, timestamp_ms: int) -> TickDecisionDict:
    """Return a no-op decision for a manually-pinned idle tick.

    Produced when :func:`resolve_owner_from_manual` resolves to
    ``"UNSET"`` — the bot is connected, healthy, and should hold
    position instead of arbitrating hunt vs. collect. Downstream:

    * :func:`tankpit_bot.bot.executor.dispatch_command` sees
      ``cmd_type == "hold"`` and returns immediately without touching
      the wire.
    * The durable mode ownership is cleared to ``UNSET`` / empty
      substate; the durable ``mode_started_ms`` is refreshed whenever
      the previous owner was something other than ``UNSET`` so the
      bot-service status stream can report accurate idle duration.

    Args:
        ai_state: Current AI state.
        timestamp_ms: Current tick timestamp in milliseconds.

    Returns:
        Tick decision that dispatches nothing and stamps ``UNSET``
        ownership onto the returned AI state.
    """
    started_ms = timestamp_ms if ai_state["mode"] != "UNSET" else ai_state["mode_started_ms"]
    return make_tick_decision(
        command=make_hold_command(),
        behavior=make_behavior_score(
            mode="HUNT",
            score=0,
            target_x=0,
            target_y=0,
            reason_kind="manual_hold",
        ),
        updated_ai_state=AIStateDict(
            **{
                **ai_state,
                "mode": "UNSET",
                "mode_state": "",
                "mode_started_ms": started_ms,
            }
        ),
        desired_equipment=[],
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


def hunt_entry_permitted(ctx: DecideCtx) -> bool:
    """Return True when the bot's inventory permits entering HUNT.

    User contract (2026-07-06, Bug 0.4): the bot must never enter a
    combat engagement below full duals + full homings + at-least
    ``combat_radar_min`` extra radars. The 22:37 live run hit HUNT
    with duals 12/25 + homings 3/25, engaged orange-8 under-armed,
    exhausted its ammo mid-fight, hit the stationary-miss classifier
    (Bug 0.6), and blocked a live target. Enforce the readiness gate
    at every yield-to-hunt gesture: COLLECT never releases the tick
    unless the bot could take the fight to completion.

    The gate is inventory-only. Fuel readiness lives in
    :func:`combat_reserve_restored` and the fuel-cascade already
    guards it. The cardinal-shot override in
    :mod:`tankpit_bot.bot.ai_strategy` (Bug 0.5) intentionally
    bypasses this predicate for a free adjacent kill; even a single
    dual advances the kill and is worth taking under-armed.

    Args:
        ctx: Decision context.

    Returns:
        True when duals and homings are at ``inventory_capacity(rank)``
        and extra radars are at least ``combat_radar_min(rank)``.
    """
    rank = ctx.self_state["rank"]
    cap = inventory_capacity(rank)
    radar_floor = combat_radar_min(rank)
    return (
        ctx.inventory["dual_shots"]["count"] >= cap
        and ctx.inventory["homing_shots"]["count"] >= cap
        and ctx.inventory["extra_radars"]["count"] >= radar_floor
    )


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
    reason = decision["behavior"]["reason_kind"]
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
    reason = decision["behavior"]["reason_kind"]
    command_type = decision["command"]["cmd_type"]
    if reason in ("forage_radar", "forage_sweep", "scan_on_landing"):
        return "SENSE"
    if reason == "search_collect_local":
        return "SEARCH"
    if command_type in ("pickup_fuel", "pickup_equipment"):
        return "PICKUP"
    return "APPROACH"


__all__ = [
    "apply_dispatch_counters",
    "apply_mode_to_decision",
    "clear_ai_mode",
    "clear_mode_on_decision",
    "derive_collect_mode_state",
    "derive_hunt_mode_state",
    "hunt_entry_permitted",
    "make_hold_decision",
    "resolve_owner_from_manual",
    "set_ai_mode",
    "should_enter_collect",
    "should_enter_hunt",
    "should_exit_collect",
    "should_exit_hunt",
]
