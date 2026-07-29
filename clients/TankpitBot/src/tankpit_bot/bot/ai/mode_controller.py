"""Durable top-level AI mode helpers and migration rules."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.modes import AIMode, AIModeState, is_valid_ai_mode_state
from tankpit_bot.bot.ai.tactics import compute_desired_equipment
from tankpit_bot.bot.ai.types import AIStateDict, make_behavior_score
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import BotCommand, make_hold_command
from tankpit_bot.inventory import InventoryState
from tankpit_bot.physics.capacity import fuel_capacity, inventory_capacity


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


def make_hold_decision(
    ai_state: AIStateDict,
    timestamp_ms: int,
    fuel: int,
    inventory: InventoryState,
) -> TickDecisionDict:
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

    Equipment: the hold keeps the NORMAL stocked loadout (dual +
    homing while stocked, radar always) instead of the empty set the
    first idle-pin implementation requested — that set actively
    DISARMED the tank, printing "Using dual shot disabled" in the
    game log and, because toggle state persists across logout
    ([[radar-mechanics]]), leaving the tank visibly disarmed for the
    next login (user report 2026-07-29: "the hot has homing and dual
    shots disabled for some reaosn?").

    Args:
        ai_state: Current AI state.
        timestamp_ms: Current tick timestamp in milliseconds.
        fuel: Current fuel level (equipment-policy input).
        inventory: Current inventory state (stock counts gate the
            dual/homing toggles).

    Returns:
        Tick decision that dispatches nothing, keeps the tank armed,
        and stamps ``UNSET`` ownership onto the returned AI state.
    """
    started_ms = timestamp_ms if ai_state["mode"] != "UNSET" else ai_state["mode_started_ms"]
    desired = compute_desired_equipment(
        "UNSET",
        fuel,
        dual_shots_count=inventory["dual_shots"]["count"],
        homing_shots_count=inventory["homing_shots"]["count"],
    )
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
        desired_equipment=sorted(desired),
    )


def should_enter_collect(ctx: DecideCtx) -> bool:
    """Return True when the unified COLLECT mode should own planning.

    Entry triggers across fuel and equipment:

    * **Fuel low** -- at or below the fuel-low threshold. Interrupts
      even an active combat target (user contract 2026-07-25: the
      2026-07-13 cardinal override let a fight outrank this break and
      the bot died trading at 84 fuel in the practice-room gang-up).
    * **Weapon emergency** -- any weapon reserve below its break
      threshold, or extra radars below the radar break threshold.
      Interrupts even an active combat target.
    * **Between kills** -- no active combat target AND anything short
      of a genuinely full tank: fuel below the rank capacity or
      inventory below the rank caps (user contract 2026-07-25: "never
      hunt if it is not full on everything except -5 max radar"; caps
      are rank-derived, replacing the fixed resume thresholds that
      under-restocked high ranks). Finishes the current kill first,
      then restocks fully before the next hunt.

    Args:
        ctx: Decision context.

    Returns:
        True when fuel or equipment reserves require collection.
    """
    if ctx.fuel <= ctx.config["fuel_low_threshold"]:
        return True
    if (
        ctx.inventory["dual_shots"]["count"] < ctx.config["dual_break_threshold"]
        or ctx.inventory["homing_shots"]["count"] < ctx.config["dual_break_threshold"]
        or ctx.inventory["extra_radars"]["count"] < ctx.config["radar_break_threshold"]
    ):
        return True
    if ctx.ai_state["combat_target_id"] != -1:
        return False
    return ctx.fuel < hunt_fuel_floor(ctx) or not hunt_entry_permitted(ctx)


def hunt_fuel_floor(ctx: DecideCtx) -> int:
    """Return the fuel level that counts as a full tank for HUNT entry.

    The rank's actual fuel capacity (user ruling 2026-07-25: "just
    determine max fuel based on the tank rank") -- 1000 at recruit
    through 1800 at general. The collect cascade's pickup ceiling is
    the same physics number, so "stop collecting fuel" and "may hunt"
    can never disagree and deadlock the owner selection.

    Args:
        ctx: Decision context.

    Returns:
        ``fuel_capacity(rank)`` for the bot's current rank.
    """
    return fuel_capacity(ctx.self_state["rank"])


def should_exit_collect(ctx: DecideCtx) -> bool:
    """Return True when COLLECT can release control.

    The mode holds until the bot is FULLY restocked: fuel at the
    rank-clamped full floor AND the inventory combat-ready
    (:func:`hunt_entry_permitted` -- duals and homings at cap, extra
    radars within 5 of cap; user contract 2026-07-25: "never hunt if
    it is not full on everything except -5 max radar"). The
    entry-at-break / exit-at-full gap gives hysteresis, so the bot
    rebuilds a full stock instead of leaving the moment it scrapes
    together one radar.

    Args:
        ctx: Decision context.

    Returns:
        True when fuel and inventory are fully restored.
    """
    if ctx.fuel < hunt_fuel_floor(ctx):
        return False
    return hunt_entry_permitted(ctx)


def should_enter_hunt(ctx: DecideCtx) -> bool:
    """Return True when HUNT is the valid top-level owner.

    HUNT is a privilege of a full tank (user contract 2026-07-25):
    fuel at the rank-clamped full floor, duals and homings at cap,
    extra radars within 5 of cap, and no COLLECT trigger pending.
    Starting a fight below full stock leads to abandoned kills when
    the break threshold pulls the bot away mid-fight.

    Args:
        ctx: Decision context.

    Returns:
        True when fully stocked and COLLECT has no entry condition.
    """
    if should_enter_collect(ctx):
        return False
    return ctx.fuel >= hunt_fuel_floor(ctx) and hunt_entry_permitted(ctx)


def combat_radar_min(rank: int) -> int:
    """Return the minimum extra-radar count for HUNT-entry readiness.

    This is bot POLICY, not game physics — it lives here with its only
    consumer, not in :mod:`tankpit_bot.physics`. User contract
    (2026-07-06): weapons must be at cap for HUNT entry, but extra
    radars are permitted up to 5 below cap because scan coverage
    during the fight consumes them faster than the between-kill
    restock can top them up. The floor is
    ``inventory_capacity(rank) - 5``.

    Args:
        rank: Wire rank field, ``0`` (recruit) through ``8`` (general).

    Returns:
        Minimum extra-radar count below which HUNT entry is refused:
        15 at recruit, 20 at private, 25 at corporal, ..., 55 at
        general.
    """
    return inventory_capacity(rank) - 5


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

    The gate is inventory-only; fuel readiness is enforced alongside
    it in :func:`should_enter_hunt` and :func:`should_exit_collect`.
    Nothing bypasses this predicate: the 2026-07-13 cardinal-shot
    override that did was deleted 2026-07-25 (user contract: the bot
    never hunts below full stock, no exceptions).

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

    A held HUNT releases only when a COLLECT trigger fires -- fuel at
    the low break, a weapon or radar break, or between-kills resume
    shortfalls. Deliberately NOT ``not should_enter_hunt``: entry
    requires a full stock, and the first shot of a fight spends a
    dual, so re-checking the entry bar every tick would thrash
    ownership one shot into every engagement.

    Args:
        ctx: Decision context.

    Returns:
        True when COLLECT now has an entry condition.
    """
    return should_enter_collect(ctx)


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
    "combat_radar_min",
    "derive_collect_mode_state",
    "derive_hunt_mode_state",
    "hunt_entry_permitted",
    "hunt_fuel_floor",
    "make_hold_decision",
    "resolve_owner_from_manual",
    "set_ai_mode",
    "should_enter_collect",
    "should_enter_hunt",
    "should_exit_collect",
    "should_exit_hunt",
]
