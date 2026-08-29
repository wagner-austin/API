"""Durable mode ownership: set, clear, and apply a mode to a decision.

Holds the mode accessors, the dispatch counters, the hold decision, and
the substate derivations. The entry/exit predicates it consults are
:mod:`tankpit_bot.bot.ai.mode_gates`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.block_harvest import anchored_window_origin
from tankpit_bot.bot.ai.scoring_types import make_behavior_score
from tankpit_bot.bot.ai.tactics import compute_desired_equipment
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import BotCommand, make_hold_command
from tankpit_bot.inventory import InventoryState
from tankpit_bot.types.modes import AIMode, AIModeState, is_valid_ai_mode_state


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


def latch_scope_shift_landing(
    decision: TickDecisionDict,
    window_left: int,
    window_top: int,
    sx: int,
    sy: int,
) -> TickDecisionDict:
    """Latch the landing-scan gate for the window a scope shift opens.

    The ``last_landing_scan_viewport`` latch predates scope shifts as
    a window-move trigger ("the viewport origin changes only on
    teleport"), so without this step every deliberate pan — a quad
    sweep steer, a harvest framing shift, the free ferry scout — read
    as a fresh LANDING next tick and drew the unconditional landing
    radar, taxing pans that are free by ruling (larder/scout,
    2026-07-27) and mislabeling the sweep's own quadrant scans. A pan
    is a deliberate look, not a landing: the pan-er decides whether a
    radar follows (the sweep's ``quad_sweep_radar`` does; the scout
    stays free). The latch is set to the origin the anchor law says
    the server's 0x5A will state ([[viewport-shift-protocol]]).

    Args:
        decision: Planner decision about to leave the arbitrator.
        window_left: Current stored window origin X.
        window_top: Current stored window origin Y.
        sx: Self X.
        sy: Self Y.

    Returns:
        The decision, with the landing latch pre-set when its command
        is a scope shift; unchanged otherwise.
    """
    if decision["command"]["cmd_type"] != "scope_shift":
        return decision
    left, top = anchored_window_origin(
        window_left,
        window_top,
        sx,
        sy,
        decision["command"]["direction"],
    )
    state = AIStateDict(
        **{**decision["updated_ai_state"], "last_landing_scan_viewport": f"{left},{top}"}
    )
    return make_tick_decision(
        command=decision["command"],
        behavior=decision["behavior"],
        updated_ai_state=state,
        desired_equipment=decision["desired_equipment"],
        secondary_command=decision["secondary_command"],
    )


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
    if reason in (
        "forage_radar",
        "forage_sweep",
        "scan_on_landing",
        "desync_rescan",
        "quad_sweep_shift",
        "quad_sweep_radar",
    ):
        return "SENSE"
    if reason in (
        "search_collect_local",
        "ferry_scope_scout",
        "gatherer_hold",
        "forage_frontier_walk",
    ):
        # The free viewport pan is a SEARCH beat: the tick looks at
        # water it cannot yet believe in, exactly like a local search
        # looks at ground ([[viewport-shift-protocol]] scope scout).
        # The gatherer's exhausted hold is the same beat between
        # searches -- waiting one window for the world to change.
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
    "latch_scope_shift_landing",
    "make_hold_decision",
    "resolve_owner_from_manual",
    "set_ai_mode",
]
