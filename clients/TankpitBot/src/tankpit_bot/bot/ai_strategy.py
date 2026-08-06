"""Top-level durable AI owner selection.

This module owns only the orchestration layer that selects exactly one durable
mode owner per tick and rewrites the returned AI state with the matching
mode/substate fields.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.ferry import compose_decision_terrain
from tankpit_bot.bot.ai.greeting import attach_human_greeting
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.mode_controller import (
    apply_dispatch_counters,
    apply_mode_to_decision,
    clear_ai_mode,
    derive_collect_mode_state,
    derive_hunt_mode_state,
    make_hold_decision,
    resolve_owner_from_manual,
    should_enter_hunt,
    should_exit_collect,
    should_exit_hunt,
)
from tankpit_bot.bot.ai.modes import AIMode, AIModeState, is_valid_ai_mode_state
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.inventory import InventoryState
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.types import SelfStateDict, WorldStateDict


def decide(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    inventory: InventoryState,
    timestamp_ms: int,
    terrain: TerrainMapProtocol | None,
    combat_feedback: CombatFeedback = "",
    map_fuel_dots: tuple[tuple[int, int], ...] = (),
) -> TickDecisionDict:
    """Run one AI decision cycle under durable owner routing.

    Args:
        world: Current world state with tanks, containers, and mines.
        self_state: Player state for this tick.
        ai_state: Current durable AI state.
        inventory: Current inventory state.
        timestamp_ms: Current game timestamp in milliseconds.
        terrain: Optional terrain map for reachability and landing checks.
        combat_feedback: Protocol-level hit or miss feedback for the last shot.
        map_fuel_dots: 0x4C fuel-dot atlas positions (empty before the
            first map open of the session).

    Returns:
        Tick decision produced by the selected durable owner.
    """
    normalized_state = _normalize_ai_state(ai_state)
    manual = resolve_owner_from_manual(normalized_state)
    if manual == "UNSET":
        emit_ai("manual mode UNSET: holding position")
        return make_hold_decision(
            normalized_state,
            timestamp_ms,
            self_state["fuel"],
            inventory,
        )
    ctx = DecideCtx(
        world,
        self_state,
        normalized_state,
        inventory,
        timestamp_ms,
        compose_decision_terrain(world, terrain, timestamp_ms),
        combat_feedback,
        map_fuel_dots,
    )
    mode = _resolve_owner_mode(ctx, manual)
    if mode == "COLLECT":
        collect_decision = decide_collect_mode(ctx)
        if collect_decision is not None:
            owned = apply_mode_to_decision(
                collect_decision,
                "COLLECT",
                derive_collect_mode_state(collect_decision),
                timestamp_ms,
            )
            return apply_dispatch_counters(owned)
        if ctx.ai_state["wind_down"]:
            # Winding down and nothing left to collect: ending early
            # and clean beats idling out the clock or re-engaging.
            raise SessionExitError(
                "session_complete",
                f"wound down; collect exhausted at fuel={ctx.fuel}",
            )
        # Collection exhausted with healthy fuel: the tank is stocked,
        # so this tick belongs to the hunt owner (fall through).
        emit_ai("collect owner yielded, handing tick to hunt owner")
    decision = decide_hunt_mode(ctx)
    owned = apply_mode_to_decision(
        decision,
        "HUNT",
        derive_hunt_mode_state(decision),
        timestamp_ms,
    )
    return apply_dispatch_counters(attach_human_greeting(ctx, owned))


def _resolve_owner_mode(ctx: DecideCtx, manual: AIMode | None) -> AIMode:
    """Select the durable owner for this tick, honouring the manual pin.

    Args:
        ctx: Decision context.
        manual: SPA-pinned mode override, or ``None`` when the
            arbitrator should run auto-selection. ``UNSET`` is handled
            upstream in :func:`decide` and never reaches this function.

    Returns:
        The mode owner for this tick. Manual ``HUNT`` and ``COLLECT``
        skip auto-arbitration; ``None`` delegates to
        :func:`_select_owner_mode`.
    """
    if manual == "HUNT":
        return "HUNT"
    if manual == "COLLECT":
        return "COLLECT"
    return _select_owner_mode(ctx)


def _normalize_ai_state(ai_state: AIStateDict) -> AIStateDict:
    """Return AI state with invalid durable ownership cleared.

    Args:
        ai_state: Current AI state.

    Returns:
        Original AI state when the durable mode pair is valid, otherwise the
        same state with durable ownership reset to ``UNSET``.
    """
    if is_valid_ai_mode_state(ai_state["mode"], ai_state["mode_state"]):
        if ai_state["mode"] == "UNSET" and ai_state["combat_target_id"] != -1:
            return _migrate_unset_combat_state(ai_state)
        return ai_state
    if ai_state["combat_target_id"] != -1:
        return _migrate_unset_combat_state(ai_state)
    return clear_ai_mode(ai_state)


def _select_owner_mode(ctx: DecideCtx) -> AIMode:
    """Select the durable owner for this tick.

    HUNT is a privilege of a full tank (user contract 2026-07-25:
    "it should never hunt when its low on fuel or equipment. it
    should never hunt if it is not full on everything except -5 max
    radar"). Entry into HUNT requires full readiness
    (:func:`should_enter_hunt`); a held HUNT releases only when a
    COLLECT trigger fires (:func:`should_exit_hunt`), so fighting
    down from cap does not thrash ownership. Everything that is not
    ready-to-hunt collects.

    The 2026-07-13 cardinal-adjacent override ("an enemy one tile
    away flips the tick to HUNT regardless of reserves") is deleted:
    it silently outranked the low-fuel break and produced the
    2026-07-25 practice-room fight-to-death. Ignoring an adjacent
    bot while collecting is safe -- bots never initiate, they only
    return fire ([[enemy-bot-behavior]]).

    During session wind-down (``ai_state["wind_down"]``, set by the
    tick loop in the final stretch of a bounded run) HUNT is closed
    entirely — held or not — and the session ends with
    ``session_complete`` the moment the tank is fully stocked (user
    request 2026-07-26: "run and then collect and exit cleanly,
    instead of the program killing it mid action"). Ending stocked is
    also what makes the NEXT session open combat-ready (run
    bot-20260726-002554 scored its first kill at t+30 s on the prior
    session's leftover stock).

    Args:
        ctx: Decision context.

    Returns:
        Durable top-level owner for the current tick.

    Raises:
        SessionExitError: ``session_complete`` when winding down and
            fully stocked — the clean exit.
    """
    if ctx.ai_state["wind_down"]:
        target = ctx.world["tanks"].get(str(ctx.ai_state["combat_target_id"]))
        finishing_kill = (
            ctx.mode == "HUNT"
            and target is not None
            and target["liveness"] == "alive"
            and not should_exit_hunt(ctx)
        )
        if finishing_kill:
            # Never abandon a fight in progress (user rulings 2026-07-25
            # and 2026-07-26): the current kill completes; the break
            # thresholds still protect, and no NEW target is acquired.
            return "HUNT"
        if should_exit_collect(ctx):
            raise SessionExitError(
                "session_complete",
                f"wound down fully stocked at fuel={ctx.fuel}",
            )
        return "COLLECT"
    current_mode = ctx.mode
    if current_mode == "COLLECT" and not should_exit_collect(ctx):
        return "COLLECT"
    if current_mode == "HUNT" and not should_exit_hunt(ctx):
        return "HUNT"
    if should_enter_hunt(ctx):
        return "HUNT"
    return "COLLECT"


def _migrate_unset_combat_state(ai_state: AIStateDict) -> AIStateDict:
    """Return unset AI state migrated into durable HUNT ownership.

    Args:
        ai_state: Current AI state with a locked combat target.

    Returns:
        AI state migrated into durable HUNT ownership.
    """
    migrated_mode_state: AIModeState = "REFRESH"
    if ai_state["last_shot_target_id"] == ai_state["combat_target_id"]:
        migrated_mode_state = "ENGAGE"
    return AIStateDict(
        **{
            **ai_state,
            "mode": "HUNT",
            "mode_state": migrated_mode_state,
            "mode_started_ms": 0,
        }
    )


__all__ = ["decide"]
