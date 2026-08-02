"""Held resource-lock continuation for the COLLECT cascade.

Cascade step 1: continue or release the equipment/fuel lock a prior
tick latched. Release is deliberate and enumerated ([[committed-intent]]):
superior executable candidate, tank at capacity, or the structural
server-confirmed move-failed mark -- transient inexecutability HOLDS
the plan.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    locked_resource_target,
    make_decision,
    set_resource_target,
)
from tankpit_bot.bot.ai.equipment import (
    is_fuel_lock_release_warranted,
    is_lock_release_warranted,
)
from tankpit_bot.bot.ai.equipment_search import find_best_fuel, find_nearest_equipment
from tankpit_bot.bot.ai.intent import release_collect_plan
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.sniffer.world_state import is_move_target_failed
from tankpit_bot.state.types import ContainerStateDict


def continue_or_release_lock(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    """Resolve any held resource lock for this tick.

    Looks up the lock kind from ``ai_state.resource_target_kind`` and
    routes to the matching continue/release path. A cleared lock leaves
    ``base_state`` with ``resource_target_kind == ""`` so downstream
    cascade steps see a clean slate.
    """
    _, equipment_lock = locked_resource_target(ctx, "equipment")
    if equipment_lock is not None:
        return _continue_or_release_equipment_lock(ctx, base_state, equipment_lock)
    _, fuel_lock = locked_resource_target(ctx, "fuel")
    if fuel_lock is not None:
        return continue_or_release_fuel_lock(ctx, base_state, fuel_lock)
    return None, base_state


def _continue_or_release_equipment_lock(
    ctx: DecideCtx,
    base_state: AIStateDict,
    locked_target: ContainerStateDict,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    if _superior_equipment_candidate(ctx, locked_target) is not None:
        emit_ai(
            "releasing equipment lock at (%d,%d): markedly closer equipment is visible",
            locked_target["x"],
            locked_target["y"],
        )
        return None, release_collect_plan(base_state, reason="superior_candidate")
    target_x = locked_target["x"]
    target_y = locked_target["y"]
    locked_command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="equipment")
    if locked_command is None:
        # Transient inexecutability HOLDS the plan ([[committed-intent]];
        # run bot-20260730-032x ticks 361/366/371: three not_executable
        # releases fired mid-approach with the plan's own map_open in
        # flight, and each target was re-locked and served 2-3 ticks
        # later — the plan was never invalid, the executor was busy).
        # Only the server-confirmed move-failed mark is structural.
        if is_move_target_failed(target_x, target_y, ctx.timestamp_ms):
            emit_ai(
                "locked equipment target at (%d,%d) marked move-failed - releasing",
                target_x,
                target_y,
            )
            return None, release_collect_plan(base_state, reason="not_executable")
        emit_ai(
            "locked equipment target at (%d,%d) not executable this tick - holding plan",
            target_x,
            target_y,
        )
        return None, base_state
    emit_ai("continue locked equipment target at (%d,%d)", target_x, target_y)
    decision = make_decision(
        locked_command,
        "COLLECT",
        COLLECT_SCORE,
        target_x,
        target_y,
        "equipment_locked",
        set_resource_target(base_state, "equipment", target_x, target_y),
        ctx.equip,
    )
    return decision, base_state


def continue_or_release_fuel_lock(
    ctx: DecideCtx,
    base_state: AIStateDict,
    locked_target: ContainerStateDict,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    """Continue or release a held fuel lock (see module docstring)."""
    if ctx.fuel >= fuel_capacity(ctx.self_state["rank"]):
        # Live run 2026-07-06 pickup loop: a held fuel lock kept
        # re-dispatching pickup_fuel at capacity because the lock path
        # had no capacity gate, only the selection path did. Every
        # dispatch drew wire 0x52 code-5 "Tank full" and the lock
        # survived to next tick. Capacity is now rank-derived
        # (:func:`tankpit_bot.physics.capacity.fuel_capacity`), so
        # this gate closes the loop at the root regardless of how the
        # lock was established.
        emit_ai(
            "releasing fuel lock at (%d,%d): tank at capacity %d",
            locked_target["x"],
            locked_target["y"],
            ctx.fuel,
        )
        return None, release_collect_plan(base_state, reason="tank_at_capacity")
    if _superior_fuel_candidate(ctx, locked_target) is not None:
        emit_ai(
            "releasing fuel lock at (%d,%d): markedly closer fuel is visible",
            locked_target["x"],
            locked_target["y"],
        )
        return None, release_collect_plan(base_state, reason="superior_candidate")
    target_x = locked_target["x"]
    target_y = locked_target["y"]
    locked_command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="fuel")
    if locked_command is None:
        # Same transient-vs-structural law as the equipment lock.
        if is_move_target_failed(target_x, target_y, ctx.timestamp_ms):
            emit_ai(
                "locked fuel target at (%d,%d) marked move-failed - releasing",
                target_x,
                target_y,
            )
            return None, release_collect_plan(base_state, reason="not_executable")
        emit_ai(
            "locked fuel target at (%d,%d) not executable this tick - holding plan",
            target_x,
            target_y,
        )
        return None, base_state
    emit_ai(
        "continue locked fuel target at (%d,%d) vol=%d (fuel=%d)",
        target_x,
        target_y,
        locked_target["volume"],
        ctx.fuel,
    )
    decision = make_decision(
        locked_command,
        "COLLECT",
        COLLECT_SCORE,
        target_x,
        target_y,
        "fuel_locked",
        set_resource_target(base_state, "fuel", target_x, target_y),
        ctx.equip,
        reason_context={"volume": locked_target["volume"]},
    )
    return decision, base_state


def _superior_equipment_candidate(
    ctx: DecideCtx,
    locked_target: ContainerStateDict,
) -> ContainerStateDict | None:
    candidate = find_nearest_equipment(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
    )
    if candidate is None:
        return None
    if (candidate["x"], candidate["y"]) == (locked_target["x"], locked_target["y"]):
        return None
    if not is_lock_release_warranted(
        ctx.self_state,
        locked_target["x"],
        locked_target["y"],
        candidate["x"],
        candidate["y"],
    ):
        return None
    # Closer is only superior when it is EXECUTABLE NOW, by the same
    # predicate execution uses. Session-12 ferry livelock (2026-07-30,
    # ~70 laps, user broke it by closing the browser): landing at the
    # ferry boarding tile for the locked (100,8), the steal test's
    # reachability said the closer (106,14) was walkable, its lock ran
    # ONE disembark leg, went "not executable - holding plan", and the
    # cascade hopped back to (100,8)'s boarding tile -- A steals B,
    # B stalls, B hops back toward A, forever, every action
    # succeeding so no disproof could fire. A candidate that cannot
    # produce a command this tick never steals a viable plan.
    if walk_or_teleport(ctx, candidate["x"], candidate["y"], pickup_kind="equipment") is None:
        return None
    return candidate


def _superior_fuel_candidate(
    ctx: DecideCtx,
    locked_target: ContainerStateDict,
) -> ContainerStateDict | None:
    candidate = find_best_fuel(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        minimum_volume=1,
    )
    if candidate is None:
        return None
    if (candidate["x"], candidate["y"]) == (locked_target["x"], locked_target["y"]):
        return None
    deficit = fuel_capacity(ctx.self_state["rank"]) - ctx.fuel
    if not is_fuel_lock_release_warranted(ctx.self_state, locked_target, candidate, deficit):
        return None
    # Same executability bar as the equipment steal (session-12 ferry
    # livelock): a candidate that cannot produce a command this tick
    # never steals a viable plan.
    if walk_or_teleport(ctx, candidate["x"], candidate["y"], pickup_kind="fuel") is None:
        return None
    return candidate


__all__ = [
    "continue_or_release_fuel_lock",
    "continue_or_release_lock",
]
