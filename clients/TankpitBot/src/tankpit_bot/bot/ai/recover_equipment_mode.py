"""Durable equipment-recovery owner and shared equipment recovery helpers."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_use_radar,
    clear_resource_target,
    locked_resource_target,
    make_decision,
    needs_emergency_equipment,
    set_resource_target,
    should_scan_resources_in_current_viewport,
)
from tankpit_bot.bot.ai.equipment import (
    describe_container_search,
    find_adjacent_container,
    find_equipment_candidates,
    find_known_equipment_candidates,
    find_nearest_equipment,
    is_lock_release_warranted,
)
from tankpit_bot.bot.ai.forage import plan_forage_search
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.resource_search import (
    is_recently_attempted,
    make_recovery_edge_decision,
    make_recovery_map_intel_decision,
    make_resource_search_hop,
    record_attempt_mark,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    BotCommand,
    make_pickup_fuel_command,
    make_radar_command,
)
from tankpit_bot.diagnostics.game_log_feedback import is_fuel_at_learned_capacity
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.state.types import ContainerStateDict

# An equipment approach teleport is considered stale after this TTL;
# a container that has not become collectable within this window is
# skipped on the next pass.
_EQUIPMENT_APPROACH_TTL_MS = 30000


def _is_equipment_target_attempted(ctx: DecideCtx, x: int, y: int) -> bool:
    """Return True when an equipment target carries a live approach mark.

    Args:
        ctx: Decision context.
        x: Target X coordinate.
        y: Target Y coordinate.

    Returns:
        True if the target was teleport-approached within the TTL.
    """
    return is_recently_attempted(
        ctx.ai_state["attempted_equipment_targets"],
        x,
        y,
        ctx.timestamp_ms,
        ttl_ms=_EQUIPMENT_APPROACH_TTL_MS,
    )


def _with_equipment_approach_recorded(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    command: BotCommand,
    target_x: int,
    target_y: int,
) -> AIStateDict:
    """Record a teleport approach at an equipment target.

    Walks and pickups are left unmarked: they either complete (the
    container disappears) or get rejected by the server (which feeds
    the ``failed_pickups`` path). Only teleports can orbit silently.

    Args:
        ctx: Decision context.
        ai_state: State being returned for this decision.
        command: Command produced for the target.
        target_x: Target X coordinate.
        target_y: Target Y coordinate.

    Returns:
        AI state with the approach mark recorded for teleports, or the
        input state unchanged for every other command kind.
    """
    if command["cmd_type"] != "teleport":
        return ai_state
    emit_diagnostic(
        diagnostic_kind="equipment_approach",
        target_x=target_x,
        target_y=target_y,
        self_x=ctx.self_state["x"],
        self_y=ctx.self_state["y"],
        fuel=ctx.fuel,
    )
    return AIStateDict(
        **{
            **ai_state,
            "attempted_equipment_targets": record_attempt_mark(
                ctx.ai_state["attempted_equipment_targets"],
                target_x,
                target_y,
                ctx.timestamp_ms,
                ttl_ms=_EQUIPMENT_APPROACH_TTL_MS,
            ),
        }
    )


def select_equipment_target(
    ctx: DecideCtx,
    *,
    allow_unreachable: bool,
) -> tuple[ContainerStateDict, BotCommand] | None:
    """Return the nearest executable equipment target.

    Targets with a live teleport-approach mark are skipped: a marked
    container already ate a teleport without becoming collectable.

    Args:
        ctx: Decision context.
        allow_unreachable: Whether terrain-blocked targets may use teleport fallback.

    Returns:
        ``(container, command)`` for the nearest executable equipment target, or
        ``None`` when no visible equipment target can currently be executed.
    """
    candidates = [
        container
        for container in find_equipment_candidates(
            ctx.filtered,
            ctx.self_state,
            ctx.terrain,
            allow_unreachable=allow_unreachable,
            now_ms=ctx.timestamp_ms,
        )
        if not _is_equipment_target_attempted(ctx, container["x"], container["y"])
    ]
    if not candidates:
        return None

    container = candidates[0]
    command = walk_or_teleport(ctx, container["x"], container["y"], pickup_kind="equipment")
    if command is None:
        return None
    return (container, command)


def try_search_critical_equipment(ctx: DecideCtx) -> TickDecisionDict | None:
    """Search locally for equipment when emergency reserves are depleted.

    Args:
        ctx: Decision context.

    Returns:
        Equipment search decision, or ``None`` when emergency search should not
        run this tick.
    """
    if not needs_emergency_equipment(ctx):
        return None
    if ctx.fuel < ctx.config["fuel_low_threshold"]:
        emit_ai("skipping equipment search - fuel too low (%d)", ctx.fuel)
        return None
    base_state = (
        clear_resource_target(ctx.base)
        if ctx.base["resource_target_kind"] == "equipment"
        else ctx.base
    )
    search_decision = _plan_equipment_sense_or_search(ctx, 925, base_state)
    if search_decision is not None:
        return search_decision
    return _equipment_recovery_fallback(ctx, base_state)


def decide_recover_equipment_mode(ctx: DecideCtx) -> TickDecisionDict:
    """Run the durable ``RECOVER_EQUIPMENT`` owner for this tick.

    The owner persists until the combat reserve exit threshold is restored,
    even after the emergency break threshold has already been crossed back
    upward. That prevents the mode from dropping out mid-restock.

    Args:
        ctx: Decision context.

    Returns:
        Mode-owned recovery decision.
    """
    target_reason = "equipment_critical" if needs_emergency_equipment(ctx) else "equipment_restock"
    target_decision = _plan_equipment_target(
        ctx,
        score=925,
        target_reason=target_reason,
        locked_reason="equipment_locked",
    )
    if target_decision is not None:
        return target_decision

    search_base = (
        clear_resource_target(ctx.base)
        if ctx.base["resource_target_kind"] == "equipment"
        else ctx.base
    )
    search_decision = _plan_equipment_sense_or_search(ctx, 925, search_base)
    if search_decision is not None:
        return search_decision
    return _equipment_recovery_fallback(
        ctx,
        (
            clear_resource_target(ctx.base)
            if ctx.base["resource_target_kind"] == "equipment"
            else ctx.base
        ),
    )


def _equipment_recovery_fallback(
    ctx: DecideCtx,
    ai_state: AIStateDict,
) -> TickDecisionDict:
    """Return the always-executable fallback when search cannot act.

    Raising here killed the bot process mid-game (live run
    20260610-000x: radar illegal, hop unaffordable at fuel=528 vs cost
    540). "No affordable search action" is a legitimate game state, so
    the owner falls back to a cheap edge walk and, when fully boxed in,
    to free map intel.

    Args:
        ctx: Decision context.
        ai_state: Base AI state to rewrite.

    Returns:
        Edge-walk or map-intel decision; never ``None``.
    """
    edge = make_recovery_edge_decision(
        ctx,
        mode="COLLECT_EQUIPMENT",
        score=925,
        reason="edge_for_equipment",
        ai_state=ai_state,
    )
    if edge is not None:
        return edge
    return make_recovery_map_intel_decision(
        ctx,
        mode="COLLECT_EQUIPMENT",
        score=925,
        reason="map_intel_for_equipment",
        ai_state=ai_state,
    )


def _plan_equipment_target(
    ctx: DecideCtx,
    *,
    score: int,
    target_reason: str,
    locked_reason: str,
) -> TickDecisionDict | None:
    """Plan an actionable equipment pickup or approach decision.

    Args:
        ctx: Decision context.
        score: Behavior score for actionable targets.
        target_reason: Behavior reason for a newly selected target.
        locked_reason: Behavior reason for a continued locked target.
    Returns:
        Recovery decision, or ``None`` when no actionable target exists.
    """
    base_state, locked_target = locked_resource_target(ctx, "equipment")
    # A pickup at capacity is a wasted action ("Tank full" in the game
    # log), so opportunistic fuel only triggers below the learned cap.
    adjacent_fuel = (
        None
        if is_fuel_at_learned_capacity(ctx.fuel)
        else find_adjacent_container(
            ctx.filtered,
            ctx.self_state,
            ctx.terrain,
            want_fuel=True,
            now_ms=ctx.timestamp_ms,
        )
    )
    if adjacent_fuel is not None:
        emit_ai(
            "opportunistic fuel pickup at (%d,%d) during equipment recovery",
            adjacent_fuel["x"],
            adjacent_fuel["y"],
        )
        return make_decision(
            make_pickup_fuel_command(adjacent_fuel["x"], adjacent_fuel["y"]),
            "COLLECT_EQUIPMENT",
            score,
            adjacent_fuel["x"],
            adjacent_fuel["y"],
            "opportunistic_fuel",
            base_state,
            ctx.equip,
        )
    if locked_target is not None and _superior_equipment_candidate(ctx, locked_target) is not None:
        emit_ai(
            "releasing equipment lock at (%d,%d): markedly closer equipment is visible",
            locked_target["x"],
            locked_target["y"],
        )
        base_state = clear_resource_target(base_state)
        locked_target = None
    if locked_target is not None:
        target_x = locked_target["x"]
        target_y = locked_target["y"]
        locked_command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="equipment")
        if locked_command is not None:
            emit_ai("continue locked equipment target at (%d,%d)", target_x, target_y)
            updated_state = _with_equipment_approach_recorded(
                ctx,
                AIStateDict(
                    **{
                        **set_resource_target(base_state, "equipment", target_x, target_y),
                        "equipment_search_failures": 0,
                    }
                ),
                locked_command,
                target_x,
                target_y,
            )
            return make_decision(
                locked_command,
                "COLLECT_EQUIPMENT",
                score,
                target_x,
                target_y,
                locked_reason,
                updated_state,
                ctx.equip,
            )
        emit_ai("locked equipment target at (%d,%d) no longer executable", target_x, target_y)
        base_state = clear_resource_target(base_state)

    selection = select_equipment_target(ctx, allow_unreachable=True)
    if selection is None:
        emit_ai(
            "no executable RECOVER_EQUIPMENT target (%s)",
            describe_container_search(
                ctx.filtered,
                ctx.self_state,
                ctx.terrain,
                want_fuel=False,
                allow_unreachable=True,
            ),
        )
        return _plan_known_equipment_target(ctx, base_state, score=score)

    container, command = selection
    target_x = container["x"]
    target_y = container["y"]
    emit_ai("collect equipment at (%d,%d)", target_x, target_y)
    updated_state = _with_equipment_approach_recorded(
        ctx,
        AIStateDict(
            **{
                **set_resource_target(base_state, "equipment", target_x, target_y),
                "equipment_search_failures": 0,
            }
        ),
        command,
        target_x,
        target_y,
    )
    return make_decision(
        command,
        "COLLECT_EQUIPMENT",
        score,
        target_x,
        target_y,
        target_reason,
        updated_state,
        ctx.equip,
    )


def _superior_equipment_candidate(
    ctx: DecideCtx,
    locked_target: ContainerStateDict,
) -> ContainerStateDict | None:
    """Return a fresh equipment candidate that warrants dropping the lock.

    Mirrors the fuel-mode rule: a candidate releases the lock only when
    it clears the half-distance + minimum-gap rule in
    :func:`tankpit_bot.bot.ai.equipment.is_lock_release_warranted`.

    Args:
        ctx: Decision context.
        locked_target: Currently locked equipment container.

    Returns:
        The markedly closer candidate, or ``None`` to keep the lock.
    """
    candidate = find_nearest_equipment(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        allow_unreachable=True,
        now_ms=ctx.timestamp_ms,
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
    return candidate


def _plan_known_equipment_target(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    *,
    score: int,
) -> TickDecisionDict | None:
    """Approach a previously validated equipment target before blind search."""
    for container in find_known_equipment_candidates(
        ctx.filtered,
        ctx.self_state,
        now_ms=ctx.timestamp_ms,
    ):
        target_x = container["x"]
        target_y = container["y"]
        if _is_equipment_target_attempted(ctx, target_x, target_y):
            continue
        command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="equipment")
        if command is None:
            continue
        emit_ai("approach known equipment at (%d,%d)", target_x, target_y)
        updated_state = _with_equipment_approach_recorded(
            ctx,
            AIStateDict(
                **{
                    **set_resource_target(ai_state, "equipment", target_x, target_y),
                    "equipment_search_failures": 0,
                }
            ),
            command,
            target_x,
            target_y,
        )
        return make_decision(
            command,
            "COLLECT_EQUIPMENT",
            score,
            target_x,
            target_y,
            "known_equipment",
            updated_state,
            ctx.equip,
        )
    return None


def _plan_equipment_sense_or_search(
    ctx: DecideCtx,
    score: int,
    ai_state: AIStateDict,
) -> TickDecisionDict | None:
    """Sense the current viewport or hop to a fresh sector for equipment.

    Args:
        ctx: Decision context.
        score: Behavior score for the sensing/search action.
        ai_state: Base AI state to rewrite.

    Returns:
        Radar or teleport-search decision, or ``None`` when neither is legal.
    """
    # At zero extra radars the viewport sweep is impossible, so grid-
    # sweep the free built-in 5x5 to refill instead of scanning and
    # ring-hopping blind. This is the single shared point so both the
    # durable owner and the weapon-emergency helper forage. Above zero
    # extras the existing viewport radar + ring hop runs.
    if ctx.inventory["extra_radars"]["count"] == 0:
        forage_decision = plan_forage_search(ctx, ai_state, score)
        if forage_decision is not None:
            return forage_decision
    if can_use_radar(ctx) and should_scan_resources_in_current_viewport(ctx):
        emit_ai(
            "radar to find equipment (dual=%d homing=%d radar=%d)",
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
        return make_decision(
            make_radar_command(),
            "COLLECT_EQUIPMENT",
            score,
            0,
            0,
            "radar_for_equipment",
            AIStateDict(
                **{
                    **ai_state,
                    "last_scan_ms": ctx.timestamp_ms,
                }
            ),
            ctx.equip,
        )
    return _plan_equipment_search(
        ctx,
        score=score,
        ai_state=ai_state,
        failure_count=ctx.ai_state["equipment_search_failures"],
    )


def _plan_equipment_search(
    ctx: DecideCtx,
    *,
    score: int,
    ai_state: AIStateDict,
    failure_count: int,
) -> TickDecisionDict | None:
    """Plan the next teleport-search hop for equipment recovery.

    Args:
        ctx: Decision context.
        score: Behavior score for the hop.
        ai_state: Base AI state to rewrite.
        failure_count: Current equipment-search failure count.

    Returns:
        Teleport-search decision, or ``None`` when search cannot proceed.
    """
    if failure_count >= ctx.config["equip_search_max_failures"]:
        emit_ai(
            "equipment search hit %d failures - continuing sweep (dual=%d homing=%d radar=%d)",
            failure_count,
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
        failure_count = 0

    return make_resource_search_hop(
        ctx,
        mode="COLLECT_EQUIPMENT",
        score=score,
        reason="search_equipment_local",
        failure_count=failure_count,
        ai_state=ai_state,
    )


__all__ = [
    "decide_recover_equipment_mode",
    "select_equipment_target",
    "try_search_critical_equipment",
]
