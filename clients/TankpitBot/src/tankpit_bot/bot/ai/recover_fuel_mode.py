"""Durable fuel-recovery owner and shared fuel recovery helpers."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import clear_combat_target
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_use_radar,
    clear_resource_target,
    locked_resource_target,
    make_decision,
    set_resource_target,
)
from tankpit_bot.bot.ai.equipment import is_lock_release_warranted
from tankpit_bot.bot.ai.equipment_search import (
    describe_container_search,
    find_best_fuel,
)
from tankpit_bot.bot.ai.forage import plan_forage_search
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.recover_equipment_mode import select_equipment_target
from tankpit_bot.bot.ai.resource_search import make_resource_search_hop
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import BotCommand
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.types import ContainerStateDict

_RADAR_FUEL_COST = 10


def minimum_recovery_fuel_volume(ctx: DecideCtx) -> int:
    """Return the minimum fuel volume worth collecting for recovery.

    Args:
        ctx: Decision context.

    Returns:
        Minimum actionable fuel volume for the current recovery state.
    """
    if ctx.fuel <= ctx.config["fuel_low_threshold"]:
        return 1
    return 100


def can_use_fuel_radar(ctx: DecideCtx) -> bool:
    """Return True when a radar scan is legal for fuel recovery.

    Scans keep the hunt operating reserve intact: live run
    20260612-131003 burned 64 unreserved scans down to 7 fuel and sat
    stranded for 28 minutes. Below the reserve the dot walk owns the
    recovery instead -- walking is free at any fuel level.

    Args:
        ctx: Decision context.

    Returns:
        True when the bot can afford a radar scan this tick without
        dipping into the operating reserve.
    """
    return can_use_radar(ctx) and ctx.fuel >= _RADAR_FUEL_COST + ctx.config["hunt_min_fuel"]


def select_fuel_target(
    ctx: DecideCtx,
    *,
    allow_unreachable: bool,
) -> tuple[ContainerStateDict, BotCommand] | None:
    """Return the best executable visible fuel target.

    Args:
        ctx: Decision context.
        allow_unreachable: Whether terrain-blocked targets may use teleport fallback.

    Returns:
        ``(container, command)`` for the best executable fuel target, or
        ``None`` when no visible fuel target can currently be executed.
    """
    container = find_best_fuel(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        allow_unreachable=allow_unreachable,
        now_ms=ctx.timestamp_ms,
        minimum_volume=minimum_recovery_fuel_volume(ctx),
    )
    if container is None:
        return None

    command = walk_or_teleport(ctx, container["x"], container["y"], pickup_kind="fuel")
    if command is None:
        return None
    return (container, command)


def _sweep_equipment_before_leaving(
    ctx: DecideCtx,
) -> tuple[ContainerStateDict, BotCommand] | None:
    """Return walkable equipment worth taking before a search hop.

    Container sightings expire after the 30s freshness TTL, so
    equipment seen during a fuel trip is forgotten by the time
    equipment recovery wants it -- run 20260611-114346 spent five
    teleport hops (each with a map-open deferral) re-finding equipment
    that fuel trips had on screen minutes earlier. When no fuel target
    is actionable and equipment reserves are below the comfortable
    resume threshold, the in-viewport equipment is swept first. Sweeps
    are strictly local: only targets inside the current viewport
    qualify, so a sweep can never turn into a relocation.

    Args:
        ctx: Decision context.

    Returns:
        ``(container, command)`` for a walkable equipment sweep, or
        ``None`` when reserves are comfortable or nothing local exists.
    """
    return select_equipment_target(ctx, allow_unreachable=False)


def decide_recover_fuel_mode(ctx: DecideCtx) -> TickDecisionDict:
    """Run the durable ``RECOVER_FUEL`` owner for this tick.

    Cascade:

    1. Opportunistic equipment pickup if a reachable target is in the
       current viewport and fuel is above the low threshold.
    2. Continue or release a held fuel lock from a previous tick.
    3. Strict fuel pickup (best executable visible fuel target).
    4. Equipment sweep before leaving the area when no fuel target
       exists and equipment reserves are low.
    5. Sense (radar) when the viewport has unscanned tiles and the
       radar fuel cost stays above the operating reserve.
    6. Hop to a fresh viewport when nothing actionable remains here.

    Args:
        ctx: Decision context.

    Returns:
        Mode-owned fuel recovery decision.

    Raises:
        ValueError: When every cascade branch declines -- the bot is
            marooned and cannot produce a legal recovery action.
    """
    base_state, locked_target = locked_resource_target(ctx, "fuel")
    base_state = clear_combat_target(base_state)
    visible_equipment = (
        select_equipment_target(ctx, allow_unreachable=False)
        if ctx.fuel > ctx.config["fuel_low_threshold"]
        else None
    )
    if visible_equipment is not None:
        adjacent_equipment, equip_command = visible_equipment
        emit_ai(
            "opportunistic equipment pickup at (%d,%d) during fuel recovery",
            adjacent_equipment["x"],
            adjacent_equipment["y"],
        )
        return make_decision(
            equip_command,
            "COLLECT_FUEL",
            900,
            adjacent_equipment["x"],
            adjacent_equipment["y"],
            "opportunistic_equipment",
            base_state,
            ctx.equip,
        )
    locked_decision, base_state = _continue_or_release_fuel_lock(ctx, base_state, locked_target)
    if locked_decision is not None:
        return locked_decision

    selection = select_fuel_target(ctx, allow_unreachable=True)
    if selection is not None:
        container, command = selection
        target_x = container["x"]
        target_y = container["y"]
        emit_ai(
            "collect fuel at (%d,%d) vol=%d (fuel=%d)",
            target_x,
            target_y,
            container["volume"],
            ctx.fuel,
        )
        return make_decision(
            command,
            "COLLECT_FUEL",
            900,
            target_x,
            target_y,
            f"fuel={container['volume']}",
            set_resource_target(base_state, "fuel", target_x, target_y),
            ctx.equip,
        )

    sweep = _sweep_equipment_before_leaving(ctx)
    if sweep is not None:
        container, command = sweep
        emit_ai(
            "sweeping equipment at (%d,%d) before leaving the area",
            container["x"],
            container["y"],
        )
        return make_decision(
            command,
            "COLLECT_FUEL",
            900,
            container["x"],
            container["y"],
            "sweep_equipment",
            base_state,
            ctx.equip,
        )

    emit_ai(
        "no actionable fuel target (%s)",
        describe_container_search(
            ctx.filtered,
            ctx.self_state,
            ctx.terrain,
            want_fuel=True,
            allow_unreachable=True,
            minimum_volume=minimum_recovery_fuel_volume(ctx),
        ),
    )

    # Sense: radar (paid full viewport or free 5x5) when the viewport has
    # unscanned tiles and the cost stays above the operating reserve;
    # otherwise walk to expand free-radar coverage on the next tick.
    forage_decision = plan_forage_search(
        ctx,
        base_state,
        score=900,
        behavior_mode="COLLECT_FUEL",
        radar_affordable=can_use_fuel_radar(ctx),
    )
    if forage_decision is not None:
        return forage_decision

    # Hop: teleport to a fresh viewport when nothing here is actionable.
    search = make_resource_search_hop(
        ctx,
        mode="COLLECT_FUEL",
        score=900,
        reason="search_fuel_local",
        ai_state=base_state,
    )
    if search is not None:
        return search

    raise ValueError(
        f"RECOVER_FUEL owner produced no decision at "
        f"({ctx.self_state['x']},{ctx.self_state['y']}) fuel={ctx.fuel}: "
        f"forager exhausted, no affordable search hop."
    )


def _continue_or_release_fuel_lock(
    ctx: DecideCtx,
    base_state: AIStateDict,
    locked_target: ContainerStateDict | None,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    """Resolve the held fuel lock for this tick.

    Three outcomes:

    1. No lock held → returns ``(None, base_state)`` unchanged.
    2. Lock released because a markedly closer fuel candidate is visible
       (see :func:`is_lock_release_warranted`) → returns ``(None,
       cleared_state)`` so the caller picks the closer target on the
       same tick.
    3. Lock still active → if the locked target is still executable,
       returns ``(continuation_decision, base_state)``; if it isn't,
       clears the lock and returns ``(None, cleared_state)``.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite when the lock clears.
        locked_target: Currently locked fuel container, or ``None``.

    Returns:
        ``(decision, updated_base_state)`` -- the decision is non-None
        only when the locked target produced an executable continuation
        command this tick.
    """
    if locked_target is None:
        return None, base_state
    if _superior_fuel_candidate(ctx, locked_target) is not None:
        emit_ai(
            "releasing fuel lock at (%d,%d): markedly closer fuel is visible",
            locked_target["x"],
            locked_target["y"],
        )
        return None, clear_resource_target(base_state)
    target_x = locked_target["x"]
    target_y = locked_target["y"]
    locked_command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="fuel")
    if locked_command is None:
        emit_ai("locked fuel target at (%d,%d) no longer executable", target_x, target_y)
        return None, clear_resource_target(base_state)
    emit_ai(
        "continue locked fuel target at (%d,%d) vol=%d (fuel=%d)",
        target_x,
        target_y,
        locked_target["volume"],
        ctx.fuel,
    )
    decision = make_decision(
        locked_command,
        "COLLECT_FUEL",
        900,
        target_x,
        target_y,
        f"fuel={locked_target['volume']}",
        set_resource_target(base_state, "fuel", target_x, target_y),
        ctx.equip,
    )
    return decision, base_state


def _superior_fuel_candidate(
    ctx: DecideCtx,
    locked_target: ContainerStateDict,
) -> ContainerStateDict | None:
    """Return a fresh fuel candidate that warrants dropping the lock.

    Locks prevent target churn, but live run 20260610-011x showed the
    bot walking across the map to a locked container past abundant
    nearby fuel. A candidate releases the lock only when it clears the
    half-distance + minimum-gap rule in
    :func:`tankpit_bot.bot.ai.equipment.is_lock_release_warranted`.

    Args:
        ctx: Decision context.
        locked_target: Currently locked fuel container.

    Returns:
        The markedly closer candidate, or ``None`` to keep the lock.
    """
    candidate = find_best_fuel(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        allow_unreachable=True,
        now_ms=ctx.timestamp_ms,
        minimum_volume=minimum_recovery_fuel_volume(ctx),
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


__all__ = [
    "can_use_fuel_radar",
    "decide_recover_fuel_mode",
    "minimum_recovery_fuel_volume",
    "select_fuel_target",
]
