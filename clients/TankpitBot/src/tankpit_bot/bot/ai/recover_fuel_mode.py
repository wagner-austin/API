"""Durable fuel-recovery owner and shared fuel recovery helpers."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import clear_combat_target
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport,
    can_use_radar,
    clear_resource_target,
    equipment_reserve_restored,
    locked_resource_target,
    make_decision,
    set_resource_target,
    should_scan_resources_in_current_viewport,
)
from tankpit_bot.bot.ai.equipment import (
    SCAN_COVERAGE_TTL_MS,
    describe_container_search,
    find_adjacent_container,
    find_best_fuel,
    find_known_fuel_candidates,
    is_lock_release_warranted,
)
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.recover_equipment_mode import select_equipment_target
from tankpit_bot.bot.ai.resource_search import (
    emit_fuel_dot_hop_diagnostic,
    make_recovery_edge_decision,
    make_recovery_map_intel_decision,
    make_resource_search_hop,
    record_attempt_mark,
    select_fuel_dot_hop,
    select_fuel_dot_walk_targets,
)
from tankpit_bot.bot.ai.threats import manhattan_distance
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    BotCommand,
    make_pickup_equipment_command,
    make_radar_command,
    make_teleport_command,
)
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
    if ctx.fuel <= ctx.config["fuel_critical_threshold"]:
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


def try_collect_critical_fuel(ctx: DecideCtx) -> TickDecisionDict | None:
    """Allow critical fuel to interrupt any mode, including combat.

    Args:
        ctx: Decision context.

    Returns:
        Fuel recovery decision, or ``None`` when fuel is not critical.
    """
    if ctx.fuel > ctx.config["fuel_critical_threshold"]:
        return None
    return _plan_fuel_recovery(ctx, owner_required=False)


def try_collect_fuel(ctx: DecideCtx) -> TickDecisionDict | None:
    """Collect fuel when below the low threshold.

    Args:
        ctx: Decision context.

    Returns:
        Fuel recovery decision, or ``None`` when fuel is healthy.
    """
    if ctx.fuel > ctx.config["fuel_low_threshold"]:
        return None
    return _plan_fuel_recovery(ctx, owner_required=False)


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
    if equipment_reserve_restored(ctx):
        return None
    # Locality is guaranteed by the selector's contract: visible
    # candidates are in-viewport by construction (the same bounds gate
    # candidacy and pickup actionability), so a sweep can never turn
    # into a relocation.
    return select_equipment_target(ctx, allow_unreachable=False)


def decide_recover_fuel_mode(ctx: DecideCtx) -> TickDecisionDict:
    """Run the durable ``RECOVER_FUEL`` owner for this tick.

    Args:
        ctx: Decision context.

    Returns:
        Mode-owned fuel recovery decision.

    Raises:
        ValueError: If the durable owner cannot legally produce a recovery
            action.
    """
    result = _plan_fuel_recovery(ctx, owner_required=True)
    if result is None:
        raise ValueError("RECOVER_FUEL owner failed to produce a decision with owner_required=True")
    return result


def _plan_fuel_recovery(
    ctx: DecideCtx,
    *,
    owner_required: bool,
) -> TickDecisionDict | None:
    """Plan the current tick's fuel recovery action.

    Args:
        ctx: Decision context.
        owner_required: Whether failing to produce an action is a hard error.

    Returns:
        Fuel recovery decision, or ``None`` when the non-owner helper cannot
        produce a legal recovery action.

    Raises:
        ValueError: If ``owner_required`` is true and no legal recovery action
            can be produced.
    """
    base_state, locked_target = locked_resource_target(ctx, "fuel")
    base_state = clear_combat_target(base_state)
    adjacent_equipment = find_adjacent_container(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        want_fuel=False,
        now_ms=ctx.timestamp_ms,
    )
    if adjacent_equipment is not None:
        emit_ai(
            "opportunistic equipment pickup at (%d,%d) during fuel recovery",
            adjacent_equipment["x"],
            adjacent_equipment["y"],
        )
        return make_decision(
            make_pickup_equipment_command(adjacent_equipment["x"], adjacent_equipment["y"]),
            "COLLECT_FUEL",
            900,
            adjacent_equipment["x"],
            adjacent_equipment["y"],
            "opportunistic_equipment",
            base_state,
            ctx.equip,
        )
    if locked_target is not None and _superior_fuel_candidate(ctx, locked_target) is not None:
        emit_ai(
            "releasing fuel lock at (%d,%d): markedly closer fuel is visible",
            locked_target["x"],
            locked_target["y"],
        )
        base_state = clear_resource_target(base_state)
        locked_target = None
    if locked_target is not None:
        target_x = locked_target["x"]
        target_y = locked_target["y"]
        locked_command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="fuel")
        if locked_command is not None:
            emit_ai(
                "continue locked fuel target at (%d,%d) vol=%d (fuel=%d)",
                target_x,
                target_y,
                locked_target["volume"],
                ctx.fuel,
            )
            return make_decision(
                locked_command,
                "COLLECT_FUEL",
                900,
                target_x,
                target_y,
                f"fuel={locked_target['volume']}",
                set_resource_target(base_state, "fuel", target_x, target_y),
                ctx.equip,
            )
        emit_ai("locked fuel target at (%d,%d) no longer executable", target_x, target_y)
        base_state = clear_resource_target(base_state)

    selection = select_fuel_target(ctx, allow_unreachable=True)
    if selection is not None:
        container, command = selection
        target_x = container["x"]
        target_y = container["y"]
        container_distance = manhattan_distance(
            ctx.self_state["x"],
            ctx.self_state["y"],
            target_x,
            target_y,
        )
        dot_refuel = _plan_fuel_dot_refuel(ctx, base_state, beat_distance=container_distance)
        if dot_refuel is not None:
            return dot_refuel
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
    dot_refuel = _plan_fuel_dot_refuel(ctx, base_state, beat_distance=None)
    if dot_refuel is not None:
        return dot_refuel
    known_target_decision = _plan_known_fuel_target(ctx, base_state)
    if known_target_decision is not None:
        return known_target_decision
    return _plan_fuel_sense_or_search(ctx, base_state, owner_required=owner_required)


def _plan_fuel_dot_refuel(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    *,
    beat_distance: int | None,
) -> TickDecisionDict | None:
    """Teleport to the nearest unrefuted map fuel dot as a refuel action.

    The dot atlas marks high-volume fuel (live probe 2026-06-11: 6/6
    visited dots held fuel, volumes 762-1189; the only off-dot fuel
    seen was volume 34/57), and teleporting onto a container tile picks
    it up on landing -- a dot teleport is a one-action refuel. It
    therefore outranks walking to a visible container whenever the dot
    is strictly closer, and outranks remembered targets, radar, and
    blind search hops outright.

    Args:
        ctx: Decision context.
        ai_state: Base AI state to rewrite for the produced command.
        beat_distance: When set, the dot must be strictly closer than
            this Manhattan distance (the competing visible container).
            ``None`` accepts the nearest affordable dot outright.

    Returns:
        Dot-refuel teleport decision, or ``None`` when no affordable
        unrefuted dot exists (or none beats the competing target).
    """
    dot = select_fuel_dot_hop(ctx)
    if dot is None:
        return None
    dot_x, dot_y = dot
    dot_distance = manhattan_distance(ctx.self_state["x"], ctx.self_state["y"], dot_x, dot_y)
    if beat_distance is not None and dot_distance >= beat_distance:
        return None
    emit_fuel_dot_hop_diagnostic(ctx, dot_x, dot_y)
    emit_ai(
        "teleporting to fuel dot at (%d,%d) dist=%d (fuel=%d)",
        dot_x,
        dot_y,
        dot_distance,
        ctx.fuel,
    )
    return make_decision(
        make_teleport_command(dot_x, dot_y),
        "COLLECT_FUEL",
        900,
        dot_x,
        dot_y,
        "fuel_dot_refuel",
        clear_resource_target(ai_state),
        ctx.equip,
    )


def _plan_fuel_dot_walk(
    ctx: DecideCtx,
    ai_state: AIStateDict,
) -> TickDecisionDict | None:
    """Walk toward the nearest unrefuted fuel dot as the stranded endgame.

    Walking is free at any fuel level and the tank auto-collects a
    container on any tile it enters (verified by the live dot probe:
    a teleport landing exactly on a dot tile picked it up). When fuel
    is too low for radar or any teleport, this is the only action that
    can still make refuel progress -- live run 20260612-131003 sat at
    7 fuel for 28 minutes spamming free map opens because no walking
    fallback existed.

    Only walk commands are accepted: teleport-shaped fallbacks from
    the movement planner are rejected because every teleport decision
    (with its operating reserve) already had its chance upstream.

    Args:
        ctx: Decision context.
        ai_state: Base AI state to rewrite for the produced command.

    Returns:
        Dot-walk move decision, or ``None`` when no dot has an
        executable walk route.
    """
    for dot_x, dot_y in select_fuel_dot_walk_targets(ctx):
        command = walk_or_teleport(ctx, dot_x, dot_y, pickup_kind=None)
        if command is None or command["cmd_type"] != "move":
            continue
        updated_state = clear_resource_target(ai_state)
        if command["target_x"] == dot_x and command["target_y"] == dot_y:
            updated_state = AIStateDict(
                **{
                    **updated_state,
                    "attempted_fuel_dots": record_attempt_mark(
                        ctx.ai_state["attempted_fuel_dots"],
                        dot_x,
                        dot_y,
                        ctx.timestamp_ms,
                        ttl_ms=SCAN_COVERAGE_TTL_MS,
                    ),
                }
            )
        emit_ai(
            "walking to fuel dot at (%d,%d) via (%d,%d) (fuel=%d)",
            dot_x,
            dot_y,
            command["target_x"],
            command["target_y"],
            ctx.fuel,
        )
        return make_decision(
            command,
            "COLLECT_FUEL",
            900,
            dot_x,
            dot_y,
            "fuel_dot_walk",
            updated_state,
            ctx.equip,
        )
    return None


def _plan_fuel_dot_escape(
    ctx: DecideCtx,
    ai_state: AIStateDict,
) -> TickDecisionDict | None:
    """Spend remaining fuel on the nearest affordable dot teleport.

    Reached only when every reserved action AND the free walking
    endgame have declined -- the marooned case. Live run
    20260612-131003 teleport-scattered onto a one-tile island in a
    lake at 87 fuel: no walk route existed in any direction, and the
    operating-reserve checks vetoed every escape teleport (cost 36-63)
    until scans bled the tank to 7, below even the shortest escape.
    Idling guarantees zero progress, so the marooned tank is allowed
    to spend its last fuel on the nearest unrefuted dot instead.

    Args:
        ctx: Decision context.
        ai_state: Base AI state to rewrite for the produced command.

    Returns:
        Escape teleport decision, or ``None`` when no dot is
        affordable even without a reserve.
    """
    targets = select_fuel_dot_walk_targets(ctx)
    if not targets:
        return None
    dot_x, dot_y = targets[0]
    if not can_afford_teleport(ctx, dot_x, dot_y):
        return None
    emit_fuel_dot_hop_diagnostic(ctx, dot_x, dot_y)
    emit_ai(
        "marooned: escape teleport to fuel dot at (%d,%d) (fuel=%d)",
        dot_x,
        dot_y,
        ctx.fuel,
    )
    attempted = record_attempt_mark(
        ctx.ai_state["attempted_fuel_dots"],
        dot_x,
        dot_y,
        ctx.timestamp_ms,
        ttl_ms=SCAN_COVERAGE_TTL_MS,
    )
    return make_decision(
        make_teleport_command(dot_x, dot_y),
        "COLLECT_FUEL",
        900,
        dot_x,
        dot_y,
        "fuel_dot_escape",
        AIStateDict(
            **{
                **clear_resource_target(ai_state),
                "attempted_fuel_dots": attempted,
            }
        ),
        ctx.equip,
    )


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


def _plan_known_fuel_target(
    ctx: DecideCtx,
    ai_state: AIStateDict,
) -> TickDecisionDict | None:
    """Approach a previously validated fuel target before blind searching."""
    for container in find_known_fuel_candidates(
        ctx.filtered,
        ctx.self_state,
        now_ms=ctx.timestamp_ms,
        minimum_volume=minimum_recovery_fuel_volume(ctx),
    ):
        target_x = container["x"]
        target_y = container["y"]
        command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="fuel")
        if command is None:
            continue
        emit_ai(
            "approach known fuel at (%d,%d) vol=%d (fuel=%d)",
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
            f"known_fuel={container['volume']}",
            set_resource_target(ai_state, "fuel", target_x, target_y),
            ctx.equip,
        )
    return None


def _plan_fuel_sense_or_search(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    *,
    owner_required: bool,
) -> TickDecisionDict | None:
    """Sense or reposition when no immediate fuel target exists.

    Args:
        ctx: Decision context.
        ai_state: Base AI state to rewrite.
        owner_required: Whether failing to produce an action is a hard error.

    Returns:
        Fuel recovery decision, or ``None`` when the non-owner helper cannot
        produce a legal recovery action.

    Raises:
        ValueError: If ``owner_required`` is true and no legal search action
            can be produced.
    """
    if can_use_fuel_radar(ctx) and should_scan_resources_in_current_viewport(ctx):
        emit_ai("radar to find fuel (fuel=%d)", ctx.fuel)
        return make_decision(
            make_radar_command(),
            "COLLECT_FUEL",
            900,
            0,
            0,
            "radar_for_fuel",
            AIStateDict(
                **{
                    **ai_state,
                    "last_scan_ms": ctx.timestamp_ms,
                }
            ),
            ctx.equip,
        )

    search = make_resource_search_hop(
        ctx,
        mode="COLLECT_FUEL",
        score=900,
        reason="search_fuel_local",
        ai_state=ai_state,
    )
    if search is not None:
        return search

    dot_walk = _plan_fuel_dot_walk(ctx, ai_state)
    if dot_walk is not None:
        return dot_walk

    edge = make_recovery_edge_decision(
        ctx,
        mode="COLLECT_FUEL",
        score=900,
        reason="edge_for_fuel",
        ai_state=ai_state,
    )
    if edge is not None:
        return edge

    escape = _plan_fuel_dot_escape(ctx, ai_state)
    if escape is not None:
        return escape

    if not owner_required:
        return None
    return make_recovery_map_intel_decision(
        ctx,
        mode="COLLECT_FUEL",
        score=900,
        reason="map_intel_for_fuel",
        ai_state=ai_state,
    )


__all__ = [
    "can_use_fuel_radar",
    "decide_recover_fuel_mode",
    "minimum_recovery_fuel_volume",
    "select_fuel_target",
    "try_collect_critical_fuel",
    "try_collect_fuel",
]
