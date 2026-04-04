"""AI strategy for the tick loop.

Priority: fuel (< 500) → equipment (any item < 20) → combat.

Combat phases (one action per tick):
  Phase 0: Open map (get fresh enemy positions)
  Phase 1: Teleport to enemy
  Phase 2: Shoot — hit → stay, miss → back to phase 0
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.combat_strategy import combat_in_progress, try_combat
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport_search,
    clear_resource_target,
    equipment_low,
    is_current_viewport_scan_failed,
    locked_resource_target,
    make_decision,
    needs_emergency_equipment,
    require_command,
    set_resource_target,
    should_scan_resources_in_current_viewport,
)
from tankpit_bot.bot.ai.equipment import (
    describe_container_search,
    find_best_fuel,
    find_equipment_candidates,
    is_current_viewport_scanned,
)
from tankpit_bot.bot.ai.movement import (
    select_exploration_command,
    walk_or_teleport,
)
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.types import (
    AIStateDict,
    BehaviorMode,
)
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    BotCommand,
    make_map_open_command,
    make_radar_command,
    make_teleport_command,
)
from tankpit_bot.inventory import InventoryState
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.types import (
    ContainerStateDict,
    SelfStateDict,
    WorldStateDict,
)

# =============================================================================
# Public entry point
# =============================================================================


def decide(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    inventory: InventoryState,
    timestamp_ms: int,
    terrain: TerrainMapProtocol | None,
    combat_feedback: CombatFeedback = "",
) -> TickDecisionDict:
    """Run one AI decision cycle.

    Priority: fuel → equipment → combat.

    Args:
        world: Current world state with tanks, containers, mines.
        self_state: Player's own state (position, fuel, team, rank).
        ai_state: Current AI state (config, cooldowns, combat target).
        inventory: Current equipment inventory.
        timestamp_ms: Current game timestamp in milliseconds.
        terrain: Optional terrain map for reachability checks.
        combat_feedback: Protocol-level hit/miss from weapon byte.

    Returns:
        TickDecisionDict with command, behavior, updated AI state, and
        desired equipment slots.
    """
    ctx = DecideCtx(world, self_state, ai_state, inventory, timestamp_ms, terrain, combat_feedback)

    combat_locked = combat_in_progress(ctx)

    return (
        _try_collect_critical_fuel(ctx)
        or _try_collect_critical_equipment(ctx)
        or _try_search_critical_equipment(ctx)
        or (try_combat(ctx) if combat_locked else None)
        or (_try_collect_fuel(ctx) if not combat_locked else None)
        or (_try_collect_equipment(ctx) if not combat_locked else None)
        or try_combat(ctx)
        or _fallback_find_enemies(ctx)
    )


# =============================================================================
# Decision steps — each returns TickDecisionDict | None
# =============================================================================


def _try_collect_equipment(ctx: DecideCtx) -> TickDecisionDict | None:
    """Collect equipment when any inventory item is below 10.

    Fires before new combat starts, but not in the middle of an active
    engagement. That prevents pickup_move from dragging the tank away after
    a teleport or just before a follow-up shot.
    """
    if not equipment_low(ctx.inventory):
        return None
    base_state, locked_target = locked_resource_target(ctx, "equipment")
    if locked_target is not None:
        tx, ty = locked_target["x"], locked_target["y"]
        locked_command = walk_or_teleport(ctx, tx, ty, pickup_kind="equipment")
        if locked_command is not None:
            emit_ai("continue locked equipment target at (%d,%d)", tx, ty)
            return make_decision(
                locked_command,
                "COLLECT_EQUIPMENT",
                800,
                tx,
                ty,
                "equipment_locked",
                set_resource_target(base_state, "equipment", tx, ty),
                ctx.equip,
            )
        emit_ai("locked equipment target at (%d,%d) no longer executable", tx, ty)
        base_state = clear_resource_target(base_state)
    selection = _select_equipment_target(ctx, allow_unreachable=True)
    if selection is None:
        emit_ai(
            "no executable equipment target (%s)",
            describe_container_search(
                ctx.filtered,
                ctx.self_state,
                ctx.terrain,
                want_fuel=False,
                allow_unreachable=True,
            ),
        )
        if ctx.inventory["extra_radars"]["count"] > 0 and should_scan_resources_in_current_viewport(
            ctx
        ):
            emit_ai(
                "radar to find equipment (dual=%d homing=%d radar=%d)",
                ctx.inventory["dual_shots"]["count"],
                ctx.inventory["homing_shots"]["count"],
                ctx.inventory["extra_radars"]["count"],
            )
            return make_decision(
                make_radar_command(),
                "COLLECT_EQUIPMENT",
                800,
                0,
                0,
                "radar_for_equipment",
                AIStateDict(**{**base_state, "last_scan_ms": ctx.timestamp_ms}),
                ctx.equip,
            )
        search = _make_resource_search_hop(
            ctx,
            mode="COLLECT_EQUIPMENT",
            score=800,
            reason="search_equipment_local",
            ai_state=base_state,
        )
        if search is not None:
            return search
        return None
    equip_target, cmd = selection
    tx, ty = equip_target["x"], equip_target["y"]
    emit_ai("collect equipment at (%d,%d)", tx, ty)
    return make_decision(
        cmd,
        "COLLECT_EQUIPMENT",
        800,
        tx,
        ty,
        "equipment_low",
        set_resource_target(base_state, "equipment", tx, ty),
        ctx.equip,
    )


def _try_collect_critical_equipment(ctx: DecideCtx) -> TickDecisionDict | None:
    """Collect equipment urgently when below break threshold.

    Uses break/resume thresholds: enters emergency mode at break,
    stays until reserve is fully restored above resume.
    """
    if not needs_emergency_equipment(ctx):
        return None
    base_state, locked_target = locked_resource_target(ctx, "equipment")
    if locked_target is not None:
        tx, ty = locked_target["x"], locked_target["y"]
        locked_command = walk_or_teleport(ctx, tx, ty, pickup_kind="equipment")
        if locked_command is not None:
            emit_ai("continue locked critical equipment target at (%d,%d)", tx, ty)
            return make_decision(
                locked_command,
                "COLLECT_EQUIPMENT",
                950,
                tx,
                ty,
                "equipment_locked",
                AIStateDict(
                    **{
                        **set_resource_target(base_state, "equipment", tx, ty),
                        "equipment_search_failures": 0,
                    },
                ),
                ctx.equip,
            )
        emit_ai("locked critical equipment target at (%d,%d) no longer executable", tx, ty)
        base_state = clear_resource_target(base_state)
    selection = _select_equipment_target(ctx, allow_unreachable=True)
    if selection is None:
        emit_ai(
            "no executable critical equipment target (%s)",
            describe_container_search(
                ctx.filtered,
                ctx.self_state,
                ctx.terrain,
                want_fuel=False,
                allow_unreachable=True,
            ),
        )
        return None
    equip_target, cmd = selection
    tx, ty = equip_target["x"], equip_target["y"]
    emit_ai("collect critical equipment at (%d,%d)", tx, ty)
    return make_decision(
        cmd,
        "COLLECT_EQUIPMENT",
        950,
        tx,
        ty,
        "equipment_critical",
        AIStateDict(
            **{
                **set_resource_target(base_state, "equipment", tx, ty),
                "equipment_search_failures": 0,
            },
        ),
        ctx.equip,
    )


def _try_search_critical_equipment(ctx: DecideCtx) -> TickDecisionDict | None:
    """Search for equipment via local hops when no nearby target exists.

    Replaces the old static cross-map patrol teleport fallback with
    sector-to-sector recovery hops. Repeated failed hops rotate the
    search direction rather than dropping back into hunt immediately,
    and fuel stabilization still takes precedence when fuel is low.

    Args:
        ctx: Decision context.

    Returns:
        Tick decision that scans or relocates for equipment, or None.
    """
    if not needs_emergency_equipment(ctx):
        return None
    base_state = (
        clear_resource_target(ctx.base)
        if ctx.base["resource_target_kind"] == "equipment"
        else ctx.base
    )

    failures = ctx.ai_state["equipment_search_failures"]
    if failures >= ctx.config["equip_search_max_failures"]:
        emit_ai(
            "equipment search hit %d failures - continuing sweep (dual=%d homing=%d radar=%d)",
            failures,
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
        failures = 0

    # Don't burn fuel on search if fuel is already low
    if ctx.fuel < ctx.config["fuel_low_threshold"]:
        emit_ai("skipping equipment search - fuel too low (%d)", ctx.fuel)
        return None

    if ctx.inventory["extra_radars"]["count"] > 0 and should_scan_resources_in_current_viewport(
        ctx
    ):
        emit_ai(
            "radar to find equipment (dual=%d homing=%d radar=%d)",
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
        return make_decision(
            make_radar_command(),
            "COLLECT_EQUIPMENT",
            925,
            0,
            0,
            "radar_for_equipment",
            AIStateDict(**{**base_state, "last_scan_ms": ctx.timestamp_ms}),
            ctx.equip,
        )

    # Local hop: short teleport in a rotating cardinal direction
    return _make_resource_search_hop(
        ctx,
        mode="COLLECT_EQUIPMENT",
        score=925,
        reason="search_equipment_local",
        failure_count=failures,
        ai_state=base_state,
    )


def _try_collect_critical_fuel(ctx: DecideCtx) -> TickDecisionDict | None:
    """Allow critical fuel to interrupt any mode, including combat."""
    if ctx.fuel >= ctx.config["fuel_critical_threshold"]:
        return None
    return _try_collect_fuel(ctx)


def _try_collect_fuel(ctx: DecideCtx) -> TickDecisionDict | None:
    """Collect fuel when below low threshold (500).

    If a fuel container (volume >= 500) is nearby, walk to it.
    If no fuel is found in the current scanned viewport, teleport to
    a fresh sector before falling back to edge walking.
    """
    if ctx.fuel >= ctx.config["fuel_low_threshold"]:
        return None
    base_state, locked_target = locked_resource_target(ctx, "fuel")
    if locked_target is not None:
        tx, ty = locked_target["x"], locked_target["y"]
        locked_command = walk_or_teleport(ctx, tx, ty, pickup_kind="fuel")
        if locked_command is not None:
            emit_ai(
                "continue locked fuel target at (%d,%d) vol=%d (fuel=%d)",
                tx,
                ty,
                locked_target["volume"],
                ctx.fuel,
            )
            return make_decision(
                locked_command,
                "COLLECT_FUEL",
                900,
                tx,
                ty,
                f"fuel={locked_target['volume']}",
                set_resource_target(base_state, "fuel", tx, ty),
                ctx.equip,
            )
        emit_ai("locked fuel target at (%d,%d) no longer executable", tx, ty)
        base_state = clear_resource_target(base_state)
    target = find_best_fuel(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        allow_unreachable=True,
        now_ms=ctx.timestamp_ms,
    )
    if target is not None:
        tx, ty = target["x"], target["y"]
        emit_ai("collect fuel at (%d,%d) vol=%d (fuel=%d)", tx, ty, target["volume"], ctx.fuel)
        cmd = _require_command(
            walk_or_teleport(ctx, tx, ty, pickup_kind="fuel"),
            tx,
            ty,
            "fuel",
        )
        return make_decision(
            cmd,
            "COLLECT_FUEL",
            900,
            tx,
            ty,
            f"fuel={target['volume']}",
            set_resource_target(base_state, "fuel", tx, ty),
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
            minimum_volume=100,
        ),
    )
    # No fuel visible — use radar for newly-entered unconfirmed viewports,
    # otherwise hop to a fresh sector instead of screen-edge wandering.
    if should_scan_resources_in_current_viewport(ctx):
        emit_ai("radar to find fuel (fuel=%d)", ctx.fuel)
        return make_decision(
            make_radar_command(),
            "COLLECT_FUEL",
            900,
            0,
            0,
            "radar_for_fuel",
            AIStateDict(**{**base_state, "last_scan_ms": ctx.timestamp_ms}),
            ctx.equip,
        )
    search = _make_resource_search_hop(
        ctx,
        mode="COLLECT_FUEL",
        score=900,
        reason="search_fuel_local",
        ai_state=base_state,
    )
    if search is not None:
        return search
    # Teleport unaffordable — last resort is to walk to a fresh viewport.
    emit_ai("walk to viewport edge for fuel (fuel=%d)", ctx.fuel)
    exploration = select_exploration_command(ctx)
    if exploration is None:
        return None
    edge_x, edge_y, edge_cmd = exploration
    return make_decision(
        edge_cmd,
        "COLLECT_FUEL",
        900,
        edge_x,
        edge_y,
        "edge_for_fuel",
        base_state,
        ctx.equip,
    )


def _fallback_find_enemies(ctx: DecideCtx) -> TickDecisionDict:
    """Fallback when no actionable combat/restock plan exists this tick.

    Respects map_open_cooldown_ms to avoid spamming map_open every tick.
    When the cooldown has not elapsed, uses radar instead (if available)
    or walks to viewport edge.
    """
    map_age = ctx.timestamp_ms - ctx.ai_state["last_map_open_ms"]
    if map_age < ctx.config["map_open_cooldown_ms"]:
        # Map open too recent — use radar if available
        scan_age = ctx.timestamp_ms - ctx.ai_state["last_scan_ms"]
        if (
            scan_age >= ctx.config["scan_cooldown_ms"]
            and ctx.inventory["extra_radars"]["count"] > 0
            and not is_current_viewport_scanned(ctx.filtered)
            and not is_current_viewport_scan_failed(ctx)
        ):
            emit_ai("radar to find enemies (map on cooldown)")
            return make_decision(
                make_radar_command(),
                "HUNT",
                0,
                0,
                0,
                "radar_for_enemies",
                AIStateDict(
                    **{**ctx.base, "last_scan_ms": ctx.timestamp_ms},
                ),
                ctx.equip,
            )
        # Radar on cooldown too — walk to edge for new area
        emit_ai("walk to viewport edge (map+radar on cooldown)")
        exploration = select_exploration_command(ctx)
        if exploration is not None:
            edge_x, edge_y, edge_cmd = exploration
            return make_decision(
                edge_cmd,
                "HUNT",
                0,
                edge_x,
                edge_y,
                "edge_for_enemies",
                ctx.base,
                ctx.equip,
            )
    threats = analyze_threats(ctx.filtered, ctx.self_state)
    if threats:
        emit_ai(
            "opening map (threats=%d dual=%d radar=%d)",
            len(threats),
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
    else:
        emit_ai("no visible threats - opening map")
    return make_decision(
        make_map_open_command(),
        "HUNT",
        0,
        0,
        0,
        "find_enemies",
        AIStateDict(
            **{**ctx.base, "last_map_open_ms": ctx.timestamp_ms},
        ),
        ctx.equip,
    )


# =============================================================================
# Helpers
# =============================================================================


# Local alias — used by _try_collect_fuel which was written before the
# context module split.  Keeping as a thin wrapper avoids importing
# require_command under a private name throughout the strategy functions.
_require_command = require_command


def _select_equipment_target(
    ctx: DecideCtx,
    *,
    allow_unreachable: bool,
) -> tuple[ContainerStateDict, BotCommand] | None:
    """Return the nearest equipment target with an executable command.

    Args:
        ctx: Decision context.
        allow_unreachable: Whether terrain-blocked targets may use teleport fallback.

    Returns:
        ``(container, command)`` for the first executable equipment target, or
        ``None`` when no visible equipment target can currently be executed.

    Raises:
        ValueError: If candidate selection returns a container that does not
            produce an executable command.
    """
    candidates = find_equipment_candidates(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        allow_unreachable=allow_unreachable,
        now_ms=ctx.timestamp_ms,
    )
    if not candidates:
        return None
    for container in candidates:
        command = walk_or_teleport(
            ctx,
            container["x"],
            container["y"],
            pickup_kind="equipment",
        )
        if command is None:
            continue
        return (container, command)
    return None


_CARDINAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)


def _local_equipment_search_hop(ctx: DecideCtx) -> tuple[int, int, int]:
    """Compute the next sector hop for resource search.

    Picks a cardinal direction based on ``patrol_waypoint_index`` and
    increases the hop radius every full cycle. This prevents the search
    loop from bouncing forever around the same four coordinates.

    Args:
        ctx: Decision context.

    Returns:
        Tuple of ``(x, y, next_direction_index)``.
    """
    raw_index = ctx.ai_state["patrol_waypoint_index"]
    index = raw_index % len(_CARDINAL_DIRECTIONS)
    dx, dy = _CARDINAL_DIRECTIONS[index]
    ring = 1 + (raw_index // len(_CARDINAL_DIRECTIONS))
    dist = ctx.config["equip_search_hop_distance"] * ring
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    hop_x = max(1, min(254, sx + dx * dist))
    hop_y = max(1, min(254, sy + dy * dist))
    next_index = raw_index + 1
    return (hop_x, hop_y, next_index)


def _make_resource_search_hop(
    ctx: DecideCtx,
    *,
    mode: BehaviorMode,
    score: int,
    reason: str,
    failure_count: int | None = None,
    ai_state: AIStateDict | None = None,
) -> TickDecisionDict | None:
    """Teleport to a fresh sector while recovering resources."""
    hop_x, hop_y, next_index = _local_equipment_search_hop(ctx)
    if not can_afford_teleport_search(ctx):
        emit_ai("cannot afford %s hop (fuel=%d)", reason, ctx.fuel)
        return None
    base_state = ctx.base if ai_state is None else ai_state
    cleared = clear_resource_target(base_state)
    if failure_count is not None:
        next_failures = failure_count + 1
        emit_ai(
            "local resource hop to (%d,%d) (dual=%d homing=%d radar=%d attempt=%d)",
            hop_x,
            hop_y,
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
            next_failures,
        )
        final_state = AIStateDict(
            **{
                **cleared,
                "patrol_waypoint_index": next_index,
                "equipment_search_failures": next_failures,
            },
        )
    else:
        emit_ai(
            "local resource hop to (%d,%d) (dual=%d homing=%d radar=%d)",
            hop_x,
            hop_y,
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
        final_state = AIStateDict(
            **{**cleared, "patrol_waypoint_index": next_index},
        )
    return make_decision(
        make_teleport_command(hop_x, hop_y),
        mode,
        score,
        hop_x,
        hop_y,
        reason,
        final_state,
        ctx.equip,
    )


__all__ = ["decide"]
