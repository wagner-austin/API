"""AI strategy for the tick loop.

Priority: fuel (< 500) → equipment (any item < 20) → combat.

Combat phases (one action per tick):
  Phase 0: Open map (get fresh enemy positions)
  Phase 1: Teleport to enemy
  Phase 2: Shoot — hit → stay, miss → back to phase 0
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.equipment import (
    describe_container_search,
    find_best_fuel,
    find_nearest_equipment,
    find_teleport_landing_tile,
)
from tankpit_bot.bot.ai.pathfinding import find_path_segment_target, is_direct_path_clear
from tankpit_bot.bot.ai.tactics import compute_desired_equipment
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.types import (
    AIConfigDict,
    AIStateDict,
    BehaviorMode,
    EnemyThreatDict,
    make_behavior_score,
)
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import (
    BotCommand,
    make_map_open_command,
    make_move_command,
    make_pickup_move_command,
    make_radar_command,
    make_shoot_command,
    make_teleport_command,
)
from tankpit_bot.inventory import InventoryState
from tankpit_bot.sniffer.world_state import is_move_target_failed
from tankpit_bot.state.types import SelfStateDict, WorldStateDict

log = get_logger(__name__)

_PICKUP_ACTION_MARGIN = 1


# =============================================================================
# Context — passed between decision steps to avoid repeated arg lists
# =============================================================================


class _DecideCtx:
    """Immutable context for one decide() call.

    Holds all inputs plus pre-computed values (equipment, filtered world,
    base AI state) so individual decision steps don't repeat work.
    """

    __slots__ = (
        "ai_state",
        "base",
        "blocked_targets",
        "combat_feedback",
        "config",
        "equip",
        "filtered",
        "fuel",
        "inventory",
        "killed",
        "self_state",
        "terrain",
        "timestamp_ms",
        "world",
    )

    def __init__(
        self,
        world: WorldStateDict,
        self_state: SelfStateDict,
        ai_state: AIStateDict,
        inventory: InventoryState,
        timestamp_ms: int,
        terrain: TerrainMapProtocol | None,
        combat_feedback: CombatFeedback,
    ) -> None:
        self.world = world
        self.self_state = self_state
        self.ai_state = ai_state
        self.inventory = inventory
        self.timestamp_ms = timestamp_ms
        self.terrain = terrain
        self.combat_feedback = combat_feedback

        self.config: AIConfigDict = ai_state["config"]
        self.fuel: int = self_state["fuel"]
        self.equip: list[int] = _compute_equipment(self.fuel, inventory)

        self.killed: dict[str, int] = _expire_kills(
            ai_state["killed_tank_ids"],
            timestamp_ms,
            self.config["kill_cooldown_ms"],
        )
        self.blocked_targets: dict[str, int] = _expire_kills(
            ai_state["blocked_combat_targets"],
            timestamp_ms,
            self.config["kill_cooldown_ms"],
        )
        self.filtered: WorldStateDict = _filter_killed_tanks(world, self.killed)
        self.base: AIStateDict = AIStateDict(
            **{
                **ai_state,
                "killed_tank_ids": self.killed,
                "blocked_combat_targets": self.blocked_targets,
                "last_shot_target_id": -1,
                "last_shot_target_name": "",
            },
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
    ctx = _DecideCtx(world, self_state, ai_state, inventory, timestamp_ms, terrain, combat_feedback)

    combat_locked = _combat_in_progress(ctx)

    return (
        _try_collect_critical_fuel(ctx)
        or _try_collect_critical_equipment(ctx)
        or _try_search_critical_equipment(ctx)
        or (_try_combat(ctx) if combat_locked else None)
        or (_try_collect_fuel(ctx) if not combat_locked else None)
        or (_try_collect_equipment(ctx) if not combat_locked else None)
        or _try_combat(ctx)
        or _fallback_find_enemies(ctx)
    )


# =============================================================================
# Decision steps — each returns TickDecisionDict | None
# =============================================================================


def _try_collect_equipment(ctx: _DecideCtx) -> TickDecisionDict | None:
    """Collect equipment when any inventory item is below 10.

    Fires before new combat starts, but not in the middle of an active
    engagement. That prevents pickup_move from dragging the tank away after
    a teleport or just before a follow-up shot.
    """
    if not _equipment_low(ctx.inventory):
        return None
    equip_target = find_nearest_equipment(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        allow_unreachable=True,
        now_ms=ctx.timestamp_ms,
    )
    if equip_target is None:
        log.info(
            "AI: no actionable equipment target (%s)",
            describe_container_search(
                ctx.filtered,
                ctx.self_state,
                ctx.terrain,
                want_fuel=False,
                allow_unreachable=True,
            ),
        )
        return None
    tx, ty = equip_target["x"], equip_target["y"]
    log.info("AI: collect equipment at (%d,%d)", tx, ty)
    cmd = _require_command(_walk_or_teleport(ctx, tx, ty), tx, ty, "equipment")
    return _make(cmd, "COLLECT_EQUIPMENT", 800, tx, ty, "equipment_low", ctx.base, ctx.equip)


def _try_collect_critical_equipment(ctx: _DecideCtx) -> TickDecisionDict | None:
    """Collect equipment urgently when below break threshold.

    Uses break/resume thresholds: enters emergency mode at break,
    stays until reserve is fully restored above resume.
    """
    if not _needs_emergency_equipment(ctx):
        return None
    equip_target = find_nearest_equipment(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        allow_unreachable=True,
        now_ms=ctx.timestamp_ms,
    )
    if equip_target is None:
        log.info(
            "AI: no actionable critical equipment target (%s)",
            describe_container_search(
                ctx.filtered,
                ctx.self_state,
                ctx.terrain,
                want_fuel=False,
                allow_unreachable=True,
            ),
        )
        return None
    tx, ty = equip_target["x"], equip_target["y"]
    log.info("AI: collect critical equipment at (%d,%d)", tx, ty)
    cmd = _require_command(_walk_or_teleport(ctx, tx, ty), tx, ty, "critical equipment")
    return _make(
        cmd,
        "COLLECT_EQUIPMENT",
        950,
        tx,
        ty,
        "equipment_critical",
        AIStateDict(**{**ctx.base, "equipment_search_failures": 0}),
        ctx.equip,
    )


def _try_search_critical_equipment(ctx: _DecideCtx) -> TickDecisionDict | None:
    """Search for equipment via local hops when no nearby target exists.

    Replaces the old static cross-map patrol teleport fallback with
    short-range local hops (~15 tiles). Bails out after repeated
    failures and defers to fuel stabilization when fuel is low.

    Args:
        ctx: Decision context.

    Returns:
        Tick decision that scans or relocates for equipment, or None.
    """
    if not _needs_emergency_equipment(ctx):
        return None

    # Bail out after too many consecutive search failures
    failures = ctx.ai_state["equipment_search_failures"]
    if failures >= ctx.config["equip_search_max_failures"]:
        log.info(
            "AI: equipment search bailed out after %d failures (dual=%d radar=%d)",
            failures,
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
        return None

    # Don't burn fuel on search if fuel is already low
    if ctx.fuel < ctx.config["fuel_low_threshold"]:
        log.info("AI: skipping equipment search — fuel too low (%d)", ctx.fuel)
        return None

    if ctx.inventory["extra_radars"]["count"] > 0:
        scan_age = ctx.timestamp_ms - ctx.ai_state["last_scan_ms"]
        if scan_age >= ctx.config["scan_cooldown_ms"]:
            log.info(
                "AI: radar to find equipment (dual=%d radar=%d)",
                ctx.inventory["dual_shots"]["count"],
                ctx.inventory["extra_radars"]["count"],
            )
            return _make(
                make_radar_command(),
                "COLLECT_EQUIPMENT",
                925,
                0,
                0,
                "radar_for_equipment",
                AIStateDict(**{**ctx.base, "last_scan_ms": ctx.timestamp_ms}),
                ctx.equip,
            )

    # Local hop: short teleport in a rotating cardinal direction
    hop_x, hop_y, next_index = _local_equipment_search_hop(ctx)
    if not _can_afford_teleport_search(ctx):
        log.info("AI: cannot afford equipment search hop (fuel=%d)", ctx.fuel)
        return None

    next_failures = failures + 1
    log.info(
        "AI: local equipment search hop to (%d,%d) (dual=%d radar=%d attempt=%d)",
        hop_x,
        hop_y,
        ctx.inventory["dual_shots"]["count"],
        ctx.inventory["extra_radars"]["count"],
        next_failures,
    )
    return _make(
        make_teleport_command(hop_x, hop_y),
        "COLLECT_EQUIPMENT",
        925,
        hop_x,
        hop_y,
        "search_equipment_local",
        AIStateDict(
            **{
                **ctx.base,
                "patrol_waypoint_index": next_index,
                "equipment_search_failures": next_failures,
            }
        ),
        ctx.equip,
    )


def _try_collect_critical_fuel(ctx: _DecideCtx) -> TickDecisionDict | None:
    """Allow critical fuel to interrupt any mode, including combat."""
    if ctx.fuel >= ctx.config["fuel_critical_threshold"]:
        return None
    return _try_collect_fuel(ctx)


def _try_collect_fuel(ctx: _DecideCtx) -> TickDecisionDict | None:
    """Collect fuel when below low threshold (500).

    If a fuel container (volume >= 500) is nearby, walk to it.
    If no fuel visible, walk to viewport edge to trigger a radar
    scan in a new area.
    """
    if ctx.fuel >= ctx.config["fuel_low_threshold"]:
        return None
    target = find_best_fuel(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        allow_unreachable=True,
        now_ms=ctx.timestamp_ms,
    )
    if target is not None:
        tx, ty = target["x"], target["y"]
        log.info("AI: collect fuel at (%d,%d) vol=%d (fuel=%d)", tx, ty, target["volume"], ctx.fuel)
        cmd = _require_command(_walk_or_teleport(ctx, tx, ty), tx, ty, "fuel")
        return _make(
            cmd,
            "COLLECT_FUEL",
            900,
            tx,
            ty,
            f"fuel={target['volume']}",
            ctx.base,
            ctx.equip,
        )
    log.info(
        "AI: no actionable fuel target (%s)",
        describe_container_search(
            ctx.filtered,
            ctx.self_state,
            ctx.terrain,
            want_fuel=True,
            allow_unreachable=True,
            minimum_volume=100,
        ),
    )
    # No fuel visible — use radar if cooldown elapsed, otherwise walk to edge
    scan_age = ctx.timestamp_ms - ctx.ai_state["last_scan_ms"]
    if scan_age >= ctx.config["scan_cooldown_ms"]:
        log.info("AI: radar to find fuel (fuel=%d)", ctx.fuel)
        return _make(
            make_radar_command(),
            "COLLECT_FUEL",
            900,
            0,
            0,
            "radar_for_fuel",
            AIStateDict(**{**ctx.base, "last_scan_ms": ctx.timestamp_ms}),
            ctx.equip,
        )
    # Radar on cooldown — walk to viewport edge to search new area
    log.info("AI: walk to viewport edge for fuel (fuel=%d)", ctx.fuel)
    edge_x, edge_y = _viewport_edge_target(ctx)
    edge_cmd = _walk_or_teleport(ctx, edge_x, edge_y, pickup=False)
    if edge_cmd is None:
        return None
    return _make(
        edge_cmd,
        "COLLECT_FUEL",
        900,
        edge_x,
        edge_y,
        "edge_for_fuel",
        ctx.base,
        ctx.equip,
    )


def _try_combat(ctx: _DecideCtx) -> TickDecisionDict | None:
    """Route to the correct combat phase."""
    if ctx.fuel < ctx.config["fuel_low_threshold"] and not _combat_in_progress(ctx):
        return None
    # dual=0 is already handled by _combat_in_progress (releases lock)
    # and _try_collect_critical_equipment (intercepts before this point).

    threats = analyze_threats(ctx.filtered, ctx.self_state)
    if not threats:
        return None

    target = _get_locked_target(ctx, threats)
    if target is None:
        # Don't start new fights if equipment reserve isn't healthy
        if not _equipment_reserve_restored(ctx):
            return None
        viable = [t for t in threats if str(t["tank_id"]) not in ctx.blocked_targets]
        if not viable:
            return None
        target = viable[0]
        log.info("AI: new target %s (id=%d)", target["name"], target["tank_id"])
        return _combat_open_map(ctx, target)

    phase = ctx.ai_state["combat_phase"]

    if phase == "engaging":
        return _combat_shoot(ctx, target)
    if phase == "closing":
        return _combat_close(ctx, target)
    return _combat_open_map(ctx, target)


def _combat_open_map(ctx: _DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase 0: Open map to get fresh enemy positions. Next tick: teleport."""
    log.info("AI: open map to find %s", target["name"])
    return _make(
        make_map_open_command(),
        "HUNT",
        800,
        0,
        0,
        f"find {target['name']}",
        AIStateDict(
            **{
                **ctx.base,
                "combat_target_id": target["tank_id"],
                "combat_target_x": target["x"],
                "combat_target_y": target["y"],
                "last_map_open_ms": ctx.timestamp_ms,
                "combat_phase": "closing",
            }
        ),
        ctx.equip,
    )


def _combat_teleport(ctx: _DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase 1: Teleport to enemy. Next tick: shoot."""
    landing_x, landing_y = _combat_landing_tile(ctx, target)
    if landing_x == -1 and landing_y == -1:
        log.info("AI: no combat landing tile for %s, blocking target", target["name"])
        return _block_combat_target_and_replan(ctx, target)
    log.info("AI: teleport near %s to (%d,%d)", target["name"], landing_x, landing_y)
    return _make(
        make_teleport_command(landing_x, landing_y),
        "HUNT",
        800,
        landing_x,
        landing_y,
        f"teleport {target['name']}",
        AIStateDict(
            **{
                **ctx.base,
                "combat_target_id": target["tank_id"],
                "combat_target_x": target["x"],
                "combat_target_y": target["y"],
                "combat_phase": "closing",
            }
        ),
        ctx.equip,
    )


def _combat_close(ctx: _DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase closing: confirm actual post-teleport geometry before shooting."""
    if _has_cardinal_combat_shot(ctx.self_state, target):
        return _combat_shoot(ctx, target)
    log.info(
        "AI: not in cardinal firing position for %s from (%d,%d); re-closing",
        target["name"],
        ctx.self_state["x"],
        ctx.self_state["y"],
    )
    return _combat_teleport(ctx, target)


def _combat_shoot(ctx: _DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase engaging: Shoot. On hit: stay engaging. On miss: reacquire.

    A miss means the target is no longer at the locked coordinates.
    The only correct response is to reacquire: open map to get fresh
    enemy positions, then teleport to the updated location. Retrying
    the same stale coordinates would fire at empty ground indefinitely.
    """
    if ctx.combat_feedback == "miss":
        log.info("AI: miss — reopening map for %s", target["name"])
        return _combat_open_map(ctx, target)

    log.info("AI: shoot %s at (%d,%d)", target["name"], target["x"], target["y"])
    return _make(
        make_shoot_command(target["x"], target["y"], target["tank_id"]),
        "HUNT",
        800,
        target["x"],
        target["y"],
        f"shoot {target['name']}",
        AIStateDict(
            **{
                **ctx.base,
                "combat_target_id": target["tank_id"],
                "combat_target_x": target["x"],
                "combat_target_y": target["y"],
                "last_shoot_ms": ctx.timestamp_ms,
                "last_shot_target_id": target["tank_id"],
                "last_shot_target_name": target["name"],
                "combat_phase": "engaging",
            }
        ),
        ctx.equip,
    )


def _fallback_find_enemies(ctx: _DecideCtx) -> TickDecisionDict:
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
        ):
            log.info("AI: radar to find enemies (map on cooldown)")
            return _make(
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
        log.info("AI: walk to viewport edge (map+radar on cooldown)")
        edge_x, edge_y = _viewport_edge_target(ctx)
        edge_cmd = _walk_or_teleport(ctx, edge_x, edge_y, pickup=False)
        if edge_cmd is not None:
            return _make(
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
        log.info(
            "AI: opening map (threats=%d dual=%d radar=%d)",
            len(threats),
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
    else:
        log.info("AI: no visible threats — opening map")
    return _make(
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


def _make(
    command: BotCommand,
    mode: BehaviorMode,
    score: int,
    tx: int,
    ty: int,
    reason: str,
    ai_state: AIStateDict,
    equip: list[int],
) -> TickDecisionDict:
    """Build a TickDecisionDict with less boilerplate."""
    behavior = make_behavior_score(mode, score, tx, ty, reason)
    return make_tick_decision(
        command=command,
        behavior=behavior,
        updated_ai_state=ai_state,
        desired_equipment=equip,
    )


def _require_command(
    command: BotCommand | None,
    tx: int,
    ty: int,
    target_kind: str,
) -> BotCommand:
    """Require a concrete command for an already-selected actionable target.

    Args:
        command: Command returned by movement planning.
        tx: Target X coordinate.
        ty: Target Y coordinate.
        target_kind: Human-readable target category for the error message.

    Returns:
        The planned BotCommand.

    Raises:
        ValueError: If no executable command exists for the selected target.
    """
    if command is None:
        raise ValueError(f"No executable command for {target_kind} target at ({tx},{ty})")
    return command


def _compute_equipment(fuel: int, inventory: InventoryState) -> list[int]:
    """Compute desired equipment as sorted list."""
    desired = compute_desired_equipment(
        "HUNT",
        fuel,
        dual_shots_count=inventory["dual_shots"]["count"],
    )
    return sorted(desired)


_EQUIP_LOW_THRESHOLD = 20


def _equipment_low(inventory: InventoryState) -> bool:
    """Check if any equipment slot is below the restock threshold."""
    return (
        inventory["armor_shields"]["count"] < _EQUIP_LOW_THRESHOLD
        or inventory["dual_shots"]["count"] < _EQUIP_LOW_THRESHOLD
        or inventory["missile_shots"]["count"] < _EQUIP_LOW_THRESHOLD
        or inventory["homing_shots"]["count"] < _EQUIP_LOW_THRESHOLD
        or inventory["extra_radars"]["count"] < _EQUIP_LOW_THRESHOLD
    )


def _needs_emergency_equipment(ctx: _DecideCtx) -> bool:
    """Check if dual or radar have dropped below the break threshold.

    Uses the config break thresholds instead of a hard-coded constant.
    """
    return (
        ctx.inventory["dual_shots"]["count"] < ctx.config["dual_break_threshold"]
        or ctx.inventory["extra_radars"]["count"] < ctx.config["radar_break_threshold"]
    )


def _equipment_reserve_restored(ctx: _DecideCtx) -> bool:
    """Check if equipment has been restocked above the resume threshold.

    The bot stays in emergency equipment mode until BOTH dual and radar
    are above their respective resume thresholds.
    """
    return (
        ctx.inventory["dual_shots"]["count"] >= ctx.config["dual_resume_threshold"]
        and ctx.inventory["extra_radars"]["count"] >= ctx.config["radar_resume_threshold"]
    )


def _expire_kills(killed: dict[str, int], now: int, cooldown_ms: int) -> dict[str, int]:
    """Remove expired entries from killed tank IDs."""
    return {k: v for k, v in killed.items() if now - v < cooldown_ms}


def _filter_killed_tanks(world: WorldStateDict, killed: dict[str, int]) -> WorldStateDict:
    """Remove killed tanks from world state."""
    if not killed:
        return world
    filtered = {k: v for k, v in world["tanks"].items() if k not in killed}
    return WorldStateDict(
        self_state=world["self_state"],
        tanks=filtered,
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=world["viewport"],
        timestamp_ms=world["timestamp_ms"],
    )


def _get_locked_target(
    ctx: _DecideCtx,
    threats: list[EnemyThreatDict],
) -> EnemyThreatDict | None:
    """Find the current combat target in the threat list.

    Returns the locked target if it's still alive and in the world.
    Returns None if no target locked or target is dead/gone.
    """
    target_id = ctx.ai_state["combat_target_id"]
    if target_id == -1:
        return None
    for t in threats:
        if t["tank_id"] == target_id:
            return t
    return None


def _combat_in_progress(ctx: _DecideCtx) -> bool:
    """Return True when the AI is in an active combat engagement.

    Returns False when combat is no longer executable:
    - dual depleted (can't shoot)
    - locked target was killed (should recover before reacquiring)

    This releases the combat lock so the planner can fall through to
    equipment/fuel recovery instead of immediately reacquiring.
    """
    target_id = ctx.ai_state["combat_target_id"]
    return (
        target_id != -1
        and str(target_id) not in ctx.killed
        and ctx.ai_state["combat_phase"] in ("closing", "engaging")
        and ctx.inventory["dual_shots"]["count"] != 0
    )


_CARDINAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)


def _local_equipment_search_hop(ctx: _DecideCtx) -> tuple[int, int, int]:
    """Compute a short-range local hop for equipment search.

    Picks a cardinal direction based on ``patrol_waypoint_index``,
    hops ``equip_search_hop_distance`` tiles from the current position,
    and clamps to map bounds [1, 254].

    Args:
        ctx: Decision context.

    Returns:
        Tuple of ``(x, y, next_direction_index)``.
    """
    index = ctx.ai_state["patrol_waypoint_index"] % len(_CARDINAL_DIRECTIONS)
    dx, dy = _CARDINAL_DIRECTIONS[index]
    dist = ctx.config["equip_search_hop_distance"]
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    hop_x = max(1, min(254, sx + dx * dist))
    hop_y = max(1, min(254, sy + dy * dist))
    next_index = (index + 1) % len(_CARDINAL_DIRECTIONS)
    return (hop_x, hop_y, next_index)


def _can_afford_teleport_search(ctx: _DecideCtx) -> bool:
    """Return True when equipment search can spend fuel on teleporting.

    Args:
        ctx: Decision context.

    Returns:
        True if current fuel can cover one teleport while staying above the
        critical fuel floor.
    """
    required_fuel = ctx.config["fuel_critical_threshold"] + ctx.config["teleport_fuel_cost"]
    return ctx.fuel >= required_fuel


def _combat_landing_tile(ctx: _DecideCtx, target: EnemyThreatDict) -> tuple[int, int]:
    """Choose the tile to teleport to for combat.

    Combat teleports should land adjacent to the enemy rather than on the
    enemy's exact coordinates. That avoids landing mismatches when the server
    resolves occupied tiles to a nearby square.

    Args:
        ctx: Decision context.
        target: Enemy threat currently being engaged.

    Returns:
        Tuple of landing coordinates.
    """
    candidates = _combat_landing_candidates(ctx, target)
    if not candidates:
        return (-1, -1)

    if ctx.terrain is not None:
        for candidate_x, candidate_y in candidates:
            if ctx.terrain.is_passable(candidate_x, candidate_y):
                return (candidate_x, candidate_y)
        return (-1, -1)

    return candidates[0]


def _combat_landing_candidates(
    ctx: _DecideCtx,
    target: EnemyThreatDict,
) -> list[tuple[int, int]]:
    """Return usable adjacent landing tiles ordered by distance to self."""
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    candidates = [
        (target["x"] + 1, target["y"]),
        (target["x"] - 1, target["y"]),
        (target["x"], target["y"] + 1),
        (target["x"], target["y"] - 1),
    ]
    usable: list[tuple[int, int]] = []
    for candidate_x, candidate_y in candidates:
        if not (0 <= candidate_x <= 255 and 0 <= candidate_y <= 255):
            continue
        if _is_dynamically_occupied(ctx, candidate_x, candidate_y):
            continue
        usable.append((candidate_x, candidate_y))
    usable.sort(key=_combat_distance_key(sx, sy))
    return usable


def _combat_distance_key(sx: int, sy: int) -> Callable[[tuple[int, int]], int]:
    """Return a stable Manhattan-distance key for combat landing sort."""

    def key(pos: tuple[int, int]) -> int:
        return abs(pos[0] - sx) + abs(pos[1] - sy)

    return key


def _is_dynamically_occupied(ctx: _DecideCtx, x: int, y: int) -> bool:
    """Return True when a tile is occupied by a tank, container, or mine."""
    if any(tank["x"] == x and tank["y"] == y for tank in ctx.filtered["tanks"].values()):
        return True
    if f"{x},{y}" in ctx.world["containers"]:
        return True
    return f"{x},{y}" in ctx.world["mines"]


def _has_cardinal_combat_shot(
    self_state: SelfStateDict,
    target: EnemyThreatDict,
) -> bool:
    """Return True when self is cardinally adjacent to the target."""
    return abs(self_state["x"] - target["x"]) + abs(self_state["y"] - target["y"]) == 1


def _block_combat_target_and_replan(
    ctx: _DecideCtx,
    target: EnemyThreatDict,
) -> TickDecisionDict:
    """Block a combat target that has no landing tile and choose the next viable threat.

    Adds the target to blocked_combat_targets so it won't be reacquired until the
    TTL expires. If another viable threat exists, engages that one. Otherwise falls
    back to generic enemy search.

    Args:
        ctx: Decision context.
        target: The unreachable combat target.

    Returns:
        Tick decision for the next viable target, or fallback enemy search.
    """
    blocked = dict(ctx.blocked_targets)
    blocked[str(target["tank_id"])] = ctx.timestamp_ms
    base_with_block = AIStateDict(
        **{
            **ctx.base,
            "blocked_combat_targets": blocked,
            "combat_target_id": -1,
            "combat_target_x": 0,
            "combat_target_y": 0,
            "combat_phase": "none",
        }
    )

    # Try the next viable threat
    threats = analyze_threats(ctx.filtered, ctx.self_state)
    skip = {*blocked, *ctx.killed}
    viable = [t for t in threats if str(t["tank_id"]) not in skip]
    if viable:
        next_target = viable[0]
        log.info(
            "AI: blocked %s, switching to %s (id=%d)",
            target["name"],
            next_target["name"],
            next_target["tank_id"],
        )
        return _make(
            make_map_open_command(),
            "HUNT",
            800,
            0,
            0,
            f"find {next_target['name']}",
            AIStateDict(
                **{
                    **base_with_block,
                    "combat_target_id": next_target["tank_id"],
                    "combat_target_x": next_target["x"],
                    "combat_target_y": next_target["y"],
                    "last_map_open_ms": ctx.timestamp_ms,
                    "combat_phase": "closing",
                }
            ),
            ctx.equip,
        )

    log.info("AI: blocked %s, no viable threats remaining", target["name"])
    return _make(
        make_map_open_command(),
        "HUNT",
        0,
        0,
        0,
        "find_enemies",
        AIStateDict(**{**base_with_block, "last_map_open_ms": ctx.timestamp_ms}),
        ctx.equip,
    )


def _walk_or_teleport(
    ctx: _DecideCtx,
    tx: int,
    ty: int,
    *,
    pickup: bool = True,
) -> BotCommand | None:
    """Plan a direct walk, waypointed walk, or teleport for a target.

    The game can walk directly across clear ground, but does not reliably path
    around terrain obstacles. This helper therefore prefers:
    1. direct move/pickup when the straight route is clear
    2. a terrain-aware waypoint along the first A* segment
    3. teleport fallback when no walk route exists

    Rejects move destinations that are occupied by enemy tanks or that
    recently failed (stalled and timed out).
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]

    # Reject recently-failed move targets (unless this is a pickup —
    # pickup failures are tracked on the container, not here).
    if not pickup and is_move_target_failed(tx, ty, ctx.timestamp_ms):
        log.info("AI: skipping failed move target (%d,%d)", tx, ty)
        return None

    if ctx.terrain is not None:
        return _walk_or_teleport_with_terrain(ctx, tx, ty, sx, sy, pickup=pickup)
    return _walk_or_teleport_without_terrain(ctx, tx, ty, pickup=pickup)


def _is_pickup_target_actionable(ctx: _DecideCtx, tx: int, ty: int) -> bool:
    """Return True when a target is inside the reliable local pickup window.

    Radar/world-state can reveal one outer ring beyond where direct
    ``pickup_move`` reliably executes. The bot should approach those edge
    targets first so the viewport recenters and the target becomes locally
    actionable.

    Args:
        ctx: Decision context with current viewport.
        tx: Target X coordinate.
        ty: Target Y coordinate.

    Returns:
        True if the target is inside the local pickup action window.
    """
    viewport = ctx.world["viewport"]
    left = viewport["left"]
    top = viewport["top"]
    right = left + viewport["width"] - 1 - _PICKUP_ACTION_MARGIN
    bottom = top + viewport["height"] - 1 - _PICKUP_ACTION_MARGIN
    return left <= tx <= right and top <= ty <= bottom


def _pickup_approach_target(ctx: _DecideCtx, tx: int, ty: int) -> tuple[int, int]:
    """Clamp an outer-ring pickup target to the inner actionable window.

    Args:
        ctx: Decision context with current viewport.
        tx: Target X coordinate.
        ty: Target Y coordinate.

    Returns:
        Inner-window approach tile that moves the bot toward the target while
        keeping the command inside the locally actionable viewport.
    """
    viewport = ctx.world["viewport"]
    left = viewport["left"]
    top = viewport["top"]
    right = left + viewport["width"] - 1 - _PICKUP_ACTION_MARGIN
    bottom = top + viewport["height"] - 1 - _PICKUP_ACTION_MARGIN
    approach_x = min(max(tx, left), right)
    approach_y = min(max(ty, top), bottom)
    return (approach_x, approach_y)


def _is_occupied_by_enemy(ctx: _DecideCtx, x: int, y: int) -> bool:
    """Return True when a tile is occupied by an enemy tank.

    Prevents the bot from choosing a move waypoint that a tank is
    sitting on. The game server may reject or silently ignore such
    moves, causing the action to stall.

    Args:
        ctx: Decision context with current world state.
        x: Tile X coordinate to check.
        y: Tile Y coordinate to check.

    Returns:
        True if the tile is occupied by an enemy.
    """
    return any(tank["x"] == x and tank["y"] == y for tank in ctx.filtered["tanks"].values())


def _walk_or_teleport_with_terrain(
    ctx: _DecideCtx,
    tx: int,
    ty: int,
    sx: int,
    sy: int,
    *,
    pickup: bool,
) -> BotCommand | None:
    """Resolve movement when terrain/pathfinding is available."""
    terrain = ctx.terrain
    assert terrain is not None  # caller guarantees this
    if pickup and abs(sx - tx) <= 1 and abs(sy - ty) <= 1:
        log.info("AI: adjacent to pickup target at (%d,%d)", tx, ty)
        return make_pickup_move_command(tx, ty)
    if pickup and not _is_pickup_target_actionable(ctx, tx, ty):
        return _pickup_approach_command(ctx, tx, ty)
    if is_direct_path_clear(terrain, sx, sy, tx, ty):
        return _direct_move_command(ctx, tx, ty, pickup=pickup)
    waypoint = find_path_segment_target(terrain, sx, sy, tx, ty)
    if waypoint is not None:
        move_cmd = _waypoint_move_command(ctx, tx, ty, waypoint)
        if move_cmd is not None:
            return move_cmd
    return _teleport_fallback_command(terrain, sx, sy, tx, ty)


def _walk_or_teleport_without_terrain(
    ctx: _DecideCtx,
    tx: int,
    ty: int,
    *,
    pickup: bool,
) -> BotCommand | None:
    """Resolve movement when only local occupancy checks are available."""
    if pickup:
        return make_pickup_move_command(tx, ty)
    if _is_occupied_by_enemy(ctx, tx, ty):
        log.info("AI: move target (%d,%d) is occupied by enemy", tx, ty)
        return None
    return make_move_command(tx, ty)


def _pickup_approach_command(ctx: _DecideCtx, tx: int, ty: int) -> BotCommand | None:
    """Return a non-pickup approach command for an outer-ring pickup target."""
    approach_x, approach_y = _pickup_approach_target(ctx, tx, ty)
    log.info(
        "AI: pickup target (%d,%d) is outer-ring only, approaching via (%d,%d)",
        tx,
        ty,
        approach_x,
        approach_y,
    )
    return _walk_or_teleport(ctx, approach_x, approach_y, pickup=False)


def _direct_move_command(
    ctx: _DecideCtx,
    tx: int,
    ty: int,
    *,
    pickup: bool,
) -> BotCommand | None:
    """Return a direct move/pickup command when the straight path is clear."""
    if pickup:
        return make_pickup_move_command(tx, ty)
    if _is_occupied_by_enemy(ctx, tx, ty):
        log.info("AI: move target (%d,%d) is occupied by enemy", tx, ty)
        return None
    return make_move_command(tx, ty)


def _waypoint_move_command(
    ctx: _DecideCtx,
    tx: int,
    ty: int,
    waypoint: tuple[int, int],
) -> BotCommand | None:
    """Return an A*-derived waypoint move when the waypoint is usable."""
    viewport = ctx.world["viewport"]
    left = viewport["left"]
    top = viewport["top"]
    right = left + viewport["width"] - 1
    bottom = top + viewport["height"] - 1
    wx, wy = waypoint
    if not (left <= wx <= right and top <= wy <= bottom):
        log.info(
            "AI: waypoint (%d,%d) for (%d,%d) is outside viewport (%d,%d)-(%d,%d)",
            wx,
            wy,
            tx,
            ty,
            left,
            top,
            right,
            bottom,
        )
        return None
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    if wx == sx and wy == sy:
        log.info("AI: waypoint is self position, skipping")
        return None
    if is_move_target_failed(wx, wy, ctx.timestamp_ms):
        log.info("AI: waypoint (%d,%d) recently failed, skipping", wx, wy)
        return None
    if _is_occupied_by_enemy(ctx, wx, wy):
        log.info("AI: waypoint (%d,%d) is occupied by enemy", wx, wy)
        return None
    log.info("AI: walking toward (%d,%d) via (%d,%d)", tx, ty, wx, wy)
    return make_move_command(wx, wy)


def _teleport_fallback_command(
    terrain: TerrainMapProtocol,
    sx: int,
    sy: int,
    tx: int,
    ty: int,
) -> BotCommand | None:
    """Return a teleport command for a terrain-blocked target when possible."""
    landing = find_teleport_landing_tile(terrain, sx, sy, tx, ty)
    if landing is None:
        log.info("AI: blocked target at (%d,%d) has no passable landing tile", tx, ty)
        return None
    lx, ly = landing
    log.info("AI: terrain blocked, teleporting near target to (%d,%d)", lx, ly)
    return make_teleport_command(lx, ly)


def _viewport_edge_target(ctx: _DecideCtx) -> tuple[int, int]:
    """Pick a viewport-edge tile to walk to for a fresh radar scan.

    Uses the actual viewport bounds rather than assuming the player is
    centered.  The player moves freely inside the fixed viewport frame;
    it only recenters when the player reaches the edge. Enemy-search
    scouting should target the inner actionable boundary, not the outer
    radar fringe, so the move command stays inside the 16x16 action area.

    Clamps to [1, 254] to stay inside the 256x256 map.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    viewport = ctx.world["viewport"]
    left = viewport["left"] + 1
    top = viewport["top"] + 1
    right = viewport["left"] + viewport["width"] - 2
    bottom = viewport["top"] + viewport["height"] - 2

    # Walk toward the viewport edge that is closer to map center.
    nx = right if sx < 128 else left
    ny = bottom if sy < 128 else top

    return (max(1, min(254, nx)), max(1, min(254, ny)))


__all__ = [
    "decide",
]
