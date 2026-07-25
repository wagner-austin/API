"""Shared decision context and helpers for AI strategy modules.

The ``DecideCtx`` class holds all inputs for a single ``decide()`` call.
Strategy sub-modules (fuel, equipment, combat, movement) import from here
rather than from ``ai_strategy`` to avoid circular dependencies.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.tactics import compute_desired_equipment
from tankpit_bot.bot.ai.types import (
    AIConfigDict,
    AIStateDict,
    BehaviorMode,
    EnemyThreatDict,
    ReasonKind,
    make_behavior_score,
)
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import BotCommand
from tankpit_bot.inventory import InventoryState
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.state.types import ContainerStateDict, SelfStateDict, WorldStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds


class DecideCtx:
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
        "map_fuel_dots",
        "mode",
        "mode_started_ms",
        "mode_state",
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
        map_fuel_dots: tuple[tuple[int, int], ...] = (),
    ) -> None:
        self.world = world
        self.self_state = self_state
        self.ai_state = ai_state
        self.inventory = inventory
        self.timestamp_ms = timestamp_ms
        self.terrain = terrain
        self.combat_feedback = combat_feedback
        # 0x4C fuel-dot atlas (session-constant like ``terrain``):
        # empty until the first map open of the session.
        self.map_fuel_dots = map_fuel_dots

        self.config: AIConfigDict = ai_state["config"]
        self.mode = ai_state["mode"]
        self.mode_state = ai_state["mode_state"]
        self.mode_started_ms = ai_state["mode_started_ms"]
        self.fuel: int = self_state["fuel"]
        self.equip: list[int] = compute_equipment(self.fuel, inventory)

        self.killed: dict[str, int] = expire_kills(
            ai_state["killed_tank_ids"],
            timestamp_ms,
            self.config["kill_cooldown_ms"],
        )
        self.blocked_targets: dict[str, int] = expire_kills(
            ai_state["blocked_combat_targets"],
            timestamp_ms,
            self.config["kill_cooldown_ms"],
        )
        self.filtered: WorldStateDict = filter_killed_tanks(world, self.killed)
        self.base: AIStateDict = AIStateDict(
            **{
                **ai_state,
                "killed_tank_ids": self.killed,
                "blocked_combat_targets": self.blocked_targets,
                "last_shot_target_id": -1,
                "last_shot_target_name": "",
            },
        )
        self.base = normalize_resource_target(self.base, self.filtered)


# =============================================================================
# Decision builder
# =============================================================================


def make_decision(
    command: BotCommand,
    mode: BehaviorMode,
    score: int,
    tx: int,
    ty: int,
    reason_kind: ReasonKind,
    ai_state: AIStateDict,
    equip: list[int],
    *,
    reason_context: dict[str, str | int] | None = None,
    secondary_command: BotCommand | None = None,
) -> TickDecisionDict:
    """Build a TickDecisionDict with less boilerplate.

    Args:
        command: Bot command to execute this tick.
        mode: Behavior mode label.
        score: Priority score.
        tx: Target X coordinate.
        ty: Target Y coordinate.
        reason_kind: Typed decision reason.
        ai_state: Updated AI state for next tick.
        equip: Desired equipment slot list.
        reason_context: Reason-specific scalar payload.
        secondary_command: Optional secondary command for multi-command ticks.

    Returns:
        Complete tick decision.
    """
    behavior = make_behavior_score(mode, score, tx, ty, reason_kind, reason_context=reason_context)
    return make_tick_decision(
        command=command,
        behavior=behavior,
        updated_ai_state=ai_state,
        desired_equipment=equip,
        secondary_command=secondary_command,
    )


# =============================================================================
# Resource target helpers
# =============================================================================


def clear_resource_target(ai_state: AIStateDict) -> AIStateDict:
    """Return AI state with any locked resource target cleared.

    Args:
        ai_state: Current AI state.

    Returns:
        New AIStateDict with resource target fields zeroed.
    """
    return AIStateDict(
        **{
            **ai_state,
            "resource_target_kind": "",
            "resource_target_x": 0,
            "resource_target_y": 0,
        },
    )


def set_resource_target(
    ai_state: AIStateDict,
    kind: str,
    tx: int,
    ty: int,
) -> AIStateDict:
    """Return AI state with a locked resource target set.

    Args:
        ai_state: Current AI state.
        kind: Resource kind ("fuel" or "equipment").
        tx: Target X coordinate.
        ty: Target Y coordinate.

    Returns:
        New AIStateDict with the specified resource target locked.
    """
    return AIStateDict(
        **{
            **ai_state,
            "resource_target_kind": kind,
            "resource_target_x": tx,
            "resource_target_y": ty,
        },
    )


def normalize_resource_target(
    ai_state: AIStateDict,
    world: WorldStateDict,
) -> AIStateDict:
    """Drop locked resource targets that are no longer pursuable.

    Applies the SAME pursuability predicate as candidate selection
    (kind match, failed pickups). A drained or failed container clears
    the lock so the cascade re-derives from live registry truth.

    Args:
        ai_state: Current AI state with resource target fields.
        world: Current world state with container positions.

    Returns:
        AI state with a no-longer-pursuable target cleared, or unchanged.
    """
    from tankpit_bot.bot.ai.equipment import is_container_pursuable

    kind = ai_state["resource_target_kind"]
    if kind not in ("fuel", "equipment"):
        return clear_resource_target(ai_state)
    tx = ai_state["resource_target_x"]
    ty = ai_state["resource_target_y"]
    target = world["containers"].get(f"{tx},{ty}")
    if target is None:
        return clear_resource_target(ai_state)
    if not is_container_pursuable(target, want_fuel=kind == "fuel"):
        return clear_resource_target(ai_state)
    return ai_state


def locked_resource_target(
    ctx: DecideCtx,
    kind: str,
) -> tuple[AIStateDict, ContainerStateDict | None]:
    """Return the normalized locked resource target for a specific kind.

    Args:
        ctx: Decision context.
        kind: Resource kind ("fuel" or "equipment").

    Returns:
        Tuple of (base AI state, locked container or None).
    """
    base_state = ctx.base
    if base_state["resource_target_kind"] != kind:
        return (base_state, None)
    tx = base_state["resource_target_x"]
    ty = base_state["resource_target_y"]
    # ctx.base was normalized against this same filtered world at
    # construction, so a surviving lock kind guarantees the container
    # exists; a KeyError here means the normalization invariant broke.
    return (base_state, ctx.filtered["containers"][f"{tx},{ty}"])


# =============================================================================
# Equipment helpers
# =============================================================================


def compute_equipment(fuel: int, inventory: InventoryState) -> list[int]:
    """Compute desired equipment as sorted list.

    Args:
        fuel: Current fuel level.
        inventory: Current inventory state.

    Returns:
        Sorted list of equipment slot numbers to enable.
    """
    desired = compute_desired_equipment(
        "HUNT",
        fuel,
        dual_shots_count=inventory["dual_shots"]["count"],
        homing_shots_count=inventory["homing_shots"]["count"],
    )
    return sorted(desired)


# =============================================================================
# Kill and combat helpers
# =============================================================================


def expire_kills(killed: dict[str, int], now: int, cooldown_ms: int) -> dict[str, int]:
    """Remove expired entries from killed tank IDs.

    Args:
        killed: Tank ID to kill-timestamp mapping.
        now: Current timestamp in milliseconds.
        cooldown_ms: Kill suppression duration.

    Returns:
        Filtered mapping with only unexpired kills.
    """
    return {k: v for k, v in killed.items() if now - v < cooldown_ms}


def filter_killed_tanks(world: WorldStateDict, killed: dict[str, int]) -> WorldStateDict:
    """Remove stale killed tanks from world state.

    Args:
        world: Current world state.
        killed: Active kill suppression mapping.

    Returns:
        World state with killed tanks filtered out.
    """
    if not killed:
        return world
    filtered = {
        tank_id: tank
        for tank_id, tank in world["tanks"].items()
        if tank_id not in killed or tank.get("timestamp_ms", 0) > killed[tank_id]
    }
    if len(filtered) == len(world["tanks"]):
        return world
    return WorldStateDict(
        self_state=world["self_state"],
        tanks=filtered,
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=world["viewport"],
        scanned_tiles=world["scanned_tiles"],
        timestamp_ms=world["timestamp_ms"],
    )


# =============================================================================
# Viewport and scan helpers
# =============================================================================


def local_actionable_bounds(ctx: DecideCtx) -> tuple[int, int, int, int]:
    """Return the inclusive visible viewport bounds.

    Args:
        ctx: Decision context with current viewport.

    Returns:
        Tuple of inclusive ``(left, top, right, bottom)`` visible bounds.
    """
    return viewport_visible_bounds(ctx.world["viewport"])


def target_position_is_fresh(ctx: DecideCtx, target: EnemyThreatDict) -> bool:
    """Return True when the target's tracked position is still trustworthy.

    HUNT/ACQUIRE consults this before teleporting at a target. The
    question it answers is the only one that matters for teleport
    decisions: do we still know where this enemy is?

    The trust signal is ``target["timestamp_ms"]`` -- the wall-clock of
    the most recent observation by ANY source (wire OR map snapshot).
    Both sources carry an authoritative ``(x, y)``: wire-sourced
    messages (0x3D MovementResponse, 0x28 TankEntry, 0x47 Movement,
    viewport scan, radar) and MAP_DATA from CMD_MAP_OPEN. Using the
    wire-only ``last_position_update_ms`` here would lock out every
    target known only through the global map snapshot
    (``state/mutations.py`` deliberately does NOT advance that field
    on a non-wire observation), so the bot would never trust the
    intel it just opened the map to fetch and would re-open every
    tick. Live run 20260620-191622 showed exactly that: 22 map_opens
    in 2.5 minutes, the AI re-deciding ``find <name>`` immediately
    after every MAP_DATA arrived.

    A fresh position means we can teleport directly. A stale position
    means we should refresh via map_open before committing fuel to a
    teleport at coordinates the enemy may have left. The cooldown is
    shared with map-open spam control because both questions ("when is
    a map snapshot useful?" and "when is a single tank's position
    stale?") are governed by the same observation-cadence floor.

    Args:
        ctx: Decision context.
        target: Enemy threat under consideration.

    Returns:
        True when the target's most recent observation is within
        ``map_open_cooldown_ms`` of the current tick.
    """
    return ctx.timestamp_ms - target["timestamp_ms"] < ctx.config["map_open_cooldown_ms"]


def can_use_radar(ctx: DecideCtx) -> bool:
    """Return True -- radar is always available regardless of fuel.

    Radar dispatch can never kill the bot: the server accepts the
    command even at zero fuel (user-confirmed 2026-06-26). The
    wire-reported 10-fuel deduction after a radar is just a debit, not
    a precondition. Keeping radar available at fuel=0 is what lets a
    stranded bot still see what's around it instead of looping silently.

    Args:
        ctx: Decision context (unused; kept for signature compatibility
            with the rest of the ``can_*`` predicate family).

    Returns:
        Always True.
    """
    del ctx
    return True


def teleport_fuel_cost_to(ctx: DecideCtx, target_x: int, target_y: int) -> int:
    """Return the exact fuel cost to teleport from self to a destination.

    Args:
        ctx: Decision context.
        target_x: Destination X coordinate.
        target_y: Destination Y coordinate.

    Returns:
        Exact teleport fuel cost for the current self position.
    """
    return teleport_cost(
        ctx.self_state["x"],
        ctx.self_state["y"],
        target_x,
        target_y,
    )


def can_afford_teleport(
    ctx: DecideCtx,
    target_x: int,
    target_y: int,
    *,
    reserve_fuel: int = 0,
) -> bool:
    """Check if the bot has enough fuel for a specific teleport.

    Args:
        ctx: Decision context.
        target_x: Destination X coordinate.
        target_y: Destination Y coordinate.
        reserve_fuel: Minimum fuel that must remain after teleporting.

    Returns:
        True if current fuel covers the exact teleport cost plus reserve.
    """
    required_fuel = teleport_fuel_cost_to(ctx, target_x, target_y) + reserve_fuel
    return ctx.fuel >= required_fuel


def require_command(
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


__all__ = [
    "DecideCtx",
    "can_afford_teleport",
    "can_use_radar",
    "clear_resource_target",
    "compute_equipment",
    "expire_kills",
    "filter_killed_tanks",
    "local_actionable_bounds",
    "locked_resource_target",
    "make_decision",
    "normalize_resource_target",
    "require_command",
    "set_resource_target",
    "target_position_is_fresh",
    "teleport_fuel_cost_to",
]
