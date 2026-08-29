"""Shared decision context and helpers for AI strategy modules.

The ``DecideCtx`` class holds all inputs for a single ``decide()`` call.
Strategy sub-modules (fuel, equipment, combat, movement) import from here
rather than from ``ai_strategy`` to avoid circular dependencies.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.intent import validate_collect_plan
from tankpit_bot.bot.ai.scoring_types import (
    BehaviorMode,
    ReasonKind,
    make_behavior_score,
)
from tankpit_bot.bot.ai.tactics import compute_desired_equipment
from tankpit_bot.bot.ai.types import (
    AIConfigDict,
    AIStateDict,
)
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import BotCommand
from tankpit_bot.inventory import InventoryState
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.scan_coverage import viewport_uncovered_count
from tankpit_bot.state.types import ContainerStateDict, SelfStateDict, WorldStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds


class DecideCtx:
    """Immutable context for one decide() call.

    Holds all inputs plus pre-computed values (equipment, filtered world,
    base AI state) so individual decision steps don't repeat work.

    Attributes:
        ws: The session's world service, for the live bookkeeping the
            ``world`` snapshot does not carry.
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
        "map_data_ingested_ms",
        "map_fuel_dots",
        "mode",
        "mode_started_ms",
        "mode_state",
        "self_state",
        "terrain",
        "timestamp_ms",
        "world",
        "ws",
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
        *,
        ws: WorldService,
    ) -> None:
        self.world = world
        self.self_state = self_state
        self.ai_state = ai_state
        self.inventory = inventory
        self.timestamp_ms = timestamp_ms
        self.terrain = terrain
        self.combat_feedback = combat_feedback
        # 0x4C map-answer snapshot, read HERE as a pair so every
        # decision step judges a consistent (stamp, dots) view. The
        # sniffer thread ingests concurrently; run bot-20260828-182401
        # quit out_of_fuel in the very second the 471-dot answer
        # landed, because the exit read the dots from one moment and
        # the ingestion stamp from another. Stamp FIRST, dots second:
        # that read order bounds the tear -- a stale stamp can only
        # under-report "answered", which defers one extra tick, never
        # exits on an empty atlas whose answer already arrived.
        self.map_data_ingested_ms = ws.map_data_ingested_ms
        self.map_fuel_dots = ws.map_fuel_dots
        # The session's world service. The planner is otherwise pure over
        # the ``world`` snapshot, but several decision steps genuinely
        # need live session bookkeeping the snapshot does not carry --
        # failed move targets, the incoming-damage rate window, movement
        # rejections. They used to reach a module global for it; carrying
        # it here makes the dependency visible and lets two sessions plan
        # independently ([[session-state-deglobalisation]] step 8).
        self.ws = ws

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
        self.base = validate_collect_plan(self.base, self.filtered)

    def derive(self, ai_state: AIStateDict) -> DecideCtx:
        """Clone this ctx with a rewritten AI state.

        The map-answer snapshot pair (``map_data_ingested_ms``,
        ``map_fuel_dots``) is COPIED from the parent, never re-read
        from ``ws``: the sniffer ingests concurrently, and a derived
        ctx mid-decision must judge the same world view as the tick
        that spawned it (the exit-races-answer TOCTOU, run
        bot-20260828-182401).

        Args:
            ai_state: Replacement AI state for the derived ctx.

        Returns:
            A ctx identical to this one except for the AI state and
            its derived fields.
        """
        derived = DecideCtx(
            self.world,
            self.self_state,
            ai_state,
            self.inventory,
            self.timestamp_ms,
            self.terrain,
            self.combat_feedback,
            ws=self.ws,
        )
        derived.map_data_ingested_ms = self.map_data_ingested_ms
        derived.map_fuel_dots = self.map_fuel_dots
        return derived


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
        ``map_intel_horizon_ms`` of the current tick.
    """
    return ctx.timestamp_ms - target["timestamp_ms"] < ctx.config["map_intel_horizon_ms"]


RADAR_SPEND_REVEAL_FLOOR_TILES = 32
"""Minimum uncovered viewport tiles that justify spending an extra radar.

The single radar-economics rule every discretionary radar site
consults ([[flag-triage-20260729]] s9-2/4/5, user 2026-07-30: "im
worried ... the viewport freshness handling is not properly wired to
the collecting system"). With extras stocked every scan CONSUMES an
item, and session 9 spent them on slivers: a displaced-landing rescan
of a fully-scanned viewport, a desync rescan of ground radared
seconds earlier, and a forage radar for a handful of tiles the tank
then hopped away from. 32 tiles is an eighth of the 256-tile
viewport — below that the reveal does not buy an item; the free
built-in radar (extras=0) stays gated only on "any uncovered tile"
because it costs nothing but the tick.
"""

RADAR_RESERVE_EXTRAS = 1
"""Extra-radar count treated as the reserve (user ruling 2026-07-31:
"if the bot runs out of radar ever ... its like dead in the water cuz
it takes so long to restock via free radar"). At or below this count
the spend bar escalates to :data:`RADAR_RESERVE_REVEAL_FLOOR_TILES`.
This is spend-gating inside the existing economics rule, NOT the
extras-toggle rationing rejected 2026-06-12 ([[radar-mechanics]]) --
the extras slot stays enabled and any scan that does fire uses the
extra."""

RADAR_RESERVE_REVEAL_FLOOR_TILES = 128
"""Uncovered-tile bar for spending the LAST extra radar: half the
256-tile viewport. The final paid sweep goes only to a near-full-value
reveal, never dribbles away on a sliver -- once it is gone, discovery
collapses to the built-in radius-2 scan and restock stalls
([[radar-mechanics]] "Death spiral at 0 extras")."""


def radar_spend_worthwhile(ctx: DecideCtx) -> bool:
    """Return True when a radar dispatch is worth its cost right now.

    Args:
        ctx: Decision context (coverage map + inventory).

    Returns:
        With extras above the reserve: True when the current viewport
        has at least :data:`RADAR_SPEND_REVEAL_FLOOR_TILES` uncovered
        tiles. At the reserve (the last extra): True only from
        :data:`RADAR_RESERVE_REVEAL_FLOOR_TILES` uncovered tiles.
        Without extras: True when any tile is uncovered (the built-in
        radar is free).
    """
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    uncovered = viewport_uncovered_count(
        ctx.world["scanned_tiles"],
        left,
        top,
        right,
        bottom,
        ctx.timestamp_ms,
    )
    extras = ctx.inventory["extra_radars"]["count"]
    if extras > RADAR_RESERVE_EXTRAS:
        return uncovered >= RADAR_SPEND_REVEAL_FLOOR_TILES
    if extras > 0:
        return uncovered >= RADAR_RESERVE_REVEAL_FLOOR_TILES
    return uncovered > 0


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
    "compute_equipment",
    "expire_kills",
    "filter_killed_tanks",
    "local_actionable_bounds",
    "locked_resource_target",
    "make_decision",
    "require_command",
    "target_position_is_fresh",
    "teleport_fuel_cost_to",
]
