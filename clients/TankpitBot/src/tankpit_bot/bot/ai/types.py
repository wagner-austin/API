"""Core TypedDicts, factory functions, and encode/decode for the AI system.

All types are immutable TypedDicts with factory functions for construction,
encode functions for JSON serialization, and decode functions with require_*
validation for deserialization.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

from tankpit_bot.bot.ai.modes import (
    AIMode,
    AIModeState,
)

# =============================================================================
# Behavior Mode
# =============================================================================

BehaviorMode = Literal[
    "HUNT",
    "COLLECT",
]

BEHAVIOR_MODES: tuple[BehaviorMode, ...] = (
    "HUNT",
    "COLLECT",
)


# =============================================================================
# BehaviorScoreDict
# =============================================================================


class BehaviorScoreDict(TypedDict):
    """A scored candidate behavior with target coordinates.

    Attributes:
        mode: Which behavior this score represents.
        score: Priority score (0-1000). Higher wins.
        target_x: Target X coordinate for this behavior.
        target_y: Target Y coordinate for this behavior.
        target_id: Tank ID of the combat target (0 if no specific target).
        reason: Human-readable reason for debugging.
    """

    mode: BehaviorMode
    score: int
    target_x: int
    target_y: int
    target_id: int
    reason: str


def make_behavior_score(
    mode: BehaviorMode,
    score: int,
    target_x: int,
    target_y: int,
    reason: str,
    target_id: int = 0,
) -> BehaviorScoreDict:
    """Create a BehaviorScoreDict.

    Args:
        mode: Behavior mode.
        score: Priority score (0-1000).
        target_x: Target X coordinate.
        target_y: Target Y coordinate.
        reason: Human-readable reason.
        target_id: Tank ID of combat target (0 if no specific target).

    Returns:
        BehaviorScoreDict with the provided values.
    """
    return BehaviorScoreDict(
        mode=mode,
        score=score,
        target_x=target_x,
        target_y=target_y,
        target_id=target_id,
        reason=reason,
    )


# =============================================================================
# EnemyThreatDict
# =============================================================================


class EnemyThreatDict(TypedDict):
    """An analyzed enemy tank with computed distance.

    Carries the three freshness timestamps from
    :class:`tankpit_bot.state.types.tank.TankStateDict` so the
    combat-strategy layer can read them without re-querying the
    registry.

    Attributes:
        tank_id: Enemy tank id.
        x: Enemy X coordinate.
        y: Enemy Y coordinate.
        distance: Manhattan distance from self.
        damage_state: Health state (0=full, 1=light, 2=medium, 3=critical).
        rank: Military rank (0-7). Lower rank = weaker.
        team: Enemy team id (0-3).
        name: Enemy player name.
        is_bot: Whether this enemy is a bot.
        timestamp_ms: When this tank was last confirmed by ANY source.
            Drives acquisition freshness.
        last_wire_seen_ms: When a wire-presence source last vouched the
            tank is in view. Drives the ghost gate.
        last_position_update_ms: When a wire-sourced observation last
            carried fresh ``(x, y)``. Drives the kill-shot gate.
        last_aim_x: Wire-reported barrel-aim X from this enemy's most
            recent 0x53 ShootEvent, or ``-1`` when never seen firing.
            Combat consumers (avoid-fire, predicted-LOS) read this to
            reason about which tile the enemy may target next.
        last_aim_y: Wire-reported barrel-aim Y from the same event.
        last_aim_weapon: Weapon byte from the same event
            (0=single, 1=dual, 2=missile, 3=homing). ``-1`` when never
            seen firing.
        last_aim_ms: Wall-clock of the most recent 0x53 ShootEvent
            attributed to this enemy. Consumers should age the aim
            with their own staleness threshold.
    """

    tank_id: int
    x: int
    y: int
    distance: int
    damage_state: int
    rank: int
    team: int
    name: str
    is_bot: bool
    timestamp_ms: int
    last_wire_seen_ms: int
    last_position_update_ms: int
    last_aim_x: int
    last_aim_y: int
    last_aim_weapon: int
    last_aim_ms: int


def make_enemy_threat(
    tank_id: int,
    x: int,
    y: int,
    distance: int,
    damage_state: int,
    rank: int,
    team: int,
    name: str,
    is_bot: bool,
    timestamp_ms: int = 0,
    last_wire_seen_ms: int = 0,
    last_position_update_ms: int = 0,
    last_aim_x: int = -1,
    last_aim_y: int = -1,
    last_aim_weapon: int = -1,
    last_aim_ms: int = 0,
) -> EnemyThreatDict:
    """Create an EnemyThreatDict.

    Args:
        tank_id: Enemy tank id.
        x: Enemy X coordinate.
        y: Enemy Y coordinate.
        distance: Manhattan distance from self.
        damage_state: Health state (0-3).
        rank: Military rank (0-7).
        team: Team id (0-3).
        name: Player name.
        is_bot: Whether this is a bot.
        timestamp_ms: When this tank was last confirmed by any source.
        last_wire_seen_ms: When a wire-presence source last vouched the
            tank is in view. Zero means never wire-confirmed.
        last_position_update_ms: When a wire-sourced observation last
            carried fresh ``(x, y)``. Zero means position has never
            been wire-confirmed.
        last_aim_x: Wire-reported barrel-aim X from the enemy's most
            recent 0x53 ShootEvent. Defaults to ``-1`` (never seen).
        last_aim_y: Wire-reported barrel-aim Y from the same event.
        last_aim_weapon: Weapon byte (0=single, 1=dual, 2=missile,
            3=homing) from the same event. ``-1`` when never seen.
        last_aim_ms: Wall-clock of the most recent 0x53 event for
            this enemy. ``0`` when never seen.

    Returns:
        EnemyThreatDict with the provided values.
    """
    return EnemyThreatDict(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=distance,
        damage_state=damage_state,
        rank=rank,
        team=team,
        name=name,
        is_bot=is_bot,
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=last_wire_seen_ms,
        last_position_update_ms=last_position_update_ms,
        last_aim_x=last_aim_x,
        last_aim_y=last_aim_y,
        last_aim_weapon=last_aim_weapon,
        last_aim_ms=last_aim_ms,
    )


# =============================================================================
# PathStepDict
# =============================================================================


class PathStepDict(TypedDict):
    """A single step in a computed path.

    Attributes:
        x: X coordinate of this step.
        y: Y coordinate of this step.
    """

    x: int
    y: int


def make_path_step(x: int, y: int) -> PathStepDict:
    """Create a PathStepDict.

    Args:
        x: X coordinate.
        y: Y coordinate.

    Returns:
        PathStepDict with the provided values.
    """
    return PathStepDict(x=x, y=y)


# =============================================================================
# AIConfigDict
# =============================================================================


class AIConfigDict(TypedDict):
    """Tunable AI parameters.

    Attributes:
        fuel_low_threshold: Below this the bot enters COLLECT. Also
            the reserve a combat teleport must leave behind -- engaging
            below it would flip priority to COLLECT the next tick.
            (The historical ``fuel_critical_threshold`` was collapsed
            into this single value 2026-06-22; the two-tier "polite low
            vs. emergency critical" distinction was dead because both
            thresholds had drifted to the same number.)
        fuel_full_threshold: Above this level, fuel collection score drops to zero.
        hunt_min_fuel: Operating reserve for search/recovery teleport hops.
        combat_range: Maximum Manhattan distance to engage an enemy.
        scan_cooldown_ms: Minimum milliseconds between radar scans.
        shot_feedback_timeout_ms: Milliseconds to wait before treating a shot as a miss.
        action_stall_timeout_ms: Milliseconds to wait before abandoning a stuck move/pickup.
        kill_cooldown_ms: Milliseconds to ignore a killed tank (avoid targeting corpse).
        map_open_cooldown_ms: Minimum milliseconds between map open commands.
        patrol_waypoints: Circuit of waypoints for PATROL behavior.
        dual_break_threshold: Emergency restock threshold for combat
            reserves. Applies to dual shots and homing shots only;
            extra radar has its own thresholds (radars are a search
            resource whose recovery SPENDS radars, so they were split
            out after the live run 20260611-232301 death spiral).
        dual_resume_threshold: Minimum healthy weapon reserve to leave
            emergency restock. Applies to dual and homing shots only.
        radar_break_threshold: Extra-radar count at or below which the
            bot enters equipment restock to rebuild radars before
            hunting. The grid-sweep forager handles the zero case.
        radar_resume_threshold: Extra-radar count to rebuild to before
            leaving restock and returning to the hunt. Radars find
            enemies and equipment, so a healthy buffer is rebuilt
            first; below it the bot restocks instead of fighting.
        equip_search_hop_distance: Teleport hop distance for resource
            search (equipment AND fuel). Set to one viewport width
            (16) so each hop lands in an adjacent, previously-
            unscanned viewport with no gap between scans. Larger
            strides leave unscanned strips between hops and burn
            disproportionately more fuel (teleport cost scales as
            6 * euclidean distance). Combat teleports use the
            target's actual coordinates via ``combat_landing_tile``
            and are NOT affected by this field.
        engagement_fuel_budget: Estimated fuel a typical kill consumes
            once adjacent (shot sequence + per-tick position cost; the
            approach teleport is priced separately per candidate in
            ``find_acquisition_target``). Starting a new engagement
            requires ``fuel >= fuel_low_threshold +
            engagement_fuel_budget``, and acquiring a map-known enemy
            additionally requires the candidate's exact teleport cost
            on top -- the bot never picks a fight it cannot pay for
            (user contract 2026-07-02). Mid-engagement
            (``combat_target_id != -1``) bypasses the mode-level gate
            -- a kill in progress is finished even on a tight budget.
            Recalibrated 2026-07-02 from wire data: shots cost ~45
            fuel plus ~10/tick position drain (live runs 2026-07-01)
            and practice-room kills take ~8-10 hits (recorded human
            sessions), so a realistic kill costs ~450. The earlier
            value (200) let the bot start fights it could not finish:
            run 2026-07-01 20:45 burned 505 fuel on the approach and
            hit the fuel-low interrupt 8 shots into the kill.
    """

    fuel_low_threshold: int
    fuel_full_threshold: int
    hunt_min_fuel: int
    combat_range: int
    scan_cooldown_ms: int
    shot_feedback_timeout_ms: int
    action_stall_timeout_ms: int
    kill_cooldown_ms: int
    map_open_cooldown_ms: int
    patrol_waypoints: list[tuple[int, int]]
    dual_break_threshold: int
    dual_resume_threshold: int
    radar_break_threshold: int
    radar_resume_threshold: int
    equip_search_hop_distance: int
    engagement_fuel_budget: int


def make_default_ai_config() -> AIConfigDict:
    """Create AIConfigDict with sensible defaults.

    Returns:
        AIConfigDict with default values suitable for lieutenant rank.
    """
    return AIConfigDict(
        fuel_low_threshold=200,
        fuel_full_threshold=1100,
        hunt_min_fuel=100,
        combat_range=20,
        scan_cooldown_ms=5000,
        shot_feedback_timeout_ms=4000,
        action_stall_timeout_ms=10000,
        kill_cooldown_ms=30000,
        map_open_cooldown_ms=5000,
        patrol_waypoints=[(64, 64), (192, 64), (192, 192), (64, 192)],
        dual_break_threshold=4,
        dual_resume_threshold=25,
        radar_break_threshold=5,
        radar_resume_threshold=20,
        equip_search_hop_distance=16,
        engagement_fuel_budget=450,
    )


class AIStateDict(TypedDict):
    """Mutable AI tick state tracking current behavior and cooldowns.

    Attributes:
        config: Tunable AI parameters.
        mode: Durable top-level AI mode owner.
        mode_state: Durable substate within the active top-level mode.
        mode_started_ms: Timestamp when the current durable mode was entered.
        last_scan_ms: Timestamp of last radar scan (milliseconds).
        last_shoot_ms: Timestamp of last shot fired (milliseconds).
        last_map_open_ms: Timestamp of last map open command (milliseconds).
        combat_target_id: Tank ID of current combat target (-1 if none).
        combat_target_x: X coordinate of combat target.
        combat_target_y: Y coordinate of combat target.
        killed_tank_ids: Tank IDs on kill cooldown {str(tank_id): timestamp_ms}.
        blocked_combat_targets: Tank IDs that are temporarily unengageable
            (e.g. no passable landing tile). {str(tank_id): timestamp_ms}.
            Expired by the same TTL as killed_tank_ids.
        last_shot_target_id: Tank ID we shot at last tick (-1 if none).
        last_shot_target_name: Name of tank we shot at last tick.
        resource_target_kind: Locked resource target kind ("", "fuel", or
            "equipment"). Used to continue an in-progress pickup plan across
            teleports and viewport recentering.
        resource_target_x: X coordinate of the locked resource target.
        resource_target_y: Y coordinate of the locked resource target.
        attempted_equipment_targets: Equipment targets that have been
            teleport-approached. {``"x,y"``: timestamp_ms}. Prevents
            repeated orbits around the same container.
    """

    config: AIConfigDict
    mode: AIMode
    mode_state: AIModeState
    mode_started_ms: int
    last_scan_ms: int
    last_shoot_ms: int
    last_map_open_ms: int
    combat_target_id: int
    combat_target_x: int
    combat_target_y: int
    killed_tank_ids: dict[str, int]
    session_kill_count: int
    session_hit_count: int
    session_miss_count: int
    blocked_combat_targets: dict[str, int]
    last_shot_target_id: int
    last_shot_target_name: str
    resource_target_kind: str
    resource_target_x: int
    resource_target_y: int
    attempted_equipment_targets: dict[str, int]


def make_initial_ai_state(
    config: AIConfigDict | None = None,
) -> AIStateDict:
    """Create initial AI state.

    Args:
        config: Optional AI config. Uses defaults if None.

    Returns:
        AIStateDict with initial values.
    """
    return AIStateDict(
        config=config if config is not None else make_default_ai_config(),
        mode="UNSET",
        mode_state="",
        mode_started_ms=0,
        last_scan_ms=1,  # Non-zero so radar doesn't auto-fire on first tick
        last_shoot_ms=0,
        last_map_open_ms=0,
        combat_target_id=-1,
        combat_target_x=0,
        combat_target_y=0,
        killed_tank_ids={},
        session_kill_count=0,
        session_hit_count=0,
        session_miss_count=0,
        blocked_combat_targets={},
        last_shot_target_id=-1,
        last_shot_target_name="",
        resource_target_kind="",
        resource_target_x=0,
        resource_target_y=0,
        attempted_equipment_targets={},
    )


__all__ = [
    "BEHAVIOR_MODES",
    "AIConfigDict",
    "AIStateDict",
    "BehaviorMode",
    "BehaviorScoreDict",
    "EnemyThreatDict",
    "PathStepDict",
    "make_behavior_score",
    "make_default_ai_config",
    "make_enemy_threat",
    "make_initial_ai_state",
    "make_path_step",
]
