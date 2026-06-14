"""Core TypedDicts, factory functions, and encode/decode for the AI system.

All types are immutable TypedDicts with factory functions for construction,
encode functions for JSON serialization, and decode functions with require_*
validation for deserialization.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_bool,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot.bot.ai.modes import (
    AIMode,
    AIModeState,
    is_valid_ai_mode_state,
    require_ai_mode,
    require_ai_mode_state,
)

# =============================================================================
# Behavior Mode
# =============================================================================

BehaviorMode = Literal[
    "HUNT",
    "COLLECT_FUEL",
    "COLLECT_EQUIPMENT",
]

BEHAVIOR_MODES: tuple[BehaviorMode, ...] = (
    "HUNT",
    "COLLECT_FUEL",
    "COLLECT_EQUIPMENT",
)


# =============================================================================
# Combat Phase — explicit FSM replacing opaque ticks_in_mode counter
# =============================================================================


def _require_behavior_mode(data: JSONObject, key: str) -> BehaviorMode:
    """Validate and extract a BehaviorMode from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated BehaviorMode value.

    Raises:
        ValueError: If value is not a valid BehaviorMode.
    """
    raw = require_str(data, key)
    for mode in BEHAVIOR_MODES:
        if raw == mode:
            return mode
    raise ValueError(f"{key} must be one of {BEHAVIOR_MODES}, got {raw!r}")


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


def encode_behavior_score(score: BehaviorScoreDict) -> JSONObject:
    """Encode BehaviorScoreDict to JSON-serializable dict.

    Args:
        score: BehaviorScoreDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "mode": score["mode"],
        "score": score["score"],
        "target_x": score["target_x"],
        "target_y": score["target_y"],
        "target_id": score["target_id"],
        "reason": score["reason"],
    }


def decode_behavior_score(data: JSONObject) -> BehaviorScoreDict:
    """Decode BehaviorScoreDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated BehaviorScoreDict.

    Raises:
        ValueError: If mode is not a valid BehaviorMode.
        JSONTypeError: If required fields are missing or invalid.
    """
    return BehaviorScoreDict(
        mode=_require_behavior_mode(data, "mode"),
        score=require_int(data, "score"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        target_id=require_int(data, "target_id"),
        reason=require_str(data, "reason"),
    )


# =============================================================================
# EnemyThreatDict
# =============================================================================


class EnemyThreatDict(TypedDict):
    """An analyzed enemy tank with computed distance.

    Attributes:
        tank_id: Enemy tank ID.
        x: Enemy X coordinate.
        y: Enemy Y coordinate.
        distance: Manhattan distance from self.
        damage_state: Health state (0=full, 1=light, 2=medium, 3=critical).
        rank: Military rank (0-7). Lower rank = weaker.
        team: Enemy team ID (0-3).
        name: Enemy player name.
        is_bot: Whether this enemy is a bot.
        timestamp_ms: When this tank was last confirmed by the server.
            Used for freshness-based target selection.
        last_wire_seen_ms: When a wire-presence source last vouched this
            tank is actually in view (zero means never). Read by the
            kill-shot gate: a target the map keeps re-listing but that no
            wire source confirms is a ghost and must not be fired at.
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
) -> EnemyThreatDict:
    """Create an EnemyThreatDict.

    Args:
        tank_id: Enemy tank ID.
        x: Enemy X coordinate.
        y: Enemy Y coordinate.
        distance: Manhattan distance from self.
        damage_state: Health state (0-3).
        rank: Military rank (0-7).
        team: Team ID (0-3).
        name: Player name.
        is_bot: Whether this is a bot.
        timestamp_ms: When this tank was last confirmed by any source.
        last_wire_seen_ms: When a wire-presence source last vouched this
            tank is in view. Zero means never wire-confirmed.

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
    )


def encode_enemy_threat(threat: EnemyThreatDict) -> JSONObject:
    """Encode EnemyThreatDict to JSON-serializable dict.

    Args:
        threat: EnemyThreatDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "tank_id": threat["tank_id"],
        "x": threat["x"],
        "y": threat["y"],
        "distance": threat["distance"],
        "damage_state": threat["damage_state"],
        "rank": threat["rank"],
        "team": threat["team"],
        "name": threat["name"],
        "is_bot": threat["is_bot"],
        "timestamp_ms": threat["timestamp_ms"],
        "last_wire_seen_ms": threat["last_wire_seen_ms"],
    }


def decode_enemy_threat(data: JSONObject) -> EnemyThreatDict:
    """Decode EnemyThreatDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated EnemyThreatDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return EnemyThreatDict(
        tank_id=require_int(data, "tank_id"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        distance=require_int(data, "distance"),
        damage_state=require_int(data, "damage_state"),
        rank=require_int(data, "rank"),
        team=require_int(data, "team"),
        name=require_str(data, "name"),
        is_bot=require_bool(data, "is_bot"),
        timestamp_ms=require_int(data, "timestamp_ms"),
        last_wire_seen_ms=require_int(data, "last_wire_seen_ms"),
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


def encode_path_step(step: PathStepDict) -> JSONObject:
    """Encode PathStepDict to JSON-serializable dict.

    Args:
        step: PathStepDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {"x": step["x"], "y": step["y"]}


def decode_path_step(data: JSONObject) -> PathStepDict:
    """Decode PathStepDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated PathStepDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return PathStepDict(x=require_int(data, "x"), y=require_int(data, "y"))


# =============================================================================
# AIConfigDict
# =============================================================================


class AIConfigDict(TypedDict):
    """Tunable AI parameters.

    Attributes:
        fuel_critical_threshold: Below this, shields activate and fuel is emergency.
        fuel_low_threshold: Below this, fuel collection gets priority boost.
            Also the reserve a combat teleport must leave behind -- engaging
            below it would flip priority to COLLECT_FUEL the next tick.
        fuel_full_threshold: Above this level, fuel collection score drops to zero.
        hunt_min_fuel: Operating reserve for search/recovery teleport hops.
        combat_range: Maximum Manhattan distance to engage an enemy.
        scan_cooldown_ms: Minimum milliseconds between radar scans.
        shoot_cooldown_ms: Minimum milliseconds between shots.
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
        equip_search_hop_distance: Teleport hop distance for local equipment search.
        equip_search_max_failures: Maximum consecutive equipment-search hops.
    """

    fuel_critical_threshold: int
    fuel_low_threshold: int
    fuel_full_threshold: int
    hunt_min_fuel: int
    combat_range: int
    scan_cooldown_ms: int
    shoot_cooldown_ms: int
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
    equip_search_max_failures: int


def make_default_ai_config() -> AIConfigDict:
    """Create AIConfigDict with sensible defaults.

    Returns:
        AIConfigDict with default values suitable for lieutenant rank.
    """
    return AIConfigDict(
        fuel_critical_threshold=500,
        fuel_low_threshold=500,
        fuel_full_threshold=1100,
        hunt_min_fuel=100,
        combat_range=20,
        scan_cooldown_ms=5000,
        shoot_cooldown_ms=2000,
        shot_feedback_timeout_ms=4000,
        action_stall_timeout_ms=10000,
        kill_cooldown_ms=30000,
        map_open_cooldown_ms=5000,
        patrol_waypoints=[(64, 64), (192, 64), (192, 192), (64, 192)],
        dual_break_threshold=12,
        dual_resume_threshold=20,
        radar_break_threshold=5,
        radar_resume_threshold=15,
        equip_search_hop_distance=30,
        equip_search_max_failures=3,
    )


def encode_ai_config(config: AIConfigDict) -> JSONObject:
    """Encode AIConfigDict to JSON-serializable dict.

    Args:
        config: AIConfigDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    waypoints: list[JSONValue] = [[x, y] for x, y in config["patrol_waypoints"]]
    return {
        "fuel_critical_threshold": config["fuel_critical_threshold"],
        "fuel_low_threshold": config["fuel_low_threshold"],
        "fuel_full_threshold": config["fuel_full_threshold"],
        "hunt_min_fuel": config["hunt_min_fuel"],
        "combat_range": config["combat_range"],
        "scan_cooldown_ms": config["scan_cooldown_ms"],
        "shoot_cooldown_ms": config["shoot_cooldown_ms"],
        "shot_feedback_timeout_ms": config["shot_feedback_timeout_ms"],
        "action_stall_timeout_ms": config["action_stall_timeout_ms"],
        "kill_cooldown_ms": config["kill_cooldown_ms"],
        "map_open_cooldown_ms": config["map_open_cooldown_ms"],
        "patrol_waypoints": waypoints,
        "dual_break_threshold": config["dual_break_threshold"],
        "dual_resume_threshold": config["dual_resume_threshold"],
        "radar_break_threshold": config["radar_break_threshold"],
        "radar_resume_threshold": config["radar_resume_threshold"],
        "equip_search_hop_distance": config["equip_search_hop_distance"],
        "equip_search_max_failures": config["equip_search_max_failures"],
    }


def _decode_patrol_waypoints(data: JSONObject) -> list[tuple[int, int]]:
    """Decode patrol waypoints from JSON.

    Args:
        data: JSON object containing patrol_waypoints field.

    Returns:
        List of (x, y) waypoint tuples.

    Raises:
        ValueError: If waypoints format is invalid.
    """
    raw = data.get("patrol_waypoints")
    if not isinstance(raw, list):
        raise ValueError("patrol_waypoints must be a list")
    result: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Each waypoint must be [x, y], got {item!r}")
        x_val = item[0]
        y_val = item[1]
        if not isinstance(x_val, int) or not isinstance(y_val, int):
            raise ValueError(f"Waypoint coords must be int, got {item!r}")
        result.append((x_val, y_val))
    return result


def decode_ai_config(data: JSONObject) -> AIConfigDict:
    """Decode AIConfigDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated AIConfigDict.

    Raises:
        ValueError: If waypoints format is invalid.
        JSONTypeError: If required fields are missing or invalid.
    """
    return AIConfigDict(
        fuel_critical_threshold=require_int(data, "fuel_critical_threshold"),
        fuel_low_threshold=require_int(data, "fuel_low_threshold"),
        fuel_full_threshold=require_int(data, "fuel_full_threshold"),
        hunt_min_fuel=require_int(data, "hunt_min_fuel"),
        combat_range=require_int(data, "combat_range"),
        scan_cooldown_ms=require_int(data, "scan_cooldown_ms"),
        shoot_cooldown_ms=require_int(data, "shoot_cooldown_ms"),
        shot_feedback_timeout_ms=require_int(data, "shot_feedback_timeout_ms"),
        action_stall_timeout_ms=require_int(data, "action_stall_timeout_ms"),
        kill_cooldown_ms=require_int(data, "kill_cooldown_ms"),
        map_open_cooldown_ms=require_int(data, "map_open_cooldown_ms"),
        patrol_waypoints=_decode_patrol_waypoints(data),
        dual_break_threshold=require_int(data, "dual_break_threshold"),
        dual_resume_threshold=require_int(data, "dual_resume_threshold"),
        radar_break_threshold=require_int(data, "radar_break_threshold"),
        radar_resume_threshold=require_int(data, "radar_resume_threshold"),
        equip_search_hop_distance=require_int(data, "equip_search_hop_distance"),
        equip_search_max_failures=require_int(data, "equip_search_max_failures"),
    )


# =============================================================================
# AIStateDict
# =============================================================================


class AIStateDict(TypedDict):
    """Mutable AI tick state tracking current behavior and cooldowns.

    Attributes:
        config: Tunable AI parameters.
        mode: Durable top-level AI mode owner.
        mode_state: Durable substate within the active top-level mode.
        mode_started_ms: Timestamp when the current durable mode was entered.
        patrol_waypoint_index: Current index in patrol waypoint circuit.
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
        local_scan_cells: Built-in radar coverage grid keyed by ``"cx,cy"``
            cell index, values are scan timestamps. Used by the equipment
            foraging sweep.
        attempted_equipment_targets: Equipment targets that have been
            teleport-approached. {``"x,y"``: timestamp_ms}. Prevents
            repeated orbits around the same container.
        attempted_fuel_dots: Fuel dots that have been approached or
            teleported to. {``"x,y"``: timestamp_ms}. Prevents revisiting
            dots within the scan coverage TTL.
    """

    config: AIConfigDict
    mode: AIMode
    mode_state: AIModeState
    mode_started_ms: int
    patrol_waypoint_index: int
    last_scan_ms: int
    last_shoot_ms: int
    last_map_open_ms: int
    combat_target_id: int
    combat_target_x: int
    combat_target_y: int
    killed_tank_ids: dict[str, int]
    blocked_combat_targets: dict[str, int]
    last_shot_target_id: int
    last_shot_target_name: str
    equipment_search_failures: int
    resource_target_kind: str
    resource_target_x: int
    resource_target_y: int
    local_scan_cells: dict[str, int]
    attempted_equipment_targets: dict[str, int]
    attempted_fuel_dots: dict[str, int]


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
        patrol_waypoint_index=0,
        last_scan_ms=1,  # Non-zero so radar doesn't auto-fire on first tick
        last_shoot_ms=0,
        last_map_open_ms=0,
        combat_target_id=-1,
        combat_target_x=0,
        combat_target_y=0,
        killed_tank_ids={},
        blocked_combat_targets={},
        last_shot_target_id=-1,
        last_shot_target_name="",
        equipment_search_failures=0,
        resource_target_kind="",
        resource_target_x=0,
        resource_target_y=0,
        local_scan_cells={},
        attempted_equipment_targets={},
        attempted_fuel_dots={},
    )


def encode_ai_state(state: AIStateDict) -> JSONObject:
    """Encode AIStateDict to JSON-serializable dict.

    Args:
        state: AIStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    killed: JSONValue = dict(state["killed_tank_ids"])
    return {
        "config": encode_ai_config(state["config"]),
        "mode": state["mode"],
        "mode_state": state["mode_state"],
        "mode_started_ms": state["mode_started_ms"],
        "patrol_waypoint_index": state["patrol_waypoint_index"],
        "last_scan_ms": state["last_scan_ms"],
        "last_shoot_ms": state["last_shoot_ms"],
        "last_map_open_ms": state["last_map_open_ms"],
        "combat_target_id": state["combat_target_id"],
        "combat_target_x": state["combat_target_x"],
        "combat_target_y": state["combat_target_y"],
        "killed_tank_ids": killed,
        "blocked_combat_targets": dict(state["blocked_combat_targets"]),
        "last_shot_target_id": state["last_shot_target_id"],
        "last_shot_target_name": state["last_shot_target_name"],
        "equipment_search_failures": state["equipment_search_failures"],
        "resource_target_kind": state["resource_target_kind"],
        "resource_target_x": state["resource_target_x"],
        "resource_target_y": state["resource_target_y"],
        "local_scan_cells": dict(state["local_scan_cells"]),
        "attempted_equipment_targets": dict(state["attempted_equipment_targets"]),
        "attempted_fuel_dots": dict(state["attempted_fuel_dots"]),
    }


def _decode_killed_tank_ids(data: JSONObject) -> dict[str, int]:
    """Decode killed_tank_ids from JSON.

    Args:
        data: JSON object containing killed_tank_ids field.

    Returns:
        Dict mapping str(tank_id) to timestamp_ms.

    Raises:
        ValueError: If format is invalid.
    """
    return _require_str_int_mapping(data, "killed_tank_ids")


def _require_str_int_mapping(data: JSONObject, key: str) -> dict[str, int]:
    """Decode a dict[str, int] field from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Dict mapping string keys to int values.

    Raises:
        ValueError: If format is invalid.
    """
    raw = data.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be an object")
    result: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(v, int):
            raise ValueError(f"{key} values must be int, got {type(v).__name__}")
        result[k] = v
    return result


def decode_ai_state(data: JSONObject) -> AIStateDict:
    """Decode AIStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated AIStateDict.

    Raises:
        ValueError: If mode or config values are invalid.
        JSONTypeError: If required fields are missing or invalid.
    """
    config_raw = data.get("config")
    if not isinstance(config_raw, dict):
        raise ValueError("config must be an object")
    mode = require_ai_mode(data, "mode")
    mode_state = require_ai_mode_state(data, "mode_state")
    if not is_valid_ai_mode_state(mode, mode_state):
        raise ValueError(f"mode_state {mode_state!r} is invalid for mode {mode!r}")
    return AIStateDict(
        config=decode_ai_config(config_raw),
        mode=mode,
        mode_state=mode_state,
        mode_started_ms=require_int(data, "mode_started_ms"),
        patrol_waypoint_index=require_int(data, "patrol_waypoint_index"),
        last_scan_ms=require_int(data, "last_scan_ms"),
        last_shoot_ms=require_int(data, "last_shoot_ms"),
        last_map_open_ms=require_int(data, "last_map_open_ms"),
        combat_target_id=require_int(data, "combat_target_id"),
        combat_target_x=require_int(data, "combat_target_x"),
        combat_target_y=require_int(data, "combat_target_y"),
        killed_tank_ids=_decode_killed_tank_ids(data),
        blocked_combat_targets=_require_str_int_mapping(data, "blocked_combat_targets"),
        last_shot_target_id=require_int(data, "last_shot_target_id"),
        last_shot_target_name=require_str(data, "last_shot_target_name"),
        equipment_search_failures=require_int(data, "equipment_search_failures"),
        resource_target_kind=require_str(data, "resource_target_kind"),
        resource_target_x=require_int(data, "resource_target_x"),
        resource_target_y=require_int(data, "resource_target_y"),
        local_scan_cells=_require_str_int_mapping(data, "local_scan_cells")
        if "local_scan_cells" in data
        else {},
        attempted_equipment_targets=_require_str_int_mapping(data, "attempted_equipment_targets")
        if "attempted_equipment_targets" in data
        else {},
        attempted_fuel_dots=_require_str_int_mapping(data, "attempted_fuel_dots")
        if "attempted_fuel_dots" in data
        else {},
    )


__all__ = [
    "BEHAVIOR_MODES",
    "AIConfigDict",
    "AIStateDict",
    "BehaviorMode",
    "BehaviorScoreDict",
    "EnemyThreatDict",
    "PathStepDict",
    "decode_ai_config",
    "decode_ai_state",
    "decode_behavior_score",
    "decode_enemy_threat",
    "decode_path_step",
    "encode_ai_config",
    "encode_ai_state",
    "encode_behavior_score",
    "encode_enemy_threat",
    "encode_path_step",
    "make_behavior_score",
    "make_default_ai_config",
    "make_enemy_threat",
    "make_initial_ai_state",
    "make_path_step",
]
