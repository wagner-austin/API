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

# =============================================================================
# Behavior Mode
# =============================================================================

BehaviorMode = Literal[
    "HUNT",
    "COLLECT_FUEL",
    "COLLECT_EQUIPMENT",
    "DEPOSIT_FUEL",
    "PATROL",
    "DEFEND",
]

BEHAVIOR_MODES: tuple[BehaviorMode, ...] = (
    "HUNT",
    "COLLECT_FUEL",
    "COLLECT_EQUIPMENT",
    "DEPOSIT_FUEL",
    "PATROL",
    "DEFEND",
)


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
        reason: Human-readable reason for debugging.
    """

    mode: BehaviorMode
    score: int
    target_x: int
    target_y: int
    reason: str


def make_behavior_score(
    mode: BehaviorMode,
    score: int,
    target_x: int,
    target_y: int,
    reason: str,
) -> BehaviorScoreDict:
    """Create a BehaviorScoreDict.

    Args:
        mode: Behavior mode.
        score: Priority score (0-1000).
        target_x: Target X coordinate.
        target_y: Target Y coordinate.
        reason: Human-readable reason.

    Returns:
        BehaviorScoreDict with the provided values.
    """
    return BehaviorScoreDict(
        mode=mode,
        score=score,
        target_x=target_x,
        target_y=target_y,
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
        fuel_full_threshold: Above this level, DEPOSIT_FUEL becomes eligible.
        hunt_min_fuel: Minimum fuel required to engage in combat.
        combat_range: Maximum Manhattan distance to engage an enemy.
        scan_cooldown_ms: Minimum milliseconds between radar scans.
        shoot_cooldown_ms: Minimum milliseconds between shots.
        patrol_waypoints: Circuit of waypoints for PATROL behavior.
    """

    fuel_critical_threshold: int
    fuel_low_threshold: int
    fuel_full_threshold: int
    hunt_min_fuel: int
    combat_range: int
    scan_cooldown_ms: int
    shoot_cooldown_ms: int
    patrol_waypoints: list[tuple[int, int]]


def make_default_ai_config() -> AIConfigDict:
    """Create AIConfigDict with sensible defaults.

    Returns:
        AIConfigDict with default values suitable for lieutenant rank.
    """
    return AIConfigDict(
        fuel_critical_threshold=200,
        fuel_low_threshold=500,
        fuel_full_threshold=1200,
        hunt_min_fuel=400,
        combat_range=20,
        scan_cooldown_ms=5000,
        shoot_cooldown_ms=2000,
        patrol_waypoints=[(64, 64), (192, 64), (192, 192), (64, 192)],
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
        "patrol_waypoints": waypoints,
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
        patrol_waypoints=_decode_patrol_waypoints(data),
    )


# =============================================================================
# AIStateDict
# =============================================================================


class AIStateDict(TypedDict):
    """Mutable AI tick state tracking current behavior and cooldowns.

    Attributes:
        config: Tunable AI parameters.
        active_mode: Currently active behavior mode.
        patrol_waypoint_index: Current index in patrol waypoint circuit.
        last_scan_ms: Timestamp of last radar scan (milliseconds).
        last_shoot_ms: Timestamp of last shot fired (milliseconds).
        combat_target_id: Tank ID of current combat target (-1 if none).
        combat_target_x: X coordinate of combat target.
        combat_target_y: Y coordinate of combat target.
        ticks_in_mode: How many ticks spent in current mode.
    """

    config: AIConfigDict
    active_mode: BehaviorMode
    patrol_waypoint_index: int
    last_scan_ms: int
    last_shoot_ms: int
    combat_target_id: int
    combat_target_x: int
    combat_target_y: int
    ticks_in_mode: int


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
        active_mode="PATROL",
        patrol_waypoint_index=0,
        last_scan_ms=0,
        last_shoot_ms=0,
        combat_target_id=-1,
        combat_target_x=0,
        combat_target_y=0,
        ticks_in_mode=0,
    )


def encode_ai_state(state: AIStateDict) -> JSONObject:
    """Encode AIStateDict to JSON-serializable dict.

    Args:
        state: AIStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "config": encode_ai_config(state["config"]),
        "active_mode": state["active_mode"],
        "patrol_waypoint_index": state["patrol_waypoint_index"],
        "last_scan_ms": state["last_scan_ms"],
        "last_shoot_ms": state["last_shoot_ms"],
        "combat_target_id": state["combat_target_id"],
        "combat_target_x": state["combat_target_x"],
        "combat_target_y": state["combat_target_y"],
        "ticks_in_mode": state["ticks_in_mode"],
    }


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
    return AIStateDict(
        config=decode_ai_config(config_raw),
        active_mode=_require_behavior_mode(data, "active_mode"),
        patrol_waypoint_index=require_int(data, "patrol_waypoint_index"),
        last_scan_ms=require_int(data, "last_scan_ms"),
        last_shoot_ms=require_int(data, "last_shoot_ms"),
        combat_target_id=require_int(data, "combat_target_id"),
        combat_target_x=require_int(data, "combat_target_x"),
        combat_target_y=require_int(data, "combat_target_y"),
        ticks_in_mode=require_int(data, "ticks_in_mode"),
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
