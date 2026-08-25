"""Encode/decode functions for AI system TypedDicts.

Separated from types.py to keep type definitions under 400 lines.
Every encode function serializes a TypedDict to JSONObject.
Every decode function validates and deserializes from JSONObject.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_dict,
    require_int,
    require_str,
)

from tankpit_bot.bot.ai.scoring_types import (
    BEHAVIOR_MODES,
    REASON_KINDS,
    BehaviorMode,
    BehaviorScoreDict,
    ReasonKind,
)
from tankpit_bot.bot.ai.types import (
    AIConfigDict,
    AIStateDict,
)
from tankpit_bot.bot.ai.world_types import (
    EnemyThreatDict,
    PathStepDict,
)
from tankpit_bot.fleetshare.codecs import require_fleet_role
from tankpit_bot.types.modes import (
    AI_MODES,
    AIMode,
    is_valid_ai_mode_state,
    require_ai_mode,
    require_ai_mode_state,
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


# =========================================================================
# BehaviorScoreDict codecs
# =========================================================================


def _require_reason_kind(data: JSONObject, key: str) -> ReasonKind:
    """Validate and extract a reason kind from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated reason kind.

    Raises:
        JSONTypeError: If the value is not a supported reason kind.
    """
    raw = require_str(data, key)
    for kind in REASON_KINDS:
        if raw == kind:
            return kind
    raise JSONTypeError(f"{key} must be one of {REASON_KINDS}, got {raw!r}")


def _require_reason_context(data: JSONObject, key: str) -> dict[str, str | int]:
    """Validate and extract a reason context map from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated scalar context map.

    Raises:
        JSONTypeError: If the value is not a str/int-valued object.
    """
    raw = require_dict(data, key)
    context: dict[str, str | int] = {}
    for field, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise JSONTypeError(f"{key}[{field!r}] must be str or int")
        context[field] = value
    return context


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
        "reason_kind": score["reason_kind"],
        "reason_context": dict(score["reason_context"]),
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
        reason_kind=_require_reason_kind(data, "reason_kind"),
        reason_context=_require_reason_context(data, "reason_context"),
    )


# =========================================================================
# EnemyThreatDict codecs
# =========================================================================


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
        "last_position_update_ms": threat["last_position_update_ms"],
        "last_aim_x": threat["last_aim_x"],
        "last_aim_y": threat["last_aim_y"],
        "last_aim_weapon": threat["last_aim_weapon"],
        "last_aim_ms": threat["last_aim_ms"],
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
        last_position_update_ms=require_int(data, "last_position_update_ms"),
        last_aim_x=require_int(data, "last_aim_x"),
        last_aim_y=require_int(data, "last_aim_y"),
        last_aim_weapon=require_int(data, "last_aim_weapon"),
        last_aim_ms=require_int(data, "last_aim_ms"),
    )


# =========================================================================
# PathStepDict codecs
# =========================================================================


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


# =========================================================================
# AIConfigDict codecs
# =========================================================================


def encode_ai_config(config: AIConfigDict) -> JSONObject:
    """Encode AIConfigDict to JSON-serializable dict.

    Args:
        config: AIConfigDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    waypoints: list[JSONValue] = [[x, y] for x, y in config["patrol_waypoints"]]
    return {
        "fuel_low_threshold": config["fuel_low_threshold"],
        "hunt_min_fuel": config["hunt_min_fuel"],
        "combat_range": config["combat_range"],
        "scan_cooldown_ms": config["scan_cooldown_ms"],
        "shot_feedback_timeout_ms": config["shot_feedback_timeout_ms"],
        "action_stall_timeout_ms": config["action_stall_timeout_ms"],
        "kill_cooldown_ms": config["kill_cooldown_ms"],
        "map_open_cooldown_ms": config["map_open_cooldown_ms"],
        "patrol_waypoints": waypoints,
        "dual_break_threshold": config["dual_break_threshold"],
        "radar_break_threshold": config["radar_break_threshold"],
        "engagement_fuel_budget": config["engagement_fuel_budget"],
        "priority_target_name": config["priority_target_name"],
        "human_target_min_rank": config["human_target_min_rank"],
        "human_target_max_rank": config["human_target_max_rank"],
        "role": config["role"],
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
        fuel_low_threshold=require_int(data, "fuel_low_threshold"),
        hunt_min_fuel=require_int(data, "hunt_min_fuel"),
        combat_range=require_int(data, "combat_range"),
        scan_cooldown_ms=require_int(data, "scan_cooldown_ms"),
        shot_feedback_timeout_ms=require_int(data, "shot_feedback_timeout_ms"),
        action_stall_timeout_ms=require_int(data, "action_stall_timeout_ms"),
        kill_cooldown_ms=require_int(data, "kill_cooldown_ms"),
        map_open_cooldown_ms=require_int(data, "map_open_cooldown_ms"),
        patrol_waypoints=_decode_patrol_waypoints(data),
        dual_break_threshold=require_int(data, "dual_break_threshold"),
        radar_break_threshold=require_int(data, "radar_break_threshold"),
        engagement_fuel_budget=require_int(data, "engagement_fuel_budget"),
        priority_target_name=require_str(data, "priority_target_name"),
        human_target_min_rank=require_int(data, "human_target_min_rank"),
        human_target_max_rank=require_int(data, "human_target_max_rank"),
        role=require_fleet_role(data, "role"),
    )


# =========================================================================
# AIStateDict codecs
# =========================================================================


def encode_ai_state(state: AIStateDict) -> JSONObject:
    """Encode AIStateDict to JSON-serializable dict.

    Args:
        state: AIStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    killed: JSONValue = dict(state["killed_tank_ids"])
    manual_mode = state["manual_mode"]
    manual_value: JSONValue = manual_mode if manual_mode is not None else None
    return {
        "config": encode_ai_config(state["config"]),
        "mode": state["mode"],
        "mode_state": state["mode_state"],
        "mode_started_ms": state["mode_started_ms"],
        "last_scan_ms": state["last_scan_ms"],
        "last_shoot_ms": state["last_shoot_ms"],
        "last_map_open_ms": state["last_map_open_ms"],
        "combat_target_id": state["combat_target_id"],
        "wind_down": state["wind_down"],
        "break_escape_until_fuel": state["break_escape_until_fuel"],
        "combat_target_x": state["combat_target_x"],
        "combat_target_y": state["combat_target_y"],
        "killed_tank_ids": killed,
        "session_kill_count": state["session_kill_count"],
        "session_hit_count": state["session_hit_count"],
        "session_miss_count": state["session_miss_count"],
        "session_reject_count": state["session_reject_count"],
        "blocked_combat_targets": dict(state["blocked_combat_targets"]),
        "last_shot_target_id": state["last_shot_target_id"],
        "last_shot_target_name": state["last_shot_target_name"],
        "resource_target_kind": state["resource_target_kind"],
        "resource_target_x": state["resource_target_x"],
        "resource_target_y": state["resource_target_y"],
        "attempted_equipment_targets": dict(state["attempted_equipment_targets"]),
        "last_landing_scan_viewport": state["last_landing_scan_viewport"],
        "suppress_landing_scan": state["suppress_landing_scan"],
        "manual_mode": manual_value,
        "live_radars_used": state["live_radars_used"],
        "live_teleports": state["live_teleports"],
        "mine_clearance_aim_key": state["mine_clearance_aim_key"],
        "mine_clearance_shot_ms": state["mine_clearance_shot_ms"],
        "greeted_tank_ids": dict(state["greeted_tank_ids"]),
        "pursuit_shot_target_id": state["pursuit_shot_target_id"],
        "pursuit_shot_ms": state["pursuit_shot_ms"],
        "visited_tank_ids": dict(state["visited_tank_ids"]),
        "last_scope_scout_ms": state["last_scope_scout_ms"],
        "sweep_anchor_x": state["sweep_anchor_x"],
        "sweep_anchor_y": state["sweep_anchor_y"],
        "maroon_pan_x": state["maroon_pan_x"],
        "maroon_pan_y": state["maroon_pan_y"],
    }


def _decode_manual_mode(data: JSONObject) -> AIMode | None:
    """Decode the required ``manual_mode`` field from an encoded AI state.

    Args:
        data: JSON object being decoded into :class:`AIStateDict`.

    Returns:
        ``None`` when the encoded value is explicit JSON ``null``
        (auto-arbitration). Otherwise the validated :data:`AIMode`.

    Raises:
        KeyError: If the field is absent — every valid AIStateDict
            carries ``manual_mode`` since 2026-07-11.
        ValueError: If the value is a string outside :data:`AI_MODES`.
        JSONTypeError: If the value is present but neither ``None`` nor
            a string.
    """
    if "manual_mode" not in data:
        raise KeyError("manual_mode")
    raw = data["manual_mode"]
    if raw is None:
        return None
    validated = require_str(data, "manual_mode")
    for mode in AI_MODES:
        if validated == mode:
            return mode
    raise ValueError(f"manual_mode must be one of {AI_MODES} or null, got {validated!r}")


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
        last_scan_ms=require_int(data, "last_scan_ms"),
        last_shoot_ms=require_int(data, "last_shoot_ms"),
        last_map_open_ms=require_int(data, "last_map_open_ms"),
        combat_target_id=require_int(data, "combat_target_id"),
        wind_down=require_bool(data, "wind_down"),
        break_escape_until_fuel=require_int(data, "break_escape_until_fuel"),
        combat_target_x=require_int(data, "combat_target_x"),
        combat_target_y=require_int(data, "combat_target_y"),
        killed_tank_ids=_decode_killed_tank_ids(data),
        session_kill_count=require_int(data, "session_kill_count"),
        session_hit_count=require_int(data, "session_hit_count"),
        session_miss_count=require_int(data, "session_miss_count"),
        session_reject_count=require_int(data, "session_reject_count"),
        blocked_combat_targets=_require_str_int_mapping(data, "blocked_combat_targets"),
        last_shot_target_id=require_int(data, "last_shot_target_id"),
        last_shot_target_name=require_str(data, "last_shot_target_name"),
        resource_target_kind=require_str(data, "resource_target_kind"),
        resource_target_x=require_int(data, "resource_target_x"),
        resource_target_y=require_int(data, "resource_target_y"),
        attempted_equipment_targets=_require_str_int_mapping(data, "attempted_equipment_targets"),
        last_landing_scan_viewport=require_str(data, "last_landing_scan_viewport"),
        suppress_landing_scan=require_bool(data, "suppress_landing_scan"),
        manual_mode=_decode_manual_mode(data),
        live_radars_used=require_int(data, "live_radars_used"),
        live_teleports=require_int(data, "live_teleports"),
        mine_clearance_aim_key=require_str(data, "mine_clearance_aim_key"),
        mine_clearance_shot_ms=require_int(data, "mine_clearance_shot_ms"),
        greeted_tank_ids=_require_str_int_mapping(data, "greeted_tank_ids"),
        pursuit_shot_target_id=require_int(data, "pursuit_shot_target_id"),
        pursuit_shot_ms=require_int(data, "pursuit_shot_ms"),
        last_scope_scout_ms=require_int(data, "last_scope_scout_ms"),
        sweep_anchor_x=require_int(data, "sweep_anchor_x"),
        sweep_anchor_y=require_int(data, "sweep_anchor_y"),
        maroon_pan_x=require_int(data, "maroon_pan_x"),
        maroon_pan_y=require_int(data, "maroon_pan_y"),
        visited_tank_ids=_require_str_int_mapping(data, "visited_tank_ids"),
    )


__all__ = [
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
]
