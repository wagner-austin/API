"""TypedDict models for combat accuracy probe sessions."""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.action_lab.types_codecs import (
    decode_teleport_startup_timing,
    encode_teleport_startup_timing,
)


class CombatShotResultDict(TypedDict):
    """Outcome of one shot fired at a target."""

    shot_number: int
    self_x: int
    self_y: int
    target_x: int
    target_y: int
    distance: int
    result: Literal["hit", "miss", "timeout"]
    weapon_byte: int | None
    target_name: str
    target_id: int
    timestamp_ms: int


class CombatEngagementDict(TypedDict):
    """Outcome of one full combat engagement against a target."""

    target_id: int
    target_name: str
    initial_target_x: int
    initial_target_y: int
    initial_distance: int
    landed_x: int
    landed_y: int
    shots: list[CombatShotResultDict]
    total_hits: int
    total_misses: int
    total_timeouts: int
    kill_confirmed: bool
    target_fled: bool
    final_target_x: int
    final_target_y: int
    final_distance: int


class CombatProbeSessionDict(TypedDict):
    """Complete live combat accuracy probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    max_engagements: int
    max_shots_per_engagement: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    engagements: list[CombatEngagementDict]


def encode_combat_shot_result(shot: CombatShotResultDict) -> JSONObject:
    """Encode a combat shot result to a JSON object.

    Args:
        shot: Shot result to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "shot_number": shot["shot_number"],
        "self_x": shot["self_x"],
        "self_y": shot["self_y"],
        "target_x": shot["target_x"],
        "target_y": shot["target_y"],
        "distance": shot["distance"],
        "result": shot["result"],
        "weapon_byte": shot["weapon_byte"],
        "target_name": shot["target_name"],
        "target_id": shot["target_id"],
        "timestamp_ms": shot["timestamp_ms"],
    }


def encode_combat_engagement(engagement: CombatEngagementDict) -> JSONObject:
    """Encode a combat engagement to a JSON object.

    Args:
        engagement: Engagement to encode.

    Returns:
        JSON-serializable object representation.
    """
    shots: list[JSONValue] = [encode_combat_shot_result(s) for s in engagement["shots"]]
    return {
        "target_id": engagement["target_id"],
        "target_name": engagement["target_name"],
        "initial_target_x": engagement["initial_target_x"],
        "initial_target_y": engagement["initial_target_y"],
        "initial_distance": engagement["initial_distance"],
        "landed_x": engagement["landed_x"],
        "landed_y": engagement["landed_y"],
        "shots": shots,
        "total_hits": engagement["total_hits"],
        "total_misses": engagement["total_misses"],
        "total_timeouts": engagement["total_timeouts"],
        "kill_confirmed": engagement["kill_confirmed"],
        "target_fled": engagement["target_fled"],
        "final_target_x": engagement["final_target_x"],
        "final_target_y": engagement["final_target_y"],
        "final_distance": engagement["final_distance"],
    }


def encode_combat_probe_session(session: CombatProbeSessionDict) -> JSONObject:
    """Encode a combat probe session to a JSON object.

    Args:
        session: Session to encode.

    Returns:
        JSON-serializable object representation.
    """
    engagements: list[JSONValue] = [encode_combat_engagement(e) for e in session["engagements"]]
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "max_engagements": session["max_engagements"],
        "max_shots_per_engagement": session["max_shots_per_engagement"],
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "engagements": engagements,
    }


def _require_shot_result_literal(
    data: JSONObject,
    field: str,
) -> Literal["hit", "miss", "timeout"]:
    """Validate a shot result literal.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Validated shot result.

    Raises:
        JSONTypeError: If the value is unsupported.
    """
    raw = require_str(data, field)
    if raw == "hit":
        return "hit"
    if raw == "miss":
        return "miss"
    if raw == "timeout":
        return "timeout"
    raise JSONTypeError(f"Field '{field}' has invalid shot result: {raw}")


def _require_optional_int(data: JSONObject, field: str) -> int | None:
    """Return an optional integer field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Integer value or None.

    Raises:
        JSONTypeError: If the field is present but not an integer.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise JSONTypeError(f"Field '{field}' must be an integer or null")
    return raw


def _require_bool_field(data: JSONObject, field: str) -> bool:
    """Return a required boolean field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If the field is not a boolean.
    """
    raw = data.get(field)
    if not isinstance(raw, bool):
        raise JSONTypeError(f"Field '{field}' must be a boolean")
    return raw


def decode_combat_shot_result(data: JSONObject) -> CombatShotResultDict:
    """Decode a combat shot result from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated shot result.
    """
    return CombatShotResultDict(
        shot_number=require_int(data, "shot_number"),
        self_x=require_int(data, "self_x"),
        self_y=require_int(data, "self_y"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        distance=require_int(data, "distance"),
        result=_require_shot_result_literal(data, "result"),
        weapon_byte=_require_optional_int(data, "weapon_byte"),
        target_name=require_str(data, "target_name"),
        target_id=require_int(data, "target_id"),
        timestamp_ms=require_int(data, "timestamp_ms"),
    )


def decode_combat_engagement(data: JSONObject) -> CombatEngagementDict:
    """Decode a combat engagement from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated combat engagement.
    """
    raw_shots = require_list(data, "shots")
    shots: list[CombatShotResultDict] = []
    for item in raw_shots:
        if not isinstance(item, dict):
            raise JSONTypeError("shots must contain objects")
        shots.append(decode_combat_shot_result(item))
    return CombatEngagementDict(
        target_id=require_int(data, "target_id"),
        target_name=require_str(data, "target_name"),
        initial_target_x=require_int(data, "initial_target_x"),
        initial_target_y=require_int(data, "initial_target_y"),
        initial_distance=require_int(data, "initial_distance"),
        landed_x=require_int(data, "landed_x"),
        landed_y=require_int(data, "landed_y"),
        shots=shots,
        total_hits=require_int(data, "total_hits"),
        total_misses=require_int(data, "total_misses"),
        total_timeouts=require_int(data, "total_timeouts"),
        kill_confirmed=_require_bool_field(data, "kill_confirmed"),
        target_fled=_require_bool_field(data, "target_fled"),
        final_target_x=require_int(data, "final_target_x"),
        final_target_y=require_int(data, "final_target_y"),
        final_distance=require_int(data, "final_distance"),
    )


def decode_combat_probe_session(
    data: JSONObject,
) -> CombatProbeSessionDict:
    """Decode a combat probe session from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated combat probe session.
    """
    startup_timing_raw = data.get("startup_timing")
    if not isinstance(startup_timing_raw, dict):
        raise JSONTypeError("Field 'startup_timing' must be an object")
    raw_engagements = require_list(data, "engagements")
    engagements: list[CombatEngagementDict] = []
    for item in raw_engagements:
        if not isinstance(item, dict):
            raise JSONTypeError("engagements must contain objects")
        engagements.append(decode_combat_engagement(item))
    return CombatProbeSessionDict(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=require_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        spawn_x=require_int(data, "spawn_x"),
        spawn_y=require_int(data, "spawn_y"),
        max_engagements=require_int(data, "max_engagements"),
        max_shots_per_engagement=require_int(data, "max_shots_per_engagement"),
        capture_session_path=require_str(data, "capture_session_path"),
        initial_sync_timeout_ms=require_int(data, "initial_sync_timeout_ms"),
        startup_timing=decode_teleport_startup_timing(startup_timing_raw),
        engagements=engagements,
    )


__all__ = [
    "CombatEngagementDict",
    "CombatProbeSessionDict",
    "CombatShotResultDict",
    "decode_combat_engagement",
    "decode_combat_probe_session",
    "decode_combat_shot_result",
    "encode_combat_engagement",
    "encode_combat_probe_session",
    "encode_combat_shot_result",
]
