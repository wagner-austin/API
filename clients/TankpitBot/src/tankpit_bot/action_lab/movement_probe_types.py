"""TypedDict models for live movement action probes."""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot.action_lab.types import (
    TeleportStartupTimingDict,
    TeleportTargetDict,
    decode_teleport_startup_timing,
    decode_teleport_target,
    encode_teleport_startup_timing,
    encode_teleport_target,
)


class MovementProbeAttemptResultDict(TypedDict):
    """Outcome of one live movement attempt."""

    target: TeleportTargetDict
    status: Literal["arrived_exact", "move_timeout"]
    move_started_ms: int
    map_open_requested_ms: int | None
    map_open_message_timestamp_ms: int | None
    completion_timestamp_ms: int
    move_elapsed_ms: int
    fuel_before: int
    fuel_after: int | None
    world_timestamp_before: int
    world_timestamp_after: int
    settled_x: int | None
    settled_y: int | None
    message_start_index: int
    message_end_index: int


class MovementProbeSessionDict(TypedDict):
    """Complete live movement probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    max_targets: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    move_timeout_ms: int
    settle_delay_ms: int
    queue_map_open_during_move: bool
    map_open_delay_ms: int
    targets: list[TeleportTargetDict]
    attempts: list[MovementProbeAttemptResultDict]


def _encode_optional_int(value: int | None) -> JSONValue:
    """Encode an optional integer."""
    return value


def _require_optional_int(data: JSONObject, field: str) -> int | None:
    """Return an optional integer field."""
    raw = data.get(field)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise JSONTypeError(f"Field '{field}' must be an integer or null")
    return raw


def _require_bool_field(data: JSONObject, field: str) -> bool:
    """Return a required boolean field."""
    raw = data.get(field)
    if not isinstance(raw, bool):
        raise JSONTypeError(f"Field '{field}' must be a boolean")
    return raw


def _require_status(
    data: JSONObject,
    field: str,
) -> Literal["arrived_exact", "move_timeout"]:
    """Validate a movement probe attempt status."""
    raw = require_str(data, field)
    if raw == "arrived_exact":
        return "arrived_exact"
    if raw == "move_timeout":
        return "move_timeout"
    raise JSONTypeError(f"Field '{field}' has invalid movement probe status: {raw}")


def encode_movement_probe_attempt_result(result: MovementProbeAttemptResultDict) -> JSONObject:
    """Encode a movement probe attempt result."""
    return {
        "target": encode_teleport_target(result["target"]),
        "status": result["status"],
        "move_started_ms": result["move_started_ms"],
        "map_open_requested_ms": _encode_optional_int(result["map_open_requested_ms"]),
        "map_open_message_timestamp_ms": _encode_optional_int(
            result["map_open_message_timestamp_ms"]
        ),
        "completion_timestamp_ms": result["completion_timestamp_ms"],
        "move_elapsed_ms": result["move_elapsed_ms"],
        "fuel_before": result["fuel_before"],
        "fuel_after": _encode_optional_int(result["fuel_after"]),
        "world_timestamp_before": result["world_timestamp_before"],
        "world_timestamp_after": result["world_timestamp_after"],
        "settled_x": _encode_optional_int(result["settled_x"]),
        "settled_y": _encode_optional_int(result["settled_y"]),
        "message_start_index": result["message_start_index"],
        "message_end_index": result["message_end_index"],
    }


def decode_movement_probe_attempt_result(data: JSONObject) -> MovementProbeAttemptResultDict:
    """Decode a movement probe attempt result with validation."""
    target_raw = data.get("target")
    if not isinstance(target_raw, dict):
        raise JSONTypeError("Field 'target' must be an object")
    return MovementProbeAttemptResultDict(
        target=decode_teleport_target(target_raw),
        status=_require_status(data, "status"),
        move_started_ms=require_int(data, "move_started_ms"),
        map_open_requested_ms=_require_optional_int(data, "map_open_requested_ms"),
        map_open_message_timestamp_ms=_require_optional_int(data, "map_open_message_timestamp_ms"),
        completion_timestamp_ms=require_int(data, "completion_timestamp_ms"),
        move_elapsed_ms=require_int(data, "move_elapsed_ms"),
        fuel_before=require_int(data, "fuel_before"),
        fuel_after=_require_optional_int(data, "fuel_after"),
        world_timestamp_before=require_int(data, "world_timestamp_before"),
        world_timestamp_after=require_int(data, "world_timestamp_after"),
        settled_x=_require_optional_int(data, "settled_x"),
        settled_y=_require_optional_int(data, "settled_y"),
        message_start_index=require_int(data, "message_start_index"),
        message_end_index=require_int(data, "message_end_index"),
    )


def encode_movement_probe_session(session: MovementProbeSessionDict) -> JSONObject:
    """Encode a movement probe session."""
    encoded_targets: list[JSONValue] = [
        encode_teleport_target(target) for target in session["targets"]
    ]
    encoded_attempts: list[JSONValue] = [
        encode_movement_probe_attempt_result(attempt) for attempt in session["attempts"]
    ]
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "max_targets": session["max_targets"],
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "move_timeout_ms": session["move_timeout_ms"],
        "settle_delay_ms": session["settle_delay_ms"],
        "queue_map_open_during_move": session["queue_map_open_during_move"],
        "map_open_delay_ms": session["map_open_delay_ms"],
        "targets": encoded_targets,
        "attempts": encoded_attempts,
    }


def _decode_targets(raw: JSONValue) -> list[TeleportTargetDict]:
    """Decode a list of movement probe targets."""
    result: list[TeleportTargetDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'targets' must contain only objects")
        result.append(decode_teleport_target(item))
    return result


def _decode_attempts(raw: JSONValue) -> list[MovementProbeAttemptResultDict]:
    """Decode a list of movement probe attempts."""
    result: list[MovementProbeAttemptResultDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'attempts' must contain only objects")
        result.append(decode_movement_probe_attempt_result(item))
    return result


def decode_movement_probe_session(data: JSONObject) -> MovementProbeSessionDict:
    """Decode a movement probe session with validation."""
    startup_timing_raw = data.get("startup_timing")
    if not isinstance(startup_timing_raw, dict):
        raise JSONTypeError("Field 'startup_timing' must be an object")
    return MovementProbeSessionDict(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=require_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        spawn_x=require_int(data, "spawn_x"),
        spawn_y=require_int(data, "spawn_y"),
        max_targets=require_int(data, "max_targets"),
        capture_session_path=require_str(data, "capture_session_path"),
        initial_sync_timeout_ms=require_int(data, "initial_sync_timeout_ms"),
        startup_timing=decode_teleport_startup_timing(startup_timing_raw),
        move_timeout_ms=require_int(data, "move_timeout_ms"),
        settle_delay_ms=require_int(data, "settle_delay_ms"),
        queue_map_open_during_move=_require_bool_field(data, "queue_map_open_during_move"),
        map_open_delay_ms=require_int(data, "map_open_delay_ms"),
        targets=_decode_targets(data.get("targets")),
        attempts=_decode_attempts(data.get("attempts")),
    )


__all__ = [
    "MovementProbeAttemptResultDict",
    "MovementProbeSessionDict",
    "decode_movement_probe_attempt_result",
    "decode_movement_probe_session",
    "encode_movement_probe_attempt_result",
    "encode_movement_probe_session",
]
