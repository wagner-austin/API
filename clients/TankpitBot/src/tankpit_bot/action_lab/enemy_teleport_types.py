"""TypedDict models for enemy-directed teleport probe sessions."""

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

from tankpit_bot.action_lab.types import (
    TeleportStartupTimingDict,
    TeleportTargetDict,
)
from tankpit_bot.action_lab.types_codecs import (
    decode_teleport_startup_timing,
    decode_teleport_target,
    encode_teleport_startup_timing,
    encode_teleport_target,
)
from tankpit_bot.bot.ai.types_codecs import decode_enemy_threat, encode_enemy_threat
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.browser.page_client_snapshot import (
    PageClientSnapshotDict,
    decode_page_client_snapshot,
    encode_page_client_snapshot,
)


class EnemyTeleportAttemptResultDict(TypedDict):
    """Outcome of one enemy-directed teleport attempt.

    Includes the page-client snapshots captured immediately before the
    acquisition command is dispatched and immediately before the attempt
    finalizes (whether it lands, times out, or terminates early).
    Comparing ``snapshot_before`` and ``snapshot_after`` lets reviewers
    confirm the live JS client's view of the tank's state at each
    boundary without watching the live game.
    """

    acquisition_strategy: Literal["map_open", "nearest_enemy"]
    status: Literal[
        "landed_adjacent",
        "landed_not_adjacent",
        "no_enemy",
        "no_landing_tile",
        "acquisition_timeout",
        "teleport_timeout",
    ]
    acquisition_started_ms: int
    acquisition_sync_timestamp_ms: int | None
    teleport_started_ms: int | None
    completion_timestamp_ms: int
    acquisition_elapsed_ms: int | None
    teleport_elapsed_ms: int | None
    fuel_before: int
    fuel_after: int | None
    world_timestamp_before: int
    world_timestamp_after: int
    enemy: EnemyThreatDict | None
    landing_target: TeleportTargetDict | None
    landed_signal_received: bool
    landed_x: int | None
    landed_y: int | None
    enemy_still_visible: bool
    enemy_distance_after: int | None
    enemy_x_after: int | None
    enemy_y_after: int | None
    message_start_index: int
    message_end_index: int
    snapshot_before: PageClientSnapshotDict
    snapshot_after: PageClientSnapshotDict


class EnemyTeleportProbeSessionDict(TypedDict):
    """Complete live enemy-directed teleport probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    acquisition_strategy: Literal["map_open", "nearest_enemy"]
    max_attempts: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    acquisition_timeout_ms: int
    teleport_timeout_ms: int
    settle_delay_ms: int
    heartbeat_interval_ms: int
    attempts: list[EnemyTeleportAttemptResultDict]


def _encode_optional_int(value: int | None) -> JSONValue:
    """Encode an optional integer as a JSON scalar.

    Args:
        value: Integer value or None.

    Returns:
        JSON scalar suitable for serialization.
    """
    return value


def _encode_optional_enemy(value: EnemyThreatDict | None) -> JSONValue:
    """Encode an optional enemy threat payload.

    Args:
        value: Enemy threat or None.

    Returns:
        JSON object or None.
    """
    if value is None:
        return None
    return encode_enemy_threat(value)


def _encode_optional_target(value: TeleportTargetDict | None) -> JSONValue:
    """Encode an optional teleport target payload.

    Args:
        value: Teleport target or None.

    Returns:
        JSON object or None.
    """
    if value is None:
        return None
    return encode_teleport_target(value)


def encode_enemy_teleport_attempt_result(result: EnemyTeleportAttemptResultDict) -> JSONObject:
    """Encode an enemy teleport attempt result to a JSON object.

    Args:
        result: Attempt result to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "acquisition_strategy": result["acquisition_strategy"],
        "status": result["status"],
        "acquisition_started_ms": result["acquisition_started_ms"],
        "acquisition_sync_timestamp_ms": _encode_optional_int(
            result["acquisition_sync_timestamp_ms"]
        ),
        "teleport_started_ms": _encode_optional_int(result["teleport_started_ms"]),
        "completion_timestamp_ms": result["completion_timestamp_ms"],
        "acquisition_elapsed_ms": _encode_optional_int(result["acquisition_elapsed_ms"]),
        "teleport_elapsed_ms": _encode_optional_int(result["teleport_elapsed_ms"]),
        "fuel_before": result["fuel_before"],
        "fuel_after": _encode_optional_int(result["fuel_after"]),
        "world_timestamp_before": result["world_timestamp_before"],
        "world_timestamp_after": result["world_timestamp_after"],
        "enemy": _encode_optional_enemy(result["enemy"]),
        "landing_target": _encode_optional_target(result["landing_target"]),
        "landed_signal_received": result["landed_signal_received"],
        "landed_x": _encode_optional_int(result["landed_x"]),
        "landed_y": _encode_optional_int(result["landed_y"]),
        "enemy_still_visible": result["enemy_still_visible"],
        "enemy_distance_after": _encode_optional_int(result["enemy_distance_after"]),
        "enemy_x_after": _encode_optional_int(result["enemy_x_after"]),
        "enemy_y_after": _encode_optional_int(result["enemy_y_after"]),
        "message_start_index": result["message_start_index"],
        "message_end_index": result["message_end_index"],
        "snapshot_before": encode_page_client_snapshot(result["snapshot_before"]),
        "snapshot_after": encode_page_client_snapshot(result["snapshot_after"]),
    }


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


def _require_acquisition_strategy(
    data: JSONObject,
    field: str,
) -> Literal["map_open", "nearest_enemy"]:
    """Validate an enemy acquisition strategy literal.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Validated acquisition strategy.

    Raises:
        JSONTypeError: If the value is unsupported.
    """
    raw = require_str(data, field)
    if raw == "map_open":
        return "map_open"
    if raw == "nearest_enemy":
        return "nearest_enemy"
    raise JSONTypeError(f"Field '{field}' has invalid acquisition strategy: {raw}")


def _require_attempt_status(
    data: JSONObject,
    field: str,
) -> Literal[
    "landed_adjacent",
    "landed_not_adjacent",
    "no_enemy",
    "no_landing_tile",
    "acquisition_timeout",
    "teleport_timeout",
]:
    """Validate an enemy teleport attempt status literal.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Validated attempt status.

    Raises:
        JSONTypeError: If the value is unsupported.
    """
    raw = require_str(data, field)
    if raw == "landed_adjacent":
        return "landed_adjacent"
    if raw == "landed_not_adjacent":
        return "landed_not_adjacent"
    if raw == "no_enemy":
        return "no_enemy"
    if raw == "no_landing_tile":
        return "no_landing_tile"
    if raw == "acquisition_timeout":
        return "acquisition_timeout"
    if raw == "teleport_timeout":
        return "teleport_timeout"
    raise JSONTypeError(f"Field '{field}' has invalid enemy teleport status: {raw}")


def _decode_optional_enemy(data: JSONObject, field: str) -> EnemyThreatDict | None:
    """Decode an optional enemy threat field.

    Args:
        data: JSON object to inspect.
        field: Field name to decode.

    Returns:
        Enemy threat or None.

    Raises:
        JSONTypeError: If the field is present but not an object.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object or null")
    return decode_enemy_threat(raw)


def _require_snapshot(data: JSONObject, field: str) -> PageClientSnapshotDict:
    """Decode a required page-client snapshot field.

    Args:
        data: JSON object being decoded.
        field: Field name to read.

    Returns:
        Validated page-client snapshot.

    Raises:
        JSONTypeError: If the field is missing or not a JSON object.
    """
    raw = data.get(field)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object")
    return decode_page_client_snapshot(raw)


def _decode_optional_target(data: JSONObject, field: str) -> TeleportTargetDict | None:
    """Decode an optional teleport target field.

    Args:
        data: JSON object to inspect.
        field: Field name to decode.

    Returns:
        Teleport target or None.

    Raises:
        JSONTypeError: If the field is present but not an object.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object or null")
    return decode_teleport_target(raw)


def decode_enemy_teleport_attempt_result(data: JSONObject) -> EnemyTeleportAttemptResultDict:
    """Decode an enemy teleport attempt result from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated attempt result.
    """
    return EnemyTeleportAttemptResultDict(
        acquisition_strategy=_require_acquisition_strategy(data, "acquisition_strategy"),
        status=_require_attempt_status(data, "status"),
        acquisition_started_ms=require_int(data, "acquisition_started_ms"),
        acquisition_sync_timestamp_ms=_require_optional_int(data, "acquisition_sync_timestamp_ms"),
        teleport_started_ms=_require_optional_int(data, "teleport_started_ms"),
        completion_timestamp_ms=require_int(data, "completion_timestamp_ms"),
        acquisition_elapsed_ms=_require_optional_int(data, "acquisition_elapsed_ms"),
        teleport_elapsed_ms=_require_optional_int(data, "teleport_elapsed_ms"),
        fuel_before=require_int(data, "fuel_before"),
        fuel_after=_require_optional_int(data, "fuel_after"),
        world_timestamp_before=require_int(data, "world_timestamp_before"),
        world_timestamp_after=require_int(data, "world_timestamp_after"),
        enemy=_decode_optional_enemy(data, "enemy"),
        landing_target=_decode_optional_target(data, "landing_target"),
        landed_signal_received=_require_bool_field(data, "landed_signal_received"),
        landed_x=_require_optional_int(data, "landed_x"),
        landed_y=_require_optional_int(data, "landed_y"),
        enemy_still_visible=_require_bool_field(data, "enemy_still_visible"),
        enemy_distance_after=_require_optional_int(data, "enemy_distance_after"),
        enemy_x_after=_require_optional_int(data, "enemy_x_after"),
        enemy_y_after=_require_optional_int(data, "enemy_y_after"),
        message_start_index=require_int(data, "message_start_index"),
        message_end_index=require_int(data, "message_end_index"),
        snapshot_before=_require_snapshot(data, "snapshot_before"),
        snapshot_after=_require_snapshot(data, "snapshot_after"),
    )


def encode_enemy_teleport_probe_session(session: EnemyTeleportProbeSessionDict) -> JSONObject:
    """Encode an enemy teleport probe session to a JSON object.

    Args:
        session: Session to encode.

    Returns:
        JSON-serializable object representation.
    """
    encoded_attempts: list[JSONValue] = [
        encode_enemy_teleport_attempt_result(attempt) for attempt in session["attempts"]
    ]
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "acquisition_strategy": session["acquisition_strategy"],
        "max_attempts": session["max_attempts"],
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "acquisition_timeout_ms": session["acquisition_timeout_ms"],
        "teleport_timeout_ms": session["teleport_timeout_ms"],
        "settle_delay_ms": session["settle_delay_ms"],
        "heartbeat_interval_ms": session["heartbeat_interval_ms"],
        "attempts": encoded_attempts,
    }


def _decode_attempts(raw: JSONValue) -> list[EnemyTeleportAttemptResultDict]:
    """Decode a list of enemy teleport attempt results from raw JSON.

    Args:
        raw: Raw JSON list value.

    Returns:
        Decoded attempt results.

    Raises:
        JSONTypeError: If any element is not an object.
    """
    result: list[EnemyTeleportAttemptResultDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'attempts' must contain only objects")
        result.append(decode_enemy_teleport_attempt_result(item))
    return result


def decode_enemy_teleport_probe_session(data: JSONObject) -> EnemyTeleportProbeSessionDict:
    """Decode an enemy teleport probe session from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated enemy teleport probe session.
    """
    startup_timing_raw = data.get("startup_timing")
    if not isinstance(startup_timing_raw, dict):
        raise JSONTypeError("Field 'startup_timing' must be an object")
    return EnemyTeleportProbeSessionDict(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=require_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        spawn_x=require_int(data, "spawn_x"),
        spawn_y=require_int(data, "spawn_y"),
        acquisition_strategy=_require_acquisition_strategy(data, "acquisition_strategy"),
        max_attempts=require_int(data, "max_attempts"),
        capture_session_path=require_str(data, "capture_session_path"),
        initial_sync_timeout_ms=require_int(data, "initial_sync_timeout_ms"),
        startup_timing=decode_teleport_startup_timing(startup_timing_raw),
        acquisition_timeout_ms=require_int(data, "acquisition_timeout_ms"),
        teleport_timeout_ms=require_int(data, "teleport_timeout_ms"),
        settle_delay_ms=require_int(data, "settle_delay_ms"),
        heartbeat_interval_ms=require_int(data, "heartbeat_interval_ms"),
        attempts=_decode_attempts(data.get("attempts")),
    )


__all__ = [
    "EnemyTeleportAttemptResultDict",
    "EnemyTeleportProbeSessionDict",
    "decode_enemy_teleport_attempt_result",
    "decode_enemy_teleport_probe_session",
    "encode_enemy_teleport_attempt_result",
    "encode_enemy_teleport_probe_session",
]
