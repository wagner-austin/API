"""TypedDict models for fuel-dot verification probe sessions."""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot.action_lab.page_client_snapshot import (
    PageClientSnapshotDict,
    decode_page_client_snapshot,
    encode_page_client_snapshot,
)
from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.action_lab.types_codecs import (
    decode_teleport_startup_timing,
    encode_teleport_startup_timing,
)


class DotContainerObservationDict(TypedDict):
    """One radar-confirmed container observed during a dot visit."""

    x: int
    y: int
    is_fuel: bool
    volume: int


class FuelDotAttemptResultDict(TypedDict):
    """Outcome of one fuel-dot verification attempt.

    Each attempt opens the map (refreshing the dot atlas), teleports to
    the nearest unvisited dot, radars at the landing, and records the
    radar truth at the dot tile: ``container_on_dot`` is the container
    sitting exactly on the dot coordinate, or ``None`` when the dot is
    empty ground.
    """

    status: Literal[
        "fuel_on_dot",
        "equipment_on_dot",
        "empty_dot",
        "acquisition_timeout",
        "teleport_timeout",
        "radar_timeout",
    ]
    acquisition_started_ms: int
    acquisition_sync_timestamp_ms: int | None
    dots_in_atlas: int
    dot_x: int | None
    dot_y: int | None
    dot_distance: int | None
    teleport_started_ms: int | None
    radar_started_ms: int | None
    radar_sync_timestamp_ms: int | None
    completion_timestamp_ms: int
    fuel_before: int
    fuel_after: int | None
    landed_signal_received: bool
    landed_x: int | None
    landed_y: int | None
    container_on_dot: DotContainerObservationDict | None
    viewport_fuel_containers: list[DotContainerObservationDict]
    message_start_index: int
    message_end_index: int
    snapshot_before: PageClientSnapshotDict
    snapshot_after: PageClientSnapshotDict


class FuelDotProbeSessionDict(TypedDict):
    """Complete live fuel-dot verification probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    max_dots: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    acquisition_timeout_ms: int
    teleport_timeout_ms: int
    radar_timeout_ms: int
    settle_delay_ms: int
    attempts: list[FuelDotAttemptResultDict]


def encode_dot_container_observation(observation: DotContainerObservationDict) -> JSONObject:
    """Encode a dot container observation to a JSON object.

    Args:
        observation: Observation to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "x": observation["x"],
        "y": observation["y"],
        "is_fuel": observation["is_fuel"],
        "volume": observation["volume"],
    }


def decode_dot_container_observation(data: JSONObject) -> DotContainerObservationDict:
    """Decode a dot container observation from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated observation.
    """
    return DotContainerObservationDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        is_fuel=require_bool(data, "is_fuel"),
        volume=require_int(data, "volume"),
    )


def _encode_optional_int(value: int | None) -> JSONValue:
    """Encode an optional integer as a JSON scalar.

    Args:
        value: Integer value or None.

    Returns:
        JSON scalar suitable for serialization.
    """
    return value


def _encode_optional_observation(value: DotContainerObservationDict | None) -> JSONValue:
    """Encode an optional dot container observation.

    Args:
        value: Observation or None.

    Returns:
        JSON object or None.
    """
    if value is None:
        return None
    return encode_dot_container_observation(value)


def encode_fuel_dot_attempt_result(result: FuelDotAttemptResultDict) -> JSONObject:
    """Encode a fuel-dot attempt result to a JSON object.

    Args:
        result: Attempt result to encode.

    Returns:
        JSON-serializable object representation.
    """
    encoded_viewport: list[JSONValue] = [
        encode_dot_container_observation(observation)
        for observation in result["viewport_fuel_containers"]
    ]
    return {
        "status": result["status"],
        "acquisition_started_ms": result["acquisition_started_ms"],
        "acquisition_sync_timestamp_ms": _encode_optional_int(
            result["acquisition_sync_timestamp_ms"]
        ),
        "dots_in_atlas": result["dots_in_atlas"],
        "dot_x": _encode_optional_int(result["dot_x"]),
        "dot_y": _encode_optional_int(result["dot_y"]),
        "dot_distance": _encode_optional_int(result["dot_distance"]),
        "teleport_started_ms": _encode_optional_int(result["teleport_started_ms"]),
        "radar_started_ms": _encode_optional_int(result["radar_started_ms"]),
        "radar_sync_timestamp_ms": _encode_optional_int(result["radar_sync_timestamp_ms"]),
        "completion_timestamp_ms": result["completion_timestamp_ms"],
        "fuel_before": result["fuel_before"],
        "fuel_after": _encode_optional_int(result["fuel_after"]),
        "landed_signal_received": result["landed_signal_received"],
        "landed_x": _encode_optional_int(result["landed_x"]),
        "landed_y": _encode_optional_int(result["landed_y"]),
        "container_on_dot": _encode_optional_observation(result["container_on_dot"]),
        "viewport_fuel_containers": encoded_viewport,
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


def _require_attempt_status(
    data: JSONObject,
    field: str,
) -> Literal[
    "fuel_on_dot",
    "equipment_on_dot",
    "empty_dot",
    "acquisition_timeout",
    "teleport_timeout",
    "radar_timeout",
]:
    """Validate a fuel-dot attempt status literal.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Validated attempt status.

    Raises:
        JSONTypeError: If the value is unsupported.
    """
    raw = require_str(data, field)
    if raw == "fuel_on_dot":
        return "fuel_on_dot"
    if raw == "equipment_on_dot":
        return "equipment_on_dot"
    if raw == "empty_dot":
        return "empty_dot"
    if raw == "acquisition_timeout":
        return "acquisition_timeout"
    if raw == "teleport_timeout":
        return "teleport_timeout"
    if raw == "radar_timeout":
        return "radar_timeout"
    raise JSONTypeError(f"Field '{field}' has invalid fuel-dot status: {raw}")


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


def _decode_optional_observation(
    data: JSONObject,
    field: str,
) -> DotContainerObservationDict | None:
    """Decode an optional dot container observation field.

    Args:
        data: JSON object to inspect.
        field: Field name to decode.

    Returns:
        Observation or None.

    Raises:
        JSONTypeError: If the field is present but not an object.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object or null")
    return decode_dot_container_observation(raw)


def _decode_observation_list(data: JSONObject, field: str) -> list[DotContainerObservationDict]:
    """Decode a required list of dot container observations.

    Args:
        data: JSON object to inspect.
        field: Field name to decode.

    Returns:
        Decoded observations.

    Raises:
        JSONTypeError: If any element is not an object.
    """
    result: list[DotContainerObservationDict] = []
    for item in require_list(data, field):
        if not isinstance(item, dict):
            raise JSONTypeError(f"Field '{field}' must contain only objects")
        result.append(decode_dot_container_observation(item))
    return result


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


def decode_fuel_dot_attempt_result(data: JSONObject) -> FuelDotAttemptResultDict:
    """Decode a fuel-dot attempt result from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated attempt result.
    """
    return FuelDotAttemptResultDict(
        status=_require_attempt_status(data, "status"),
        acquisition_started_ms=require_int(data, "acquisition_started_ms"),
        acquisition_sync_timestamp_ms=_require_optional_int(data, "acquisition_sync_timestamp_ms"),
        dots_in_atlas=require_int(data, "dots_in_atlas"),
        dot_x=_require_optional_int(data, "dot_x"),
        dot_y=_require_optional_int(data, "dot_y"),
        dot_distance=_require_optional_int(data, "dot_distance"),
        teleport_started_ms=_require_optional_int(data, "teleport_started_ms"),
        radar_started_ms=_require_optional_int(data, "radar_started_ms"),
        radar_sync_timestamp_ms=_require_optional_int(data, "radar_sync_timestamp_ms"),
        completion_timestamp_ms=require_int(data, "completion_timestamp_ms"),
        fuel_before=require_int(data, "fuel_before"),
        fuel_after=_require_optional_int(data, "fuel_after"),
        landed_signal_received=_require_bool_field(data, "landed_signal_received"),
        landed_x=_require_optional_int(data, "landed_x"),
        landed_y=_require_optional_int(data, "landed_y"),
        container_on_dot=_decode_optional_observation(data, "container_on_dot"),
        viewport_fuel_containers=_decode_observation_list(data, "viewport_fuel_containers"),
        message_start_index=require_int(data, "message_start_index"),
        message_end_index=require_int(data, "message_end_index"),
        snapshot_before=_require_snapshot(data, "snapshot_before"),
        snapshot_after=_require_snapshot(data, "snapshot_after"),
    )


def encode_fuel_dot_probe_session(session: FuelDotProbeSessionDict) -> JSONObject:
    """Encode a fuel-dot probe session to a JSON object.

    Args:
        session: Session to encode.

    Returns:
        JSON-serializable object representation.
    """
    encoded_attempts: list[JSONValue] = [
        encode_fuel_dot_attempt_result(attempt) for attempt in session["attempts"]
    ]
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "max_dots": session["max_dots"],
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "acquisition_timeout_ms": session["acquisition_timeout_ms"],
        "teleport_timeout_ms": session["teleport_timeout_ms"],
        "radar_timeout_ms": session["radar_timeout_ms"],
        "settle_delay_ms": session["settle_delay_ms"],
        "attempts": encoded_attempts,
    }


def _decode_attempts(raw: JSONValue) -> list[FuelDotAttemptResultDict]:
    """Decode a list of fuel-dot attempt results from raw JSON.

    Args:
        raw: Raw JSON list value.

    Returns:
        Decoded attempt results.

    Raises:
        JSONTypeError: If any element is not an object.
    """
    result: list[FuelDotAttemptResultDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'attempts' must contain only objects")
        result.append(decode_fuel_dot_attempt_result(item))
    return result


def decode_fuel_dot_probe_session(data: JSONObject) -> FuelDotProbeSessionDict:
    """Decode a fuel-dot probe session from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated fuel-dot probe session.
    """
    startup_timing_raw = data.get("startup_timing")
    if not isinstance(startup_timing_raw, dict):
        raise JSONTypeError("Field 'startup_timing' must be an object")
    return FuelDotProbeSessionDict(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=require_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        spawn_x=require_int(data, "spawn_x"),
        spawn_y=require_int(data, "spawn_y"),
        max_dots=require_int(data, "max_dots"),
        capture_session_path=require_str(data, "capture_session_path"),
        initial_sync_timeout_ms=require_int(data, "initial_sync_timeout_ms"),
        startup_timing=decode_teleport_startup_timing(startup_timing_raw),
        acquisition_timeout_ms=require_int(data, "acquisition_timeout_ms"),
        teleport_timeout_ms=require_int(data, "teleport_timeout_ms"),
        radar_timeout_ms=require_int(data, "radar_timeout_ms"),
        settle_delay_ms=require_int(data, "settle_delay_ms"),
        attempts=_decode_attempts(data.get("attempts")),
    )


__all__ = [
    "DotContainerObservationDict",
    "FuelDotAttemptResultDict",
    "FuelDotProbeSessionDict",
    "decode_dot_container_observation",
    "decode_fuel_dot_attempt_result",
    "decode_fuel_dot_probe_session",
    "encode_dot_container_observation",
    "encode_fuel_dot_attempt_result",
    "encode_fuel_dot_probe_session",
]
