"""TypedDict models for live teleport probe sessions."""

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


class TeleportTargetDict(TypedDict):
    """Requested destination for a teleport probe attempt.

    Attributes:
        label: Human-readable label for the destination.
        x: Requested world X tile coordinate.
        y: Requested world Y tile coordinate.
    """

    label: str
    x: int
    y: int


def encode_teleport_target(target: TeleportTargetDict) -> JSONObject:
    """Encode a teleport target to a JSON object.

    Args:
        target: Teleport target to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "label": target["label"],
        "x": target["x"],
        "y": target["y"],
    }


def decode_teleport_target(data: JSONObject) -> TeleportTargetDict:
    """Decode a teleport target from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated teleport target.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return TeleportTargetDict(
        label=require_str(data, "label"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
    )


class TeleportAttemptResultDict(TypedDict):
    """Outcome of one teleport probe attempt.

    Attributes:
        target: Requested target for the attempt.
        teleport_cycle_id: Teleport phase cycle id for the attempt.
        status: Attempt result classification.
        map_open_started_ms: Timestamp when the map-open toggle was sent.
        map_sync_timestamp_ms: Timestamp of the first fresh world sync after
            the map-open toggle, if any.
        teleport_started_ms: Timestamp when the teleport command was sent, if any.
        completion_timestamp_ms: Timestamp when the attempt reached a terminal outcome.
        map_sync_elapsed_ms: Milliseconds from map-open send to fresh sync, if any.
        teleport_elapsed_ms: Milliseconds from teleport send to terminal outcome, if any.
        fuel_before: Fuel immediately before the map-open command.
        fuel_after: Fuel observed at completion, if self state is available.
        world_timestamp_before: World-state timestamp before the attempt began.
        world_timestamp_after: World-state timestamp at completion.
        landed_signal_received: Whether a teleport-landed confirmation was observed.
        landed_x: Actual landed X coordinate, if available.
        landed_y: Actual landed Y coordinate, if available.
        message_start_index: Index of the first raw captured message for the attempt.
        message_end_index: Exclusive index after the last raw captured message for the attempt.
        page_snapshots: Page-client diagnostic snapshots captured during the attempt.
    """

    target: TeleportTargetDict
    teleport_cycle_id: int
    status: Literal["landed_exact", "landed_offset", "map_sync_timeout", "teleport_timeout"]
    map_open_started_ms: int
    map_sync_timestamp_ms: int | None
    teleport_started_ms: int | None
    completion_timestamp_ms: int
    map_sync_elapsed_ms: int | None
    teleport_elapsed_ms: int | None
    fuel_before: int
    fuel_after: int | None
    world_timestamp_before: int
    world_timestamp_after: int
    landed_signal_received: bool
    landed_x: int | None
    landed_y: int | None
    message_start_index: int
    message_end_index: int
    page_snapshots: list[TeleportPageSnapshotDict]


class TeleportPageSnapshotDict(TypedDict):
    """Observed page-client state at a specific teleport attempt phase.

    Attributes:
        phase: Attempt phase when the snapshot was captured.
        timestamp_ms: Local timestamp when the snapshot was captured.
        client_present: Whether the page exposes an active game instance.
        map_visible: Whether the game client believes the map is open.
        client_state: Internal page-client action state identifier.
        client_busy: Whether the page-client marks itself busy.
        pending_actions: Number of queued page-client actions.
        heartbeat_age_ms: Milliseconds since the page-client heartbeat timestamp.
        last_page_client_send_age_ms: Milliseconds since the last page-client send.
        last_bot_send_age_ms: Milliseconds since the last bot-injected send.
        ws_ready_state: Browser WebSocket ready state for the captured socket.
        current_send_label: Bot send label currently active in the browser hook.
        sent_frame_meta_queue_length: Pending outbound metadata queue length.
    """

    phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]
    timestamp_ms: int
    client_present: bool
    map_visible: bool | None
    client_state: int | None
    client_busy: bool | None
    pending_actions: int | None
    heartbeat_age_ms: int | None
    last_page_client_send_age_ms: int | None
    last_bot_send_age_ms: int | None
    ws_ready_state: int | None
    current_send_label: str | None
    sent_frame_meta_queue_length: int


class TeleportStartupTimingDict(TypedDict):
    """Startup timing milestones for a live teleport probe session.

    Attributes:
        game_ready_timestamp_ms: Timestamp when the game-ready wait completed.
        intel_ready_timestamp_ms: Timestamp when probe intel collection completed.
        initial_sync_started_ms: Timestamp when initial world/self sync wait began.
        initial_world_timestamp_ms: Timestamp when the initial self state became available.
        command_ready_timestamp_ms: Timestamp when startup state advancement reached IDLE.
        first_attempt_started_ms: Timestamp when the first teleport attempt began, if any.
        game_ready_to_intel_ready_ms: Delay from game-ready to intel completion.
        intel_ready_to_initial_world_ms: Delay from intel completion to initial self state.
        initial_world_to_command_ready_ms: Delay from first self state to command-ready state.
        command_ready_to_first_attempt_ms: Delay from command-ready to first attempt, if any.
    """

    game_ready_timestamp_ms: int
    intel_ready_timestamp_ms: int
    initial_sync_started_ms: int
    initial_world_timestamp_ms: int
    command_ready_timestamp_ms: int
    first_attempt_started_ms: int | None
    game_ready_to_intel_ready_ms: int
    intel_ready_to_initial_world_ms: int
    initial_world_to_command_ready_ms: int
    command_ready_to_first_attempt_ms: int | None


def _encode_optional_int(value: int | None) -> JSONValue:
    """Encode an optional integer as a JSON scalar.

    Args:
        value: Integer value or None.

    Returns:
        JSON scalar suitable for serialization.
    """
    return value


def _encode_optional_str(value: str | None) -> JSONValue:
    """Encode an optional string as a JSON scalar.

    Args:
        value: String value or None.

    Returns:
        JSON scalar suitable for serialization.
    """
    return value


def _encode_optional_bool(value: bool | None) -> JSONValue:
    """Encode an optional boolean as a JSON scalar.

    Args:
        value: Boolean value or None.

    Returns:
        JSON scalar suitable for serialization.
    """
    return value


def encode_teleport_page_snapshot(snapshot: TeleportPageSnapshotDict) -> JSONObject:
    """Encode a teleport page snapshot to a JSON object.

    Args:
        snapshot: Snapshot to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "phase": snapshot["phase"],
        "timestamp_ms": snapshot["timestamp_ms"],
        "client_present": snapshot["client_present"],
        "map_visible": _encode_optional_bool(snapshot["map_visible"]),
        "client_state": _encode_optional_int(snapshot["client_state"]),
        "client_busy": _encode_optional_bool(snapshot["client_busy"]),
        "pending_actions": _encode_optional_int(snapshot["pending_actions"]),
        "heartbeat_age_ms": _encode_optional_int(snapshot["heartbeat_age_ms"]),
        "last_page_client_send_age_ms": _encode_optional_int(
            snapshot["last_page_client_send_age_ms"]
        ),
        "last_bot_send_age_ms": _encode_optional_int(snapshot["last_bot_send_age_ms"]),
        "ws_ready_state": _encode_optional_int(snapshot["ws_ready_state"]),
        "current_send_label": _encode_optional_str(snapshot["current_send_label"]),
        "sent_frame_meta_queue_length": snapshot["sent_frame_meta_queue_length"],
    }


def _encode_page_snapshot_list(snapshots: list[TeleportPageSnapshotDict]) -> JSONValue:
    """Encode a list of teleport page snapshots.

    Args:
        snapshots: Snapshot list to encode.

    Returns:
        JSON array representation.
    """
    return [encode_teleport_page_snapshot(snapshot) for snapshot in snapshots]


def encode_teleport_attempt_result(result: TeleportAttemptResultDict) -> JSONObject:
    """Encode a teleport attempt result to a JSON object.

    Args:
        result: Attempt result to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "target": encode_teleport_target(result["target"]),
        "teleport_cycle_id": result["teleport_cycle_id"],
        "status": result["status"],
        "map_open_started_ms": result["map_open_started_ms"],
        "map_sync_timestamp_ms": _encode_optional_int(result["map_sync_timestamp_ms"]),
        "teleport_started_ms": _encode_optional_int(result["teleport_started_ms"]),
        "completion_timestamp_ms": result["completion_timestamp_ms"],
        "map_sync_elapsed_ms": _encode_optional_int(result["map_sync_elapsed_ms"]),
        "teleport_elapsed_ms": _encode_optional_int(result["teleport_elapsed_ms"]),
        "fuel_before": result["fuel_before"],
        "fuel_after": _encode_optional_int(result["fuel_after"]),
        "world_timestamp_before": result["world_timestamp_before"],
        "world_timestamp_after": result["world_timestamp_after"],
        "landed_signal_received": result["landed_signal_received"],
        "landed_x": _encode_optional_int(result["landed_x"]),
        "landed_y": _encode_optional_int(result["landed_y"]),
        "message_start_index": result["message_start_index"],
        "message_end_index": result["message_end_index"],
        "page_snapshots": _encode_page_snapshot_list(result["page_snapshots"]),
    }


def _require_optional_int(data: JSONObject, field: str) -> int | None:
    """Return an optional integer field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Integer value or None when the field is null or missing.

    Raises:
        JSONTypeError: If the field is present but not an integer.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise JSONTypeError(f"Field '{field}' must be an integer or null")
    return raw


def _require_optional_bool(data: JSONObject, field: str) -> bool | None:
    """Return an optional boolean field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Boolean value or None.

    Raises:
        JSONTypeError: If the field is present but not a boolean.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise JSONTypeError(f"Field '{field}' must be a boolean or null")
    return raw


def _require_optional_str(data: JSONObject, field: str) -> str | None:
    """Return an optional string field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        String value or None.

    Raises:
        JSONTypeError: If the field is present but not a string.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise JSONTypeError(f"Field '{field}' must be a string or null")
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


def _require_page_snapshot_phase(
    data: JSONObject,
    field: str,
) -> Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]:
    """Validate a teleport page snapshot phase literal.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Validated snapshot phase.

    Raises:
        JSONTypeError: If the phase is unsupported.
    """
    raw = require_str(data, field)
    if raw == "before_map_open":
        return "before_map_open"
    if raw == "before_teleport":
        return "before_teleport"
    if raw == "after_map_data":
        return "after_map_data"
    if raw == "landed":
        return "landed"
    if raw == "timeout":
        return "timeout"
    raise JSONTypeError(f"Field '{field}' has invalid teleport page snapshot phase: {raw}")


def decode_teleport_page_snapshot(data: JSONObject) -> TeleportPageSnapshotDict:
    """Decode a teleport page snapshot from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated teleport page snapshot.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return TeleportPageSnapshotDict(
        phase=_require_page_snapshot_phase(data, "phase"),
        timestamp_ms=require_int(data, "timestamp_ms"),
        client_present=_require_bool_field(data, "client_present"),
        map_visible=_require_optional_bool(data, "map_visible"),
        client_state=_require_optional_int(data, "client_state"),
        client_busy=_require_optional_bool(data, "client_busy"),
        pending_actions=_require_optional_int(data, "pending_actions"),
        heartbeat_age_ms=_require_optional_int(data, "heartbeat_age_ms"),
        last_page_client_send_age_ms=_require_optional_int(data, "last_page_client_send_age_ms"),
        last_bot_send_age_ms=_require_optional_int(data, "last_bot_send_age_ms"),
        ws_ready_state=_require_optional_int(data, "ws_ready_state"),
        current_send_label=_require_optional_str(data, "current_send_label"),
        sent_frame_meta_queue_length=require_int(data, "sent_frame_meta_queue_length"),
    )


def _decode_page_snapshot_list(raw: JSONValue) -> list[TeleportPageSnapshotDict]:
    """Decode a list of teleport page snapshots.

    Args:
        raw: Raw JSON value to decode.

    Returns:
        Validated snapshot list.

    Raises:
        JSONTypeError: If the payload is not a list of objects.
    """
    items = require_list({"page_snapshots": raw}, "page_snapshots")
    result: list[TeleportPageSnapshotDict] = []
    for item in items:
        item_obj = item
        if not isinstance(item_obj, dict):
            raise JSONTypeError("Field 'page_snapshots' must contain only objects")
        result.append(decode_teleport_page_snapshot(item_obj))
    return result


def _require_attempt_status(
    data: JSONObject,
    field: str,
) -> Literal["landed_exact", "landed_offset", "map_sync_timeout", "teleport_timeout"]:
    """Validate a teleport attempt status literal.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Validated attempt status.

    Raises:
        JSONTypeError: If the status is not one of the supported literals.
    """
    raw = require_str(data, field)
    if raw == "landed_exact":
        return "landed_exact"
    if raw == "landed_offset":
        return "landed_offset"
    if raw == "map_sync_timeout":
        return "map_sync_timeout"
    if raw == "teleport_timeout":
        return "teleport_timeout"
    raise JSONTypeError(f"Field '{field}' has invalid teleport attempt status: {raw}")


def _require_teleport_strategy(
    data: JSONObject,
    field: str,
) -> Literal["sync_before_teleport", "immediate_after_map_open"]:
    """Validate a teleport strategy literal.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Validated teleport strategy.

    Raises:
        JSONTypeError: If the strategy is not one of the supported literals.
    """
    raw = require_str(data, field)
    if raw == "sync_before_teleport":
        return "sync_before_teleport"
    if raw == "immediate_after_map_open":
        return "immediate_after_map_open"
    raise JSONTypeError(f"Field '{field}' has invalid teleport strategy: {raw}")


def decode_teleport_attempt_result(data: JSONObject) -> TeleportAttemptResultDict:
    """Decode a teleport attempt result from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated teleport attempt result.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    target_raw = data.get("target")
    if not isinstance(target_raw, dict):
        raise JSONTypeError("Field 'target' must be an object")
    landed_signal_received_raw = data.get("landed_signal_received")
    if not isinstance(landed_signal_received_raw, bool):
        raise JSONTypeError("Field 'landed_signal_received' must be a boolean")
    return TeleportAttemptResultDict(
        target=decode_teleport_target(target_raw),
        teleport_cycle_id=require_int(data, "teleport_cycle_id"),
        status=_require_attempt_status(data, "status"),
        map_open_started_ms=require_int(data, "map_open_started_ms"),
        map_sync_timestamp_ms=_require_optional_int(data, "map_sync_timestamp_ms"),
        teleport_started_ms=_require_optional_int(data, "teleport_started_ms"),
        completion_timestamp_ms=require_int(data, "completion_timestamp_ms"),
        map_sync_elapsed_ms=_require_optional_int(data, "map_sync_elapsed_ms"),
        teleport_elapsed_ms=_require_optional_int(data, "teleport_elapsed_ms"),
        fuel_before=require_int(data, "fuel_before"),
        fuel_after=_require_optional_int(data, "fuel_after"),
        world_timestamp_before=require_int(data, "world_timestamp_before"),
        world_timestamp_after=require_int(data, "world_timestamp_after"),
        landed_signal_received=landed_signal_received_raw,
        landed_x=_require_optional_int(data, "landed_x"),
        landed_y=_require_optional_int(data, "landed_y"),
        message_start_index=require_int(data, "message_start_index"),
        message_end_index=require_int(data, "message_end_index"),
        page_snapshots=_decode_page_snapshot_list(data.get("page_snapshots")),
    )


def encode_teleport_startup_timing(timing: TeleportStartupTimingDict) -> JSONObject:
    """Encode teleport startup timing to a JSON object.

    Args:
        timing: Startup timing payload to encode.

    Returns:
        JSON-serializable object representation.
    """
    return {
        "game_ready_timestamp_ms": timing["game_ready_timestamp_ms"],
        "intel_ready_timestamp_ms": timing["intel_ready_timestamp_ms"],
        "initial_sync_started_ms": timing["initial_sync_started_ms"],
        "initial_world_timestamp_ms": timing["initial_world_timestamp_ms"],
        "command_ready_timestamp_ms": timing["command_ready_timestamp_ms"],
        "first_attempt_started_ms": _encode_optional_int(timing["first_attempt_started_ms"]),
        "game_ready_to_intel_ready_ms": timing["game_ready_to_intel_ready_ms"],
        "intel_ready_to_initial_world_ms": timing["intel_ready_to_initial_world_ms"],
        "initial_world_to_command_ready_ms": timing["initial_world_to_command_ready_ms"],
        "command_ready_to_first_attempt_ms": _encode_optional_int(
            timing["command_ready_to_first_attempt_ms"]
        ),
    }


def decode_teleport_startup_timing(data: JSONObject) -> TeleportStartupTimingDict:
    """Decode teleport startup timing from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated startup timing payload.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return TeleportStartupTimingDict(
        game_ready_timestamp_ms=require_int(data, "game_ready_timestamp_ms"),
        intel_ready_timestamp_ms=require_int(data, "intel_ready_timestamp_ms"),
        initial_sync_started_ms=require_int(data, "initial_sync_started_ms"),
        initial_world_timestamp_ms=require_int(data, "initial_world_timestamp_ms"),
        command_ready_timestamp_ms=require_int(data, "command_ready_timestamp_ms"),
        first_attempt_started_ms=_require_optional_int(data, "first_attempt_started_ms"),
        game_ready_to_intel_ready_ms=require_int(data, "game_ready_to_intel_ready_ms"),
        intel_ready_to_initial_world_ms=require_int(data, "intel_ready_to_initial_world_ms"),
        initial_world_to_command_ready_ms=require_int(data, "initial_world_to_command_ready_ms"),
        command_ready_to_first_attempt_ms=_require_optional_int(
            data,
            "command_ready_to_first_attempt_ms",
        ),
    )


class TeleportProbeSessionDict(TypedDict):
    """Complete live teleport probe session.

    Attributes:
        session_id: Probe session identifier.
        start_timestamp_ms: Session start timestamp in milliseconds.
        end_timestamp_ms: Session end timestamp in milliseconds.
        base_url: Target URL used for the session.
        spawn_x: Initial spawn X coordinate after joining the game.
        spawn_y: Initial spawn Y coordinate after joining the game.
        teleport_strategy: Selected teleport sequencing strategy.
        max_targets: Maximum number of targets requested for the session, if limited.
        capture_session_path: Path to the replayable raw capture session JSON.
        initial_sync_timeout_ms: Configured initial self-state sync timeout.
        startup_timing: Startup timing milestones before the first attempt.
        map_sync_timeout_ms: Configured map-sync timeout.
        teleport_timeout_ms: Configured teleport timeout.
        settle_delay_ms: Delay inserted after each completed attempt.
        targets: Requested target list for the session.
        attempts: Recorded attempt outcomes in order.
    """

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"]
    max_targets: int | None
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    map_sync_timeout_ms: int
    teleport_timeout_ms: int
    settle_delay_ms: int
    targets: list[TeleportTargetDict]
    attempts: list[TeleportAttemptResultDict]


def encode_teleport_probe_session(session: TeleportProbeSessionDict) -> JSONObject:
    """Encode a teleport probe session to a JSON object.

    Args:
        session: Session to encode.

    Returns:
        JSON-serializable object representation.
    """
    encoded_targets: list[JSONValue] = [encode_teleport_target(t) for t in session["targets"]]
    encoded_attempts: list[JSONValue] = [
        encode_teleport_attempt_result(a) for a in session["attempts"]
    ]
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "teleport_strategy": session["teleport_strategy"],
        "max_targets": _encode_optional_int(session["max_targets"]),
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "map_sync_timeout_ms": session["map_sync_timeout_ms"],
        "teleport_timeout_ms": session["teleport_timeout_ms"],
        "settle_delay_ms": session["settle_delay_ms"],
        "targets": encoded_targets,
        "attempts": encoded_attempts,
    }


def _decode_teleport_target_list(raw: JSONValue) -> list[TeleportTargetDict]:
    """Decode a list of teleport targets from raw JSON.

    Args:
        raw: Raw JSON list value.

    Returns:
        Decoded teleport targets.

    Raises:
        JSONTypeError: If any element is not an object.
    """
    result: list[TeleportTargetDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'targets' must contain only objects")
        result.append(decode_teleport_target(item))
    return result


def _decode_teleport_attempt_list(raw: JSONValue) -> list[TeleportAttemptResultDict]:
    """Decode a list of teleport attempt results from raw JSON.

    Args:
        raw: Raw JSON list value.

    Returns:
        Decoded teleport attempt results.

    Raises:
        JSONTypeError: If any element is not an object.
    """
    result: list[TeleportAttemptResultDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'attempts' must contain only objects")
        result.append(decode_teleport_attempt_result(item))
    return result


def decode_teleport_probe_session(data: JSONObject) -> TeleportProbeSessionDict:
    """Decode a teleport probe session from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated teleport probe session.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    startup_timing_raw = data.get("startup_timing")
    if not isinstance(startup_timing_raw, dict):
        raise JSONTypeError("Field 'startup_timing' must be an object")
    teleport_strategy = _require_teleport_strategy(data, "teleport_strategy")
    return TeleportProbeSessionDict(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=require_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        spawn_x=require_int(data, "spawn_x"),
        spawn_y=require_int(data, "spawn_y"),
        teleport_strategy=teleport_strategy,
        max_targets=_require_optional_int(data, "max_targets"),
        capture_session_path=require_str(data, "capture_session_path"),
        initial_sync_timeout_ms=require_int(data, "initial_sync_timeout_ms"),
        startup_timing=decode_teleport_startup_timing(startup_timing_raw),
        map_sync_timeout_ms=require_int(data, "map_sync_timeout_ms"),
        teleport_timeout_ms=require_int(data, "teleport_timeout_ms"),
        settle_delay_ms=require_int(data, "settle_delay_ms"),
        targets=_decode_teleport_target_list(data.get("targets")),
        attempts=_decode_teleport_attempt_list(data.get("attempts")),
    )


__all__ = [
    "TeleportAttemptResultDict",
    "TeleportPageSnapshotDict",
    "TeleportProbeSessionDict",
    "TeleportStartupTimingDict",
    "TeleportTargetDict",
    "decode_teleport_attempt_result",
    "decode_teleport_page_snapshot",
    "decode_teleport_probe_session",
    "decode_teleport_startup_timing",
    "decode_teleport_target",
    "encode_teleport_attempt_result",
    "encode_teleport_page_snapshot",
    "encode_teleport_probe_session",
    "encode_teleport_startup_timing",
    "encode_teleport_target",
]
