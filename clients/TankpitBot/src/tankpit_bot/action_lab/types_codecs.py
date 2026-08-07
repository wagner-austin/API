"""Encode/decode functions for action-lab teleport probe TypedDicts.

Separated from types.py to keep type definitions under 400 lines.
"""

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

from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportProbeSessionDict,
    TeleportStartupTimingDict,
    TeleportTargetDict,
)
from tankpit_bot.browser.page_client_snapshot import (
    decode_page_client_snapshot,
    encode_page_client_snapshot,
)


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


def _encode_optional_int(value: int | None) -> JSONValue:
    """Encode an optional integer as a JSON scalar.

    Args:
        value: Integer value or None.

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
    encoded: JSONObject = {"phase": snapshot["phase"]}
    encoded.update(encode_page_client_snapshot(snapshot))
    return encoded


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
    phase = _require_page_snapshot_phase(data, "phase")
    base = decode_page_client_snapshot(data)
    return TeleportPageSnapshotDict(
        phase=phase,
        timestamp_ms=base["timestamp_ms"],
        client_present=base["client_present"],
        map_visible=base["map_visible"],
        client_state=base["client_state"],
        client_busy=base["client_busy"],
        pending_actions=base["pending_actions"],
        heartbeat_age_ms=base["heartbeat_age_ms"],
        last_page_client_send_age_ms=base["last_page_client_send_age_ms"],
        last_bot_send_age_ms=base["last_bot_send_age_ms"],
        ws_ready_state=base["ws_ready_state"],
        current_send_label=base["current_send_label"],
        sent_frame_meta_queue_length=base["sent_frame_meta_queue_length"],
        self_fields=base["self_fields"],
        world_fields=base["world_fields"],
        map_fields=base["map_fields"],
        world_collections=base["world_collections"],
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
