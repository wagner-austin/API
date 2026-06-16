"""TypedDict models for live command queue probes."""

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
    decode_teleport_startup_timing,
    encode_teleport_startup_timing,
)

QueueExperimentKind = Literal[
    "shoot_then_pickup",
    "shoot_then_shoot",
    "move_then_pickup",
]


class QueueCommandTimingDict(TypedDict):
    """Timing for a single command within a queue experiment."""

    label: str
    sent_ms: int
    ack_ms: int | None
    elapsed_ms: int | None


class QueueExperimentResultDict(TypedDict):
    """Outcome of one command queue experiment."""

    kind: QueueExperimentKind
    status: Literal["both_processed", "second_dropped", "timeout"]
    primary: QueueCommandTimingDict
    secondary: QueueCommandTimingDict
    inter_send_delay_ms: int
    total_elapsed_ms: int
    message_start_index: int
    message_end_index: int


class QueueProbeSessionDict(TypedDict):
    """Complete live command queue probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    experiment_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    experiments: list[QueueExperimentResultDict]


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


def _require_experiment_kind(data: JSONObject, field: str) -> QueueExperimentKind:
    """Validate an experiment kind field."""
    raw = require_str(data, field)
    if raw == "shoot_then_pickup":
        return "shoot_then_pickup"
    if raw == "shoot_then_shoot":
        return "shoot_then_shoot"
    if raw == "move_then_pickup":
        return "move_then_pickup"
    raise JSONTypeError(f"Field '{field}' has invalid experiment kind: {raw}")


def _require_experiment_status(
    data: JSONObject,
    field: str,
) -> Literal["both_processed", "second_dropped", "timeout"]:
    """Validate an experiment status field."""
    raw = require_str(data, field)
    if raw == "both_processed":
        return "both_processed"
    if raw == "second_dropped":
        return "second_dropped"
    if raw == "timeout":
        return "timeout"
    raise JSONTypeError(f"Field '{field}' has invalid experiment status: {raw}")


def encode_queue_command_timing(timing: QueueCommandTimingDict) -> JSONObject:
    """Encode a command timing entry."""
    return {
        "label": timing["label"],
        "sent_ms": timing["sent_ms"],
        "ack_ms": _encode_optional_int(timing["ack_ms"]),
        "elapsed_ms": _encode_optional_int(timing["elapsed_ms"]),
    }


def decode_queue_command_timing(data: JSONObject) -> QueueCommandTimingDict:
    """Decode a command timing entry with validation."""
    return QueueCommandTimingDict(
        label=require_str(data, "label"),
        sent_ms=require_int(data, "sent_ms"),
        ack_ms=_require_optional_int(data, "ack_ms"),
        elapsed_ms=_require_optional_int(data, "elapsed_ms"),
    )


def encode_queue_experiment_result(result: QueueExperimentResultDict) -> JSONObject:
    """Encode a queue experiment result."""
    return {
        "kind": result["kind"],
        "status": result["status"],
        "primary": encode_queue_command_timing(result["primary"]),
        "secondary": encode_queue_command_timing(result["secondary"]),
        "inter_send_delay_ms": result["inter_send_delay_ms"],
        "total_elapsed_ms": result["total_elapsed_ms"],
        "message_start_index": result["message_start_index"],
        "message_end_index": result["message_end_index"],
    }


def decode_queue_experiment_result(data: JSONObject) -> QueueExperimentResultDict:
    """Decode a queue experiment result with validation."""
    primary_raw = data.get("primary")
    if not isinstance(primary_raw, dict):
        raise JSONTypeError("Field 'primary' must be an object")
    secondary_raw = data.get("secondary")
    if not isinstance(secondary_raw, dict):
        raise JSONTypeError("Field 'secondary' must be an object")
    return QueueExperimentResultDict(
        kind=_require_experiment_kind(data, "kind"),
        status=_require_experiment_status(data, "status"),
        primary=decode_queue_command_timing(primary_raw),
        secondary=decode_queue_command_timing(secondary_raw),
        inter_send_delay_ms=require_int(data, "inter_send_delay_ms"),
        total_elapsed_ms=require_int(data, "total_elapsed_ms"),
        message_start_index=require_int(data, "message_start_index"),
        message_end_index=require_int(data, "message_end_index"),
    )


def encode_queue_probe_session(session: QueueProbeSessionDict) -> JSONObject:
    """Encode a queue probe session."""
    encoded_experiments: list[JSONValue] = [
        encode_queue_experiment_result(exp) for exp in session["experiments"]
    ]
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "experiment_timeout_ms": session["experiment_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "experiments": encoded_experiments,
    }


def _decode_experiments(raw: JSONValue) -> list[QueueExperimentResultDict]:
    """Decode a list of queue experiment results."""
    result: list[QueueExperimentResultDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'experiments' must contain only objects")
        result.append(decode_queue_experiment_result(item))
    return result


def decode_queue_probe_session(data: JSONObject) -> QueueProbeSessionDict:
    """Decode a queue probe session with validation."""
    startup_timing_raw = data.get("startup_timing")
    if not isinstance(startup_timing_raw, dict):
        raise JSONTypeError("Field 'startup_timing' must be an object")
    return QueueProbeSessionDict(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=require_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        spawn_x=require_int(data, "spawn_x"),
        spawn_y=require_int(data, "spawn_y"),
        capture_session_path=require_str(data, "capture_session_path"),
        initial_sync_timeout_ms=require_int(data, "initial_sync_timeout_ms"),
        experiment_timeout_ms=require_int(data, "experiment_timeout_ms"),
        startup_timing=decode_teleport_startup_timing(startup_timing_raw),
        experiments=_decode_experiments(data.get("experiments")),
    )


__all__ = [
    "QueueCommandTimingDict",
    "QueueExperimentKind",
    "QueueExperimentResultDict",
    "QueueProbeSessionDict",
    "decode_queue_command_timing",
    "decode_queue_experiment_result",
    "decode_queue_probe_session",
    "encode_queue_command_timing",
    "encode_queue_experiment_result",
    "encode_queue_probe_session",
]
