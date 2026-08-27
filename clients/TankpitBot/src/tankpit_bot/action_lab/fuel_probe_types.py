"""TypedDict models for live fuel action probes."""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseOverlapDict,
    FuelDecisionBasisDict,
    decode_action_phase_overlap,
    decode_fuel_decision_basis,
    encode_action_phase_overlap,
    encode_fuel_decision_basis,
)
from tankpit_bot.action_lab.types import TeleportStartupTimingDict, TeleportTargetDict
from tankpit_bot.action_lab.types_codecs import (
    decode_teleport_startup_timing,
    decode_teleport_target,
    encode_teleport_startup_timing,
    encode_teleport_target,
)
from tankpit_bot.browser.page_client_snapshot import (
    PageClientSnapshotDict,
    decode_page_client_snapshot,
    encode_page_client_snapshot,
)


class FuelProbeAttemptResultDict(TypedDict):
    """Outcome of one teleport-radar-fuel attempt.

    Includes the page-client snapshots captured immediately before the
    first command of the attempt is dispatched and immediately before the
    attempt finalizes at its terminal branch. Comparing
    ``snapshot_before`` and ``snapshot_after`` lets reviewers confirm
    the live JS client's view of the tank's state at each boundary
    without watching the live game.
    """

    target: TeleportTargetDict
    teleport_cycle_ids: list[int]
    radar_cycle_id: int | None
    move_cycle_id: int | None
    pickup_cycle_id: int | None
    status: Literal[
        "picked_up_fuel",
        "no_fuel_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    ]
    map_open_started_ms: int
    map_sync_timestamp_ms: int | None
    teleport_started_ms: int | None
    radar_started_ms: int | None
    radar_sync_timestamp_ms: int | None
    reposition_map_open_started_ms: int | None
    reposition_map_sync_timestamp_ms: int | None
    reposition_teleport_started_ms: int | None
    pickup_started_ms: int | None
    completion_timestamp_ms: int
    fuel_before: int
    fuel_after: int | None
    landed_signal_received: bool
    landed_x: int | None
    landed_y: int | None
    fuel_target_x: int | None
    fuel_target_y: int | None
    fuel_target_volume: int | None
    phase_overlaps: list[ActionPhaseOverlapDict]
    decision_basis: FuelDecisionBasisDict | None
    message_start_index: int
    message_end_index: int
    snapshot_before: PageClientSnapshotDict
    snapshot_after: PageClientSnapshotDict


class FuelProbeSessionDict(TypedDict):
    """Complete live fuel action probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    target_pickups: int
    max_attempts: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    map_sync_timeout_ms: int
    teleport_timeout_ms: int
    radar_timeout_ms: int
    pickup_timeout_ms: int
    settle_delay_ms: int
    attempts: list[FuelProbeAttemptResultDict]


def _encode_optional_int(value: int | None) -> JSONValue:
    """Encode an optional integer."""
    return value


def _encode_int_list(values: list[int]) -> list[JSONValue]:
    """Encode a list of integers as JSON values."""
    return list(values)


def encode_fuel_probe_attempt_result(result: FuelProbeAttemptResultDict) -> JSONObject:
    """Encode a fuel probe attempt result."""
    return {
        "target": encode_teleport_target(result["target"]),
        "teleport_cycle_ids": _encode_int_list(result["teleport_cycle_ids"]),
        "radar_cycle_id": _encode_optional_int(result["radar_cycle_id"]),
        "move_cycle_id": _encode_optional_int(result["move_cycle_id"]),
        "pickup_cycle_id": _encode_optional_int(result["pickup_cycle_id"]),
        "status": result["status"],
        "map_open_started_ms": result["map_open_started_ms"],
        "map_sync_timestamp_ms": _encode_optional_int(result["map_sync_timestamp_ms"]),
        "teleport_started_ms": _encode_optional_int(result["teleport_started_ms"]),
        "radar_started_ms": _encode_optional_int(result["radar_started_ms"]),
        "radar_sync_timestamp_ms": _encode_optional_int(result["radar_sync_timestamp_ms"]),
        "reposition_map_open_started_ms": _encode_optional_int(
            result["reposition_map_open_started_ms"]
        ),
        "reposition_map_sync_timestamp_ms": _encode_optional_int(
            result["reposition_map_sync_timestamp_ms"]
        ),
        "reposition_teleport_started_ms": _encode_optional_int(
            result["reposition_teleport_started_ms"]
        ),
        "pickup_started_ms": _encode_optional_int(result["pickup_started_ms"]),
        "completion_timestamp_ms": result["completion_timestamp_ms"],
        "fuel_before": result["fuel_before"],
        "fuel_after": _encode_optional_int(result["fuel_after"]),
        "landed_signal_received": result["landed_signal_received"],
        "landed_x": _encode_optional_int(result["landed_x"]),
        "landed_y": _encode_optional_int(result["landed_y"]),
        "fuel_target_x": _encode_optional_int(result["fuel_target_x"]),
        "fuel_target_y": _encode_optional_int(result["fuel_target_y"]),
        "fuel_target_volume": _encode_optional_int(result["fuel_target_volume"]),
        "phase_overlaps": [
            encode_action_phase_overlap(overlap) for overlap in result["phase_overlaps"]
        ],
        "decision_basis": (
            None
            if result["decision_basis"] is None
            else encode_fuel_decision_basis(result["decision_basis"])
        ),
        "message_start_index": result["message_start_index"],
        "message_end_index": result["message_end_index"],
        "snapshot_before": encode_page_client_snapshot(result["snapshot_before"]),
        "snapshot_after": encode_page_client_snapshot(result["snapshot_after"]),
    }


def _require_optional_int(data: JSONObject, field: str) -> int | None:
    """Return an optional integer field."""
    raw = data.get(field)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise JSONTypeError(f"Field '{field}' must be an integer or null")
    return raw


def _decode_phase_overlaps(raw: JSONValue) -> list[ActionPhaseOverlapDict]:
    """Decode a list of action-phase overlap diagnostics."""
    result: list[ActionPhaseOverlapDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'phase_overlaps' must contain only objects")
        result.append(decode_action_phase_overlap(item))
    return result


def _decode_optional_decision_basis(data: JSONObject, field: str) -> FuelDecisionBasisDict | None:
    """Decode an optional decision basis object."""
    raw = data.get(field)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object or null")
    return decode_fuel_decision_basis(raw)


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


def _decode_int_list(data: JSONObject, field: str) -> list[int]:
    """Decode a required list of integers."""
    result: list[int] = []
    for item in require_list(data, field):
        if isinstance(item, bool) or not isinstance(item, int):
            raise JSONTypeError(f"Field '{field}' must contain only integers")
        result.append(item)
    return result


def _require_status(
    data: JSONObject,
    field: str,
) -> Literal[
    "picked_up_fuel",
    "no_fuel_visible",
    "radar_timeout",
    "map_sync_timeout",
    "reposition_map_sync_timeout",
    "teleport_timeout",
    "reposition_teleport_timeout",
    "pickup_timeout",
]:
    """Validate a fuel probe attempt status."""
    raw = require_str(data, field)
    if raw == "picked_up_fuel":
        return "picked_up_fuel"
    if raw == "no_fuel_visible":
        return "no_fuel_visible"
    if raw == "radar_timeout":
        return "radar_timeout"
    if raw == "map_sync_timeout":
        return "map_sync_timeout"
    if raw == "reposition_map_sync_timeout":
        return "reposition_map_sync_timeout"
    if raw == "teleport_timeout":
        return "teleport_timeout"
    if raw == "reposition_teleport_timeout":
        return "reposition_teleport_timeout"
    if raw == "pickup_timeout":
        return "pickup_timeout"
    raise JSONTypeError(f"Field '{field}' has invalid fuel probe status: {raw}")


def decode_fuel_probe_attempt_result(data: JSONObject) -> FuelProbeAttemptResultDict:
    """Decode a fuel probe attempt result with validation."""
    target_raw = data.get("target")
    if not isinstance(target_raw, dict):
        raise JSONTypeError("Field 'target' must be an object")
    return FuelProbeAttemptResultDict(
        target=decode_teleport_target(target_raw),
        teleport_cycle_ids=_decode_int_list(data, "teleport_cycle_ids"),
        radar_cycle_id=_require_optional_int(data, "radar_cycle_id"),
        move_cycle_id=_require_optional_int(data, "move_cycle_id"),
        pickup_cycle_id=_require_optional_int(data, "pickup_cycle_id"),
        status=_require_status(data, "status"),
        map_open_started_ms=require_int(data, "map_open_started_ms"),
        map_sync_timestamp_ms=_require_optional_int(data, "map_sync_timestamp_ms"),
        teleport_started_ms=_require_optional_int(data, "teleport_started_ms"),
        radar_started_ms=_require_optional_int(data, "radar_started_ms"),
        radar_sync_timestamp_ms=_require_optional_int(data, "radar_sync_timestamp_ms"),
        reposition_map_open_started_ms=_require_optional_int(
            data,
            "reposition_map_open_started_ms",
        ),
        reposition_map_sync_timestamp_ms=_require_optional_int(
            data, "reposition_map_sync_timestamp_ms"
        ),
        reposition_teleport_started_ms=_require_optional_int(
            data, "reposition_teleport_started_ms"
        ),
        pickup_started_ms=_require_optional_int(data, "pickup_started_ms"),
        completion_timestamp_ms=require_int(data, "completion_timestamp_ms"),
        fuel_before=require_int(data, "fuel_before"),
        fuel_after=_require_optional_int(data, "fuel_after"),
        landed_signal_received=require_bool(data, "landed_signal_received"),
        landed_x=_require_optional_int(data, "landed_x"),
        landed_y=_require_optional_int(data, "landed_y"),
        fuel_target_x=_require_optional_int(data, "fuel_target_x"),
        fuel_target_y=_require_optional_int(data, "fuel_target_y"),
        fuel_target_volume=_require_optional_int(data, "fuel_target_volume"),
        phase_overlaps=_decode_phase_overlaps(data.get("phase_overlaps")),
        decision_basis=_decode_optional_decision_basis(data, "decision_basis"),
        message_start_index=require_int(data, "message_start_index"),
        message_end_index=require_int(data, "message_end_index"),
        snapshot_before=_require_snapshot(data, "snapshot_before"),
        snapshot_after=_require_snapshot(data, "snapshot_after"),
    )


def encode_fuel_probe_session(session: FuelProbeSessionDict) -> JSONObject:
    """Encode a fuel probe session."""
    encoded_attempts: list[JSONValue] = [
        encode_fuel_probe_attempt_result(attempt) for attempt in session["attempts"]
    ]
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "target_pickups": session["target_pickups"],
        "max_attempts": session["max_attempts"],
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "map_sync_timeout_ms": session["map_sync_timeout_ms"],
        "teleport_timeout_ms": session["teleport_timeout_ms"],
        "radar_timeout_ms": session["radar_timeout_ms"],
        "pickup_timeout_ms": session["pickup_timeout_ms"],
        "settle_delay_ms": session["settle_delay_ms"],
        "attempts": encoded_attempts,
    }


def _decode_attempts(raw: JSONValue) -> list[FuelProbeAttemptResultDict]:
    """Decode a list of fuel probe attempts."""
    result: list[FuelProbeAttemptResultDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'attempts' must contain only objects")
        result.append(decode_fuel_probe_attempt_result(item))
    return result


def decode_fuel_probe_session(data: JSONObject) -> FuelProbeSessionDict:
    """Decode a fuel probe session with validation."""
    startup_timing_raw = data.get("startup_timing")
    if not isinstance(startup_timing_raw, dict):
        raise JSONTypeError("Field 'startup_timing' must be an object")
    return FuelProbeSessionDict(
        session_id=require_str(data, "session_id"),
        start_timestamp_ms=require_int(data, "start_timestamp_ms"),
        end_timestamp_ms=require_int(data, "end_timestamp_ms"),
        base_url=require_str(data, "base_url"),
        spawn_x=require_int(data, "spawn_x"),
        spawn_y=require_int(data, "spawn_y"),
        target_pickups=require_int(data, "target_pickups"),
        max_attempts=require_int(data, "max_attempts"),
        capture_session_path=require_str(data, "capture_session_path"),
        initial_sync_timeout_ms=require_int(data, "initial_sync_timeout_ms"),
        startup_timing=decode_teleport_startup_timing(startup_timing_raw),
        map_sync_timeout_ms=require_int(data, "map_sync_timeout_ms"),
        teleport_timeout_ms=require_int(data, "teleport_timeout_ms"),
        radar_timeout_ms=require_int(data, "radar_timeout_ms"),
        pickup_timeout_ms=require_int(data, "pickup_timeout_ms"),
        settle_delay_ms=require_int(data, "settle_delay_ms"),
        attempts=_decode_attempts(data.get("attempts")),
    )


__all__ = [
    "FuelProbeAttemptResultDict",
    "FuelProbeSessionDict",
    "decode_fuel_probe_attempt_result",
    "decode_fuel_probe_session",
    "encode_fuel_probe_attempt_result",
    "encode_fuel_probe_session",
]
