"""Codecs for the issue report's per-event records, plus the JSON guards.

The base codec layer: the four ``_require_*`` narrowing helpers every
decoder in this family uses, and the encode/decode pairs for the
records that carry no nested structure. The scorecard codecs are
:mod:`tankpit_bot.diagnostics.issue_report_codecs_scorecard`; the
top-level report is :mod:`tankpit_bot.diagnostics.issue_report_codecs`.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_bool,
    require_dict,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot.diagnostics.issue_report_types import (
    ActionOutcomeRowDict,
    DisplacedTeleportRecordDict,
    FuelTargetSelectionRecordDict,
    MapOpenSkippedRecordDict,
    SessionRoomRecordDict,
    SuppressedDispatchRecordDict,
    TeleportAttemptRecordDict,
)


def _require_object(data: JSONObject, key: str) -> JSONObject:
    """Return a required nested JSON object field.

    Args:
        data: Parent JSON object.
        key: Field name.

    Returns:
        Nested object.

    Raises:
        JSONTypeError: When the field is absent or not a dict.
    """
    raw = data.get(key)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"{key} must be object, got {type(raw).__name__}")
    return raw


def _require_object_list(data: JSONObject, key: str) -> list[JSONObject]:
    """Return a required list field whose items are all JSON objects.

    Args:
        data: Parent JSON object.
        key: Field name to extract.

    Returns:
        Validated list of JSONObject items.

    Raises:
        JSONTypeError: When the field is absent, not a list, or contains
            a non-object element.
    """
    raw_items = require_list(data, key)
    result: list[JSONObject] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise JSONTypeError(f"{key}[{index}] must be object, got {type(raw).__name__}")
        result.append(raw)
    return result


def _require_object_or_none(data: JSONObject, key: str) -> JSONObject | None:
    """Return a nested JSON object or ``None`` when the field is null/absent.

    Args:
        data: Parent JSON object.
        key: Field name.

    Returns:
        Decoded object, or ``None`` when the value is JSON null / absent.

    Raises:
        JSONTypeError: When the field exists and is neither a dict nor null.
    """
    if key not in data:
        return None
    raw = data[key]
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise JSONTypeError(f"{key} must be object or null, got {type(raw).__name__}")
    return raw


def _require_str_int_map(data: JSONObject, key: str) -> dict[str, int]:
    """Validate and extract a str->int map from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated mapping.

    Raises:
        JSONTypeError: If any value is not an int.
    """
    raw = require_dict(data, key)
    result: dict[str, int] = {}
    for field, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise JSONTypeError(f"{key}[{field!r}] must be int")
        result[field] = value
    return result


def encode_teleport_attempt_record(record: TeleportAttemptRecordDict) -> JSONObject:
    """Encode a teleport attempt record to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "target_x": record["target_x"],
        "target_y": record["target_y"],
        "teleport_cycle_id": record["teleport_cycle_id"],
        "status": record["status"],
        "timestamp": record["timestamp"],
        "sent_window": record["sent_window"],
        "received_window": record["received_window"],
        "page_snapshot_count": record["page_snapshot_count"],
    }


def decode_teleport_attempt_record(data: JSONObject) -> TeleportAttemptRecordDict:
    """Decode a teleport attempt record from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record.
    """
    return TeleportAttemptRecordDict(
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        teleport_cycle_id=require_int(data, "teleport_cycle_id"),
        status=require_str(data, "status"),
        timestamp=require_str(data, "timestamp"),
        sent_window=require_str(data, "sent_window"),
        received_window=require_str(data, "received_window"),
        page_snapshot_count=require_int(data, "page_snapshot_count"),
    )


def encode_map_open_skipped_record(record: MapOpenSkippedRecordDict) -> JSONObject:
    """Encode a map_open_skipped record to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {"origin": record["origin"], "timestamp": record["timestamp"]}


def decode_map_open_skipped_record(data: JSONObject) -> MapOpenSkippedRecordDict:
    """Decode a map_open_skipped record from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record.
    """
    return MapOpenSkippedRecordDict(
        origin=require_str(data, "origin"),
        timestamp=require_str(data, "timestamp"),
    )


def encode_fuel_target_selection_record(
    record: FuelTargetSelectionRecordDict,
) -> JSONObject:
    """Encode a fuel target selection record to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "radar_cycle_id": record["radar_cycle_id"],
        "target_present": record["target_present"],
        "target_x": record["target_x"],
        "target_y": record["target_y"],
        "summary": record["summary"],
        "decision_basis": record["decision_basis"],
        "timestamp": record["timestamp"],
    }


def decode_fuel_target_selection_record(
    data: JSONObject,
) -> FuelTargetSelectionRecordDict:
    """Decode a fuel target selection record from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record.

    Raises:
        JSONTypeError: When ``target_present`` is not a boolean.
    """
    raw_present = data.get("target_present")
    if not isinstance(raw_present, bool):
        raise JSONTypeError(f"target_present must be bool, got {type(raw_present).__name__}")
    return FuelTargetSelectionRecordDict(
        radar_cycle_id=require_int(data, "radar_cycle_id"),
        target_present=raw_present,
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        summary=require_str(data, "summary"),
        decision_basis=require_str(data, "decision_basis"),
        timestamp=require_str(data, "timestamp"),
    )


def encode_action_outcome_row(record: ActionOutcomeRowDict) -> JSONObject:
    """Encode an action outcome row to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "action_kind": record["action_kind"],
        "outcome": record["outcome"],
        "event_id": record["event_id"],
        "attempt_id": record["attempt_id"],
        "duration_ms": record["duration_ms"],
        "dispatched": record["dispatched"],
        "timestamp": record["timestamp"],
    }


def decode_action_outcome_row(data: JSONObject) -> ActionOutcomeRowDict:
    """Decode an action outcome row from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record.
    """
    return ActionOutcomeRowDict(
        action_kind=require_str(data, "action_kind"),
        outcome=require_str(data, "outcome"),
        event_id=require_int(data, "event_id"),
        attempt_id=require_int(data, "attempt_id"),
        duration_ms=require_int(data, "duration_ms"),
        dispatched=require_bool(data, "dispatched"),
        timestamp=require_str(data, "timestamp"),
    )


def encode_suppressed_dispatch_record(record: SuppressedDispatchRecordDict) -> JSONObject:
    """Encode a suppressed-dispatch tally row to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "command_name": record["command_name"],
        "target_x": record["target_x"],
        "target_y": record["target_y"],
        "predicted_error_code": record["predicted_error_code"],
        "count": record["count"],
    }


def decode_suppressed_dispatch_record(data: JSONObject) -> SuppressedDispatchRecordDict:
    """Decode a suppressed-dispatch tally row from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record.
    """
    return SuppressedDispatchRecordDict(
        command_name=require_str(data, "command_name"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        predicted_error_code=require_int(data, "predicted_error_code"),
        count=require_int(data, "count"),
    )


def encode_displaced_teleport_record(record: DisplacedTeleportRecordDict) -> JSONObject:
    """Encode a displaced-teleport tally row to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "requested_x": record["requested_x"],
        "requested_y": record["requested_y"],
        "count": record["count"],
        "max_displacement": record["max_displacement"],
    }


def decode_displaced_teleport_record(data: JSONObject) -> DisplacedTeleportRecordDict:
    """Decode a displaced-teleport tally row from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record.
    """
    return DisplacedTeleportRecordDict(
        requested_x=require_int(data, "requested_x"),
        requested_y=require_int(data, "requested_y"),
        count=require_int(data, "count"),
        max_displacement=require_int(data, "max_displacement"),
    )


def encode_session_room_record(record: SessionRoomRecordDict) -> JSONObject:
    """Encode a session room record to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "room_id": record["room_id"],
        "field_image": record["field_image"],
        "timestamp": record["timestamp"],
    }


def decode_session_room_record(data: JSONObject) -> SessionRoomRecordDict:
    """Decode a session room record from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record.
    """
    return SessionRoomRecordDict(
        room_id=require_str(data, "room_id"),
        field_image=require_str(data, "field_image"),
        timestamp=require_str(data, "timestamp"),
    )


__all__ = [
    "decode_action_outcome_row",
    "decode_displaced_teleport_record",
    "decode_fuel_target_selection_record",
    "decode_map_open_skipped_record",
    "decode_session_room_record",
    "decode_suppressed_dispatch_record",
    "decode_teleport_attempt_record",
    "encode_action_outcome_row",
    "encode_displaced_teleport_record",
    "encode_fuel_target_selection_record",
    "encode_map_open_skipped_record",
    "encode_session_room_record",
    "encode_suppressed_dispatch_record",
    "encode_teleport_attempt_record",
]
