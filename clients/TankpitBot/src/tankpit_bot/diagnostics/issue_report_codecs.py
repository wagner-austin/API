"""Encode/decode functions for issue report TypedDicts."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot.diagnostics.issue_report_types import (
    FuelTargetSelectionRecordDict,
    InventoryCountsDict,
    IssueReportDict,
    MapOpenSkippedRecordDict,
    SessionRoomRecordDict,
    SessionScorecardDict,
    StateBudgetRecordDict,
    TargetedTeleportRecordDict,
    TeleportAttemptRecordDict,
    WireCompleteRecordDict,
)


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


def encode_wire_complete_record(record: WireCompleteRecordDict) -> JSONObject:
    """Encode a wire complete record to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "action_kind": record["action_kind"],
        "duration_ms": record["duration_ms"],
        "signal": record["signal"],
        "timestamp": record["timestamp"],
    }


def decode_wire_complete_record(data: JSONObject) -> WireCompleteRecordDict:
    """Decode a wire complete record from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record.
    """
    return WireCompleteRecordDict(
        action_kind=require_str(data, "action_kind"),
        duration_ms=require_int(data, "duration_ms"),
        signal=require_str(data, "signal"),
        timestamp=require_str(data, "timestamp"),
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


def encode_state_budget_record(record: StateBudgetRecordDict) -> JSONObject:
    """Encode a state budget record to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {"state": record["state"], "seconds": record["seconds"]}


def decode_state_budget_record(data: JSONObject) -> StateBudgetRecordDict:
    """Decode a state budget record from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record.
    """
    return StateBudgetRecordDict(
        state=require_str(data, "state"),
        seconds=require_int(data, "seconds"),
    )


def encode_targeted_teleport_record(record: TargetedTeleportRecordDict) -> JSONObject:
    """Encode a targeted teleport record to JSON.

    Args:
        record: Record to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "target_x": record["target_x"],
        "target_y": record["target_y"],
        "fuel": record["fuel"],
        "timestamp": record["timestamp"],
    }


def decode_targeted_teleport_record(data: JSONObject) -> TargetedTeleportRecordDict:
    """Decode a targeted teleport record from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated record.
    """
    return TargetedTeleportRecordDict(
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        fuel=require_int(data, "fuel"),
        timestamp=require_str(data, "timestamp"),
    )


def encode_inventory_counts(counts: InventoryCountsDict) -> JSONObject:
    """Encode inventory counts to JSON.

    Args:
        counts: Counts to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "armor": counts["armor"],
        "dual": counts["dual"],
        "missile": counts["missile"],
        "homing": counts["homing"],
        "radar": counts["radar"],
    }


def decode_inventory_counts(data: JSONObject) -> InventoryCountsDict:
    """Decode inventory counts from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated counts.
    """
    return InventoryCountsDict(
        armor=require_int(data, "armor"),
        dual=require_int(data, "dual"),
        missile=require_int(data, "missile"),
        homing=require_int(data, "homing"),
        radar=require_int(data, "radar"),
    )


def encode_session_scorecard(scorecard: SessionScorecardDict) -> JSONObject:
    """Encode a session scorecard to JSON.

    Args:
        scorecard: Scorecard to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "duration_seconds": scorecard["duration_seconds"],
        "state_budget": [encode_state_budget_record(r) for r in scorecard["state_budget"]],
        "kills": scorecard["kills"],
        "shots": scorecard["shots"],
        "combat_misses": scorecard["combat_misses"],
        "combat_ghosts_blocked": scorecard["combat_ghosts_blocked"],
        "combat_stale_positions_blocked": scorecard["combat_stale_positions_blocked"],
        "tank_damage_changes": scorecard["tank_damage_changes"],
        "fuel_min": scorecard["fuel_min"],
        "fuel_last": scorecard["fuel_last"],
        "fuel_sample_count": scorecard["fuel_sample_count"],
        "inventory_first": encode_inventory_counts(scorecard["inventory_first"]),
        "inventory_last": encode_inventory_counts(scorecard["inventory_last"]),
        "inventory_sample_count": scorecard["inventory_sample_count"],
        "equipment_gain_events": scorecard["equipment_gain_events"],
        "equipment_gained": encode_inventory_counts(scorecard["equipment_gained"]),
        "scans_extra": scorecard["scans_extra"],
        "scans_builtin": scorecard["scans_builtin"],
        "equipment_approaches": [
            encode_targeted_teleport_record(r) for r in scorecard["equipment_approaches"]
        ],
        "equipment_approach_distinct_targets": scorecard["equipment_approach_distinct_targets"],
        "equipment_approach_max_repeats": scorecard["equipment_approach_max_repeats"],
        "career_destroyed_last": scorecard["career_destroyed_last"],
        "career_deactivated_last": scorecard["career_deactivated_last"],
        "career_score_last": scorecard["career_score_last"],
        "career_playtime_seconds_last": scorecard["career_playtime_seconds_last"],
        "container_pickups_full": scorecard["container_pickups_full"],
        "container_pickups_partial": scorecard["container_pickups_partial"],
    }


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


def decode_session_scorecard(data: JSONObject) -> SessionScorecardDict:
    """Decode a session scorecard from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated scorecard.
    """
    return SessionScorecardDict(
        duration_seconds=require_int(data, "duration_seconds"),
        state_budget=[
            decode_state_budget_record(item) for item in _require_object_list(data, "state_budget")
        ],
        kills=require_int(data, "kills"),
        shots=require_int(data, "shots"),
        combat_misses=require_int(data, "combat_misses"),
        combat_ghosts_blocked=require_int(data, "combat_ghosts_blocked"),
        combat_stale_positions_blocked=require_int(data, "combat_stale_positions_blocked"),
        tank_damage_changes=require_int(data, "tank_damage_changes"),
        fuel_min=require_int(data, "fuel_min"),
        fuel_last=require_int(data, "fuel_last"),
        fuel_sample_count=require_int(data, "fuel_sample_count"),
        inventory_first=decode_inventory_counts(_require_object(data, "inventory_first")),
        inventory_last=decode_inventory_counts(_require_object(data, "inventory_last")),
        inventory_sample_count=require_int(data, "inventory_sample_count"),
        equipment_gain_events=require_int(data, "equipment_gain_events"),
        equipment_gained=decode_inventory_counts(_require_object(data, "equipment_gained")),
        scans_extra=require_int(data, "scans_extra"),
        scans_builtin=require_int(data, "scans_builtin"),
        equipment_approaches=[
            decode_targeted_teleport_record(item)
            for item in _require_object_list(data, "equipment_approaches")
        ],
        equipment_approach_distinct_targets=require_int(
            data,
            "equipment_approach_distinct_targets",
        ),
        equipment_approach_max_repeats=require_int(data, "equipment_approach_max_repeats"),
        career_destroyed_last=(
            require_int(data, "career_destroyed_last") if "career_destroyed_last" in data else -1
        ),
        career_deactivated_last=(
            require_int(data, "career_deactivated_last")
            if "career_deactivated_last" in data
            else -1
        ),
        career_score_last=(
            require_int(data, "career_score_last") if "career_score_last" in data else -1
        ),
        career_playtime_seconds_last=(
            require_int(data, "career_playtime_seconds_last")
            if "career_playtime_seconds_last" in data
            else -1
        ),
        container_pickups_full=(
            require_int(data, "container_pickups_full") if "container_pickups_full" in data else 0
        ),
        container_pickups_partial=(
            require_int(data, "container_pickups_partial")
            if "container_pickups_partial" in data
            else 0
        ),
    )


def encode_issue_report(report: IssueReportDict) -> JSONObject:
    """Encode a complete issue report to JSON.

    Args:
        report: Report to encode.

    Returns:
        JSON-compatible representation.
    """
    session_room: JSONValue = (
        None
        if report["session_room"] is None
        else encode_session_room_record(report["session_room"])
    )
    return {
        "source_path": report["source_path"],
        "mode": report["mode"],
        "event_count": report["event_count"],
        "session_room": session_room,
        "teleport_attempts": [
            encode_teleport_attempt_record(r) for r in report["teleport_attempts"]
        ],
        "map_open_skipped": [encode_map_open_skipped_record(r) for r in report["map_open_skipped"]],
        "fuel_target_selections": [
            encode_fuel_target_selection_record(r) for r in report["fuel_target_selections"]
        ],
        "wire_completes": [encode_wire_complete_record(r) for r in report["wire_completes"]],
        "teleport_success_count": report["teleport_success_count"],
        "teleport_failure_count": report["teleport_failure_count"],
        "fuel_selected_count": report["fuel_selected_count"],
        "fuel_rejected_count": report["fuel_rejected_count"],
        "map_open_dispatches": report["map_open_dispatches"],
        "map_open_completions": report["map_open_completions"],
        "scorecard": encode_session_scorecard(report["scorecard"]),
        "recovery_boxed_in_count": report["recovery_boxed_in_count"],
    }


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


def decode_issue_report(data: JSONObject) -> IssueReportDict:
    """Decode an issue report from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated report.
    """
    session_raw = _require_object_or_none(data, "session_room")
    session = None if session_raw is None else decode_session_room_record(session_raw)
    return IssueReportDict(
        source_path=require_str(data, "source_path"),
        mode=require_str(data, "mode"),
        event_count=require_int(data, "event_count"),
        session_room=session,
        teleport_attempts=[
            decode_teleport_attempt_record(item)
            for item in _require_object_list(data, "teleport_attempts")
        ],
        map_open_skipped=[
            decode_map_open_skipped_record(item)
            for item in _require_object_list(data, "map_open_skipped")
        ],
        fuel_target_selections=[
            decode_fuel_target_selection_record(item)
            for item in _require_object_list(data, "fuel_target_selections")
        ],
        wire_completes=[
            decode_wire_complete_record(item)
            for item in _require_object_list(data, "wire_completes")
        ],
        teleport_success_count=require_int(data, "teleport_success_count"),
        teleport_failure_count=require_int(data, "teleport_failure_count"),
        fuel_selected_count=require_int(data, "fuel_selected_count"),
        fuel_rejected_count=require_int(data, "fuel_rejected_count"),
        map_open_dispatches=require_int(data, "map_open_dispatches"),
        map_open_completions=require_int(data, "map_open_completions"),
        scorecard=decode_session_scorecard(_require_object(data, "scorecard")),
        recovery_boxed_in_count=require_int(data, "recovery_boxed_in_count"),
    )


__all__ = [
    "decode_fuel_target_selection_record",
    "decode_inventory_counts",
    "decode_issue_report",
    "decode_map_open_skipped_record",
    "decode_session_room_record",
    "decode_session_scorecard",
    "decode_state_budget_record",
    "decode_targeted_teleport_record",
    "decode_teleport_attempt_record",
    "decode_wire_complete_record",
    "encode_fuel_target_selection_record",
    "encode_inventory_counts",
    "encode_issue_report",
    "encode_map_open_skipped_record",
    "encode_session_room_record",
    "encode_session_scorecard",
    "encode_state_budget_record",
    "encode_targeted_teleport_record",
    "encode_teleport_attempt_record",
    "encode_wire_complete_record",
]
