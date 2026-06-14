"""Strict TypedDict payloads for the post-run issue report.

Each section of an :class:`IssueReportDict` is its own TypedDict with
the explicit fields that the report renderer and consumers can rely on.
Every section follows the project's encode / decode / ``require_*``
pattern so the report can be persisted, replayed, and compared across
runs without ambiguity.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict


def _require_object_list(data: JSONObject, key: str) -> list[JSONObject]:
    """Return a required list field whose items are all JSON objects.

    Wraps :func:`platform_core.json_utils.require_list` with the
    per-item ``isinstance(item, dict)`` validation the decoder needs
    before delegating to a typed-record decoder. Raising on any
    non-object element is intentional: the issue report relies on every
    list element decoding into a strict TypedDict, so partial lists are
    rejected rather than truncated.

    Args:
        data: Parent JSON object.
        key: Field name to extract.

    Returns:
        Validated list of :class:`JSONObject` items.

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


class TeleportAttemptRecordDict(TypedDict):
    """One teleport attempt observed in the event stream.

    Attributes:
        target_x: Teleport target X coordinate as recorded in the
            ``teleport_attempt`` diagnostic.
        target_y: Teleport target Y coordinate.
        teleport_cycle_id: Cycle ID stamped on the diagnostic.
        status: Terminal status string (``landed_exact``,
            ``teleport_timeout``, ``map_sync_timeout``, ...).
        timestamp: ISO timestamp from the event record.
        sent_window: Compact text rendering of the sent message window
            covering this attempt (already produced by the action_lab
            diagnostic emitter).
        received_window: Compact text rendering of the received window.
        page_snapshot_count: Number of teleport-phase page snapshots
            captured for the attempt.
    """

    target_x: int
    target_y: int
    teleport_cycle_id: int
    status: str
    timestamp: str
    sent_window: str
    received_window: str
    page_snapshot_count: int


class MapOpenSkippedRecordDict(TypedDict):
    """One ``map_open_skipped_already_open`` event.

    Attributes:
        origin: Code site that emitted the skip
            (``acquisition_phase`` or ``executor.dispatch_command.*``).
        timestamp: ISO timestamp from the event record.
    """

    origin: str
    timestamp: str


class FuelTargetSelectionRecordDict(TypedDict):
    """One ``fuel_target_selection`` event from a probe radar cycle.

    Attributes:
        radar_cycle_id: Radar cycle ID stamped on the diagnostic.
        target_present: Whether a fuel target was selected.
        target_x: Selected target X coordinate (``-1`` when none).
        target_y: Selected target Y coordinate (``-1`` when none).
        summary: Compact ``describe_container_search`` summary string.
        decision_basis: Compact decision-basis breakdown string.
        timestamp: ISO timestamp from the event record.
    """

    radar_cycle_id: int
    target_present: bool
    target_x: int
    target_y: int
    summary: str
    decision_basis: str
    timestamp: str


class WireCompleteRecordDict(TypedDict):
    """One ``WIRE_COMPLETE`` event.

    Attributes:
        action_kind: Kind of action that completed (``map_open``,
            ``move``, ``teleport``, ``collect``, ``scan``).
        duration_ms: Wall-clock milliseconds between dispatch and the
            observed completion signal.
        signal: Authoritative completion signal name.
        timestamp: ISO timestamp from the event record.
    """

    action_kind: str
    duration_ms: int
    signal: str
    timestamp: str


class SessionRoomRecordDict(TypedDict):
    """The single ``session_room_joined`` event for the run.

    Attributes:
        room_id: Room ID joined for the session.
        field_image: Field image name reported by the server, or
            ``unknown`` when the room image cache was empty.
        timestamp: ISO timestamp from the event record.
    """

    room_id: str
    field_image: str
    timestamp: str


class StateBudgetRecordDict(TypedDict):
    """Seconds spent in one bot state across the session.

    Attributes:
        state: Bot state name (``COMBAT``, ``MOVING``, ``IDLE``, ...).
        seconds: Whole seconds attributed to the state, summed across
            every visit (event timestamps have second granularity).
    """

    state: str
    seconds: int


class TargetedTeleportRecordDict(TypedDict):
    """One targeted-teleport DIAGNOSTIC event.

    Shared row shape for the ``fuel_dot_hop`` and
    ``equipment_approach`` diagnostics -- both record a deliberate
    teleport at a known coordinate with the fuel level at dispatch.

    Attributes:
        target_x: Target X coordinate.
        target_y: Target Y coordinate.
        fuel: Fuel level when the teleport was planned.
        timestamp: ISO timestamp from the event record.
    """

    target_x: int
    target_y: int
    fuel: int
    timestamp: str


class InventoryCountsDict(TypedDict):
    """Absolute counts for the five inventory item types.

    Attributes:
        armor: Armor shield count.
        dual: Dual shot count.
        missile: Missile shot count.
        homing: Homing shot count.
        radar: Extra radar count.
    """

    armor: int
    dual: int
    missile: int
    homing: int
    radar: int


def make_zero_inventory_counts() -> InventoryCountsDict:
    """Return inventory counts with every item at zero.

    Returns:
        All-zero inventory counts, used as the gain-total accumulator
        seed.
    """
    return InventoryCountsDict(armor=0, dual=0, missile=0, homing=0, radar=0)


def make_unsampled_inventory_counts() -> InventoryCountsDict:
    """Return the sentinel inventory counts for a run with no samples.

    Returns:
        Inventory counts with every item at ``-1``, mirroring the
        ``fuel_min == -1`` no-samples convention.
    """
    return InventoryCountsDict(armor=-1, dual=-1, missile=-1, homing=-1, radar=-1)


class SessionScorecardDict(TypedDict):
    """Per-run outcome scorecard distilled from the event stream.

    This is the audit every live run gets compared on: where the time
    went, what combat produced, how low fuel dipped, how the inventory
    moved, what each radar press actually consumed, and whether the
    dot-atlas refuels or equipment approaches show pathological
    repetition (the orbit class of bug from live runs 20260612-062453
    and 20260612-071918).

    Attributes:
        duration_seconds: Whole seconds between the first and last
            event record.
        state_budget: Seconds per bot state, sorted by descending
            seconds then state name.
        kills: Count of ``tank_deactivated`` DIAGNOSTIC events.
        shots: Count of ``WIRE`` events whose message starts with
            ``shoot(``.
        fuel_min: Lowest ``belief_fuel`` across
            ``self_alignment_sample`` events, or ``-1`` with no samples.
        fuel_last: Final ``belief_fuel`` sample, or ``-1`` with no
            samples.
        fuel_sample_count: Number of fuel samples observed.
        dot_hops: Every ``fuel_dot_hop`` event in order.
        dot_hop_distinct_targets: Number of distinct dot coordinates
            targeted.
        dot_hop_max_repeats: Highest event count for any single dot
            coordinate, ``0`` with no hops.
        inventory_first: First ``inventory_sample`` counts, or the
            all ``-1`` sentinel with no samples.
        inventory_last: Final ``inventory_sample`` counts, or the
            all ``-1`` sentinel with no samples.
        inventory_sample_count: Number of inventory samples observed.
        equipment_gain_events: Count of ``equipment_gain`` events
            (0x67 messages -- one per equipment container collected).
        equipment_gained: Per-type totals summed across every
            ``equipment_gain`` event.
        scans_extra: Radar dispatches that consumed an extra radar.
        scans_builtin: Radar dispatches that used the free 5x5 scan.
        equipment_approaches: Every ``equipment_approach`` event in
            order.
        equipment_approach_distinct_targets: Number of distinct
            equipment coordinates teleport-approached.
        equipment_approach_max_repeats: Highest event count for any
            single equipment coordinate, ``0`` with no approaches.
    """

    duration_seconds: int
    state_budget: list[StateBudgetRecordDict]
    kills: int
    shots: int
    fuel_min: int
    fuel_last: int
    fuel_sample_count: int
    dot_hops: list[TargetedTeleportRecordDict]
    dot_hop_distinct_targets: int
    dot_hop_max_repeats: int
    inventory_first: InventoryCountsDict
    inventory_last: InventoryCountsDict
    inventory_sample_count: int
    equipment_gain_events: int
    equipment_gained: InventoryCountsDict
    scans_extra: int
    scans_builtin: int
    equipment_approaches: list[TargetedTeleportRecordDict]
    equipment_approach_distinct_targets: int
    equipment_approach_max_repeats: int


class IssueReportDict(TypedDict):
    """Aggregated post-run analysis of a JSONL event artifact.

    Attributes:
        source_path: JSONL path the report was built from.
        mode: Runtime mode string from the events (``bot``, ``sniff``,
            ``probe:<name>``).
        event_count: Total event records in the artifact.
        session_room: Recorded room/field for this session, or ``None``
            when the artifact does not include a ``session_room_joined``
            diagnostic.
        teleport_attempts: Every teleport attempt observed.
        map_open_skipped: Every ``map_open_skipped_already_open`` event.
        fuel_target_selections: Every fuel target selection (selected
            and rejected).
        wire_completes: Every ``WIRE_COMPLETE`` event.
        teleport_success_count: Count of attempts whose status is
            ``landed_exact`` or ``landed_inexact``.
        teleport_failure_count: Count of attempts whose status is not a
            "landed" status (timeouts, etc.).
        fuel_selected_count: Count of fuel target selections where
            ``target_present`` is True.
        fuel_rejected_count: Count of fuel target selections where
            ``target_present`` is False.
        map_open_dispatches: Count of ``WIRE`` events whose first
            message starts with ``map_open`` (i.e. successful sends).
        map_open_completions: Count of ``WIRE_COMPLETE`` events whose
            ``action_kind`` is ``map_open``.
        scorecard: Per-run outcome scorecard (time budget, combat,
            fuel trajectory, dot-hop ledger).
    """

    source_path: str
    mode: str
    event_count: int
    session_room: SessionRoomRecordDict | None
    teleport_attempts: list[TeleportAttemptRecordDict]
    map_open_skipped: list[MapOpenSkippedRecordDict]
    fuel_target_selections: list[FuelTargetSelectionRecordDict]
    wire_completes: list[WireCompleteRecordDict]
    teleport_success_count: int
    teleport_failure_count: int
    fuel_selected_count: int
    fuel_rejected_count: int
    map_open_dispatches: int
    map_open_completions: int
    scorecard: SessionScorecardDict
    recovery_boxed_in_count: int


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
        "fuel_min": scorecard["fuel_min"],
        "fuel_last": scorecard["fuel_last"],
        "fuel_sample_count": scorecard["fuel_sample_count"],
        "dot_hops": [encode_targeted_teleport_record(r) for r in scorecard["dot_hops"]],
        "dot_hop_distinct_targets": scorecard["dot_hop_distinct_targets"],
        "dot_hop_max_repeats": scorecard["dot_hop_max_repeats"],
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
        fuel_min=require_int(data, "fuel_min"),
        fuel_last=require_int(data, "fuel_last"),
        fuel_sample_count=require_int(data, "fuel_sample_count"),
        dot_hops=[
            decode_targeted_teleport_record(item) for item in _require_object_list(data, "dot_hops")
        ],
        dot_hop_distinct_targets=require_int(data, "dot_hop_distinct_targets"),
        dot_hop_max_repeats=require_int(data, "dot_hop_max_repeats"),
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
    "FuelTargetSelectionRecordDict",
    "InventoryCountsDict",
    "IssueReportDict",
    "MapOpenSkippedRecordDict",
    "SessionRoomRecordDict",
    "SessionScorecardDict",
    "StateBudgetRecordDict",
    "TargetedTeleportRecordDict",
    "TeleportAttemptRecordDict",
    "WireCompleteRecordDict",
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
    "make_unsampled_inventory_counts",
    "make_zero_inventory_counts",
]
