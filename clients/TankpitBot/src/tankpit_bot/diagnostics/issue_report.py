"""Build :class:`IssueReportDict` from a JSONL event artifact.

This module is pure: it takes a path to a JSONL events stream (the
artifact :mod:`tankpit_bot.runtime_logging` writes during ``make bot``
or ``make <name>-probe`` runs), parses every event through the real
:func:`tankpit_bot.runtime_logging.decode_runtime_event_record`
decoder, classifies the relevant DIAGNOSTIC / WIRE / WIRE_COMPLETE
events into structured records, and returns the aggregate report.

Categorization rules:

* ``teleport_attempt`` DIAGNOSTIC events become
  :class:`TeleportAttemptRecordDict` rows. Success vs failure is
  decided by the ``status`` field -- the only status strings considered
  successful are ``landed_exact`` and ``landed_inexact``.
* ``map_open_skipped_already_open`` DIAGNOSTIC events become
  :class:`MapOpenSkippedRecordDict` rows.
* ``fuel_target_selection`` DIAGNOSTIC events become
  :class:`FuelTargetSelectionRecordDict` rows.
* ``session_room_joined`` DIAGNOSTIC events populate the report's
  ``session_room`` field; if more than one is present the LAST one
  wins so reconfigured sessions are reflected.
* ``WIRE_COMPLETE`` channel events become
  :class:`WireCompleteRecordDict` rows.
* ``WIRE`` channel events whose message starts with ``map_open`` count
  toward the ``map_open_dispatches`` total.
* ``STATE`` transitions, ``shoot(`` WIRE dispatches,
  ``tank_deactivated`` / ``self_alignment_sample`` / ``fuel_dot_hop``
  DIAGNOSTIC events feed the per-run :class:`SessionScorecardDict`.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from platform_core.logging import get_logger
from typing_extensions import TypedDict

from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.diagnostics.issue_report_types import (
    FuelTargetSelectionRecordDict,
    IssueReportDict,
    MapOpenSkippedRecordDict,
    SessionRoomRecordDict,
    SessionScorecardDict,
    StateBudgetRecordDict,
    TargetedTeleportRecordDict,
    TeleportAttemptRecordDict,
    WireCompleteRecordDict,
    make_unsampled_inventory_counts,
    make_zero_inventory_counts,
)
from tankpit_bot.runtime_logging import (
    RuntimeEventRecordDict,
    require_int_field,
    require_str_field,
)

log = get_logger(__name__)


_LANDED_STATUSES: frozenset[str] = frozenset({"landed_exact", "landed_inexact"})


def _require_bool_field(
    fields: dict[str, str | int | float | bool],
    key: str,
) -> bool:
    """Extract a required bool-valued structured field.

    Args:
        fields: Decoded structured payload from a runtime event record.
        key: Field name to extract.

    Returns:
        Validated bool value.

    Raises:
        KeyError: When ``key`` is absent from ``fields``.
        TypeError: When the field is not a bool.
    """
    if key not in fields:
        raise KeyError(f"runtime field {key!r} is required")
    value = fields[key]
    if not isinstance(value, bool):
        raise TypeError(f"runtime field {key!r} must be bool, got {type(value).__name__}")
    return value


def _classify_teleport_attempt(record: RuntimeEventRecordDict) -> TeleportAttemptRecordDict:
    """Build a typed teleport-attempt row from a DIAGNOSTIC event.

    Args:
        record: Decoded event record whose ``diagnostic_kind`` is
            ``teleport_attempt``.

    Returns:
        Strict-typed teleport attempt row.
    """
    fields = record["fields"]
    return TeleportAttemptRecordDict(
        target_x=require_int_field(fields, "target_x"),
        target_y=require_int_field(fields, "target_y"),
        teleport_cycle_id=require_int_field(fields, "teleport_cycle_id"),
        status=require_str_field(fields, "status"),
        timestamp=record["timestamp"],
        sent_window=require_str_field(fields, "sent_window"),
        received_window=require_str_field(fields, "received_window"),
        page_snapshot_count=require_int_field(fields, "page_snapshot_count"),
    )


def _classify_map_open_skipped(record: RuntimeEventRecordDict) -> MapOpenSkippedRecordDict:
    """Build a typed map_open_skipped row from a DIAGNOSTIC event.

    Args:
        record: Decoded event record whose ``diagnostic_kind`` is
            ``map_open_skipped_already_open``.

    Returns:
        Strict-typed map_open_skipped row.
    """
    fields = record["fields"]
    return MapOpenSkippedRecordDict(
        origin=require_str_field(fields, "origin"),
        timestamp=record["timestamp"],
    )


def _classify_fuel_target_selection(
    record: RuntimeEventRecordDict,
) -> FuelTargetSelectionRecordDict:
    """Build a typed fuel_target_selection row from a DIAGNOSTIC event.

    Args:
        record: Decoded event record whose ``diagnostic_kind`` is
            ``fuel_target_selection``.

    Returns:
        Strict-typed fuel target selection row.
    """
    fields = record["fields"]
    return FuelTargetSelectionRecordDict(
        radar_cycle_id=require_int_field(fields, "radar_cycle_id"),
        target_present=_require_bool_field(fields, "target_present"),
        target_x=require_int_field(fields, "target_x"),
        target_y=require_int_field(fields, "target_y"),
        summary=require_str_field(fields, "summary"),
        decision_basis=require_str_field(fields, "decision_basis"),
        timestamp=record["timestamp"],
    )


def _classify_session_room(record: RuntimeEventRecordDict) -> SessionRoomRecordDict:
    """Build a typed session_room row from a DIAGNOSTIC event.

    Args:
        record: Decoded event record whose ``diagnostic_kind`` is
            ``session_room_joined``.

    Returns:
        Strict-typed session room row.
    """
    fields = record["fields"]
    return SessionRoomRecordDict(
        room_id=require_str_field(fields, "room_id"),
        field_image=require_str_field(fields, "field_image"),
        timestamp=record["timestamp"],
    )


def _classify_wire_complete(record: RuntimeEventRecordDict) -> WireCompleteRecordDict:
    """Build a typed WIRE_COMPLETE row.

    Args:
        record: Decoded event record on the ``WIRE_COMPLETE`` channel.

    Returns:
        Strict-typed wire complete row.
    """
    fields = record["fields"]
    return WireCompleteRecordDict(
        action_kind=require_str_field(fields, "action_kind"),
        duration_ms=require_int_field(fields, "duration_ms"),
        signal=require_str_field(fields, "signal"),
        timestamp=record["timestamp"],
    )


class _ReportAccumulatorDict(TypedDict):
    """Mutable scratch space used during :func:`build_issue_report`.

    Attributes:
        teleport_attempts: Teleport attempts observed so far.
        map_open_skipped: ``map_open_skipped_already_open`` events observed so far.
        fuel_target_selections: Fuel target selections observed so far.
        wire_completes: WIRE_COMPLETE events observed so far.
        session_room: Last ``session_room_joined`` event seen, or None.
        mode: Latest non-empty mode string observed.
        map_open_dispatches: Count of ``WIRE`` events whose message
            starts with ``map_open``.
        recovery_boxed_in_count: Count of ``recovery_boxed_in`` events.
        state_transitions: ``(timestamp, message)`` pairs from the
            ``STATE`` channel, in stream order.
        kills: Count of ``tank_deactivated`` events.
        shots: Count of ``WIRE`` events whose message starts with
            ``shoot(``.
        fuel_samples: ``belief_fuel`` values from every
            ``self_alignment_sample`` event, in stream order.
        dot_hops: Every ``fuel_dot_hop`` event, in stream order.
        first_timestamp: Timestamp of the first record, or ``""``.
        last_timestamp: Timestamp of the last record, or ``""``.
    """

    teleport_attempts: list[TeleportAttemptRecordDict]
    map_open_skipped: list[MapOpenSkippedRecordDict]
    fuel_target_selections: list[FuelTargetSelectionRecordDict]
    wire_completes: list[WireCompleteRecordDict]
    session_room: SessionRoomRecordDict | None
    mode: str
    map_open_dispatches: int
    recovery_boxed_in_count: int
    state_transitions: list[tuple[str, str]]
    kills: int
    shots: int
    fuel_samples: list[int]
    dot_hops: list[TargetedTeleportRecordDict]
    first_timestamp: str
    last_timestamp: str


def _new_accumulator() -> _ReportAccumulatorDict:
    """Return a fresh :class:`_ReportAccumulatorDict` with empty collections."""
    return _ReportAccumulatorDict(
        teleport_attempts=[],
        map_open_skipped=[],
        fuel_target_selections=[],
        wire_completes=[],
        session_room=None,
        mode="unconfigured",
        map_open_dispatches=0,
        recovery_boxed_in_count=0,
        state_transitions=[],
        kills=0,
        shots=0,
        fuel_samples=[],
        dot_hops=[],
        first_timestamp="",
        last_timestamp="",
    )


def _classify_diagnostic_record(
    record: RuntimeEventRecordDict,
    accumulator: _ReportAccumulatorDict,
) -> None:
    """Route one ``DIAGNOSTIC`` channel record into the right bucket."""
    kind = record["fields"].get("diagnostic_kind")
    if not isinstance(kind, str):
        return
    if kind == "teleport_attempt":
        accumulator["teleport_attempts"].append(_classify_teleport_attempt(record))
    elif kind == "map_open_skipped_already_open":
        accumulator["map_open_skipped"].append(_classify_map_open_skipped(record))
    elif kind == "fuel_target_selection":
        accumulator["fuel_target_selections"].append(_classify_fuel_target_selection(record))
    elif kind == "session_room_joined":
        accumulator["session_room"] = _classify_session_room(record)
    elif kind == "recovery_boxed_in":
        accumulator["recovery_boxed_in_count"] += 1
    elif kind == "tank_deactivated":
        accumulator["kills"] += 1
    elif kind == "self_alignment_sample":
        accumulator["fuel_samples"].append(require_int_field(record["fields"], "belief_fuel"))
    elif kind == "fuel_dot_hop":
        accumulator["dot_hops"].append(
            TargetedTeleportRecordDict(
                target_x=require_int_field(record["fields"], "target_x"),
                target_y=require_int_field(record["fields"], "target_y"),
                fuel=require_int_field(record["fields"], "fuel"),
                timestamp=record["timestamp"],
            )
        )


def _route_record(
    record: RuntimeEventRecordDict,
    accumulator: _ReportAccumulatorDict,
) -> None:
    """Route a decoded event record into the report accumulator."""
    if not accumulator["first_timestamp"]:
        accumulator["first_timestamp"] = record["timestamp"]
    accumulator["last_timestamp"] = record["timestamp"]
    if record["mode"]:
        accumulator["mode"] = record["mode"]
    channel = record["channel"]
    if channel == "WIRE_COMPLETE":
        accumulator["wire_completes"].append(_classify_wire_complete(record))
    elif channel == "WIRE":
        if record["message"].startswith("map_open"):
            accumulator["map_open_dispatches"] += 1
        if record["message"].startswith("shoot("):
            accumulator["shots"] += 1
    elif channel == "STATE":
        accumulator["state_transitions"].append((record["timestamp"], record["message"]))
    elif channel == "DIAGNOSTIC":
        _classify_diagnostic_record(record, accumulator)


def _budget_sort_key(record: StateBudgetRecordDict) -> tuple[int, str]:
    """Sort key for the state budget: descending seconds, then name.

    Args:
        record: State budget record to key.

    Returns:
        Tuple of ``(-seconds, state)``.
    """
    return (-record["seconds"], record["state"])


def _build_state_budget(transitions: list[tuple[str, str]]) -> list[StateBudgetRecordDict]:
    """Sum seconds spent in each bot state from STATE-channel transitions.

    The interval between consecutive ``A -> B`` transitions is credited
    to the EARLIER transition's destination -- the state the bot was
    actually in during that interval. Non-transition STATE lines (the
    initial bare state announcement) carry no interval and are skipped.

    Args:
        transitions: ``(timestamp, message)`` pairs in stream order.

    Returns:
        Per-state totals sorted by descending seconds then state name.
    """
    totals: Counter[str] = Counter()
    previous_state = ""
    previous_moment: datetime | None = None
    for timestamp, message in transitions:
        if " -> " not in message:
            continue
        _, _, destination = message.partition(" -> ")
        moment = datetime.fromisoformat(timestamp)
        if previous_moment is not None:
            totals[previous_state] += int((moment - previous_moment).total_seconds())
        previous_state = destination
        previous_moment = moment
    records = [
        StateBudgetRecordDict(state=state, seconds=seconds) for state, seconds in totals.items()
    ]
    records.sort(key=_budget_sort_key)
    return records


def _build_session_scorecard(accumulator: _ReportAccumulatorDict) -> SessionScorecardDict:
    """Distill the per-run outcome scorecard from the accumulator.

    Args:
        accumulator: Fully routed event accumulator.

    Returns:
        Session scorecard.
    """
    duration_seconds = 0
    if accumulator["first_timestamp"] and accumulator["last_timestamp"]:
        first = datetime.fromisoformat(accumulator["first_timestamp"])
        last = datetime.fromisoformat(accumulator["last_timestamp"])
        duration_seconds = int((last - first).total_seconds())
    fuel_samples = accumulator["fuel_samples"]
    dot_hops = accumulator["dot_hops"]
    hop_counts = Counter((hop["target_x"], hop["target_y"]) for hop in dot_hops)
    return SessionScorecardDict(
        duration_seconds=duration_seconds,
        state_budget=_build_state_budget(accumulator["state_transitions"]),
        kills=accumulator["kills"],
        shots=accumulator["shots"],
        fuel_min=min(fuel_samples) if fuel_samples else -1,
        fuel_last=fuel_samples[-1] if fuel_samples else -1,
        fuel_sample_count=len(fuel_samples),
        dot_hops=dot_hops,
        dot_hop_distinct_targets=len(hop_counts),
        dot_hop_max_repeats=max(hop_counts.values()) if hop_counts else 0,
        inventory_first=make_unsampled_inventory_counts(),
        inventory_last=make_unsampled_inventory_counts(),
        inventory_sample_count=0,
        equipment_gain_events=0,
        equipment_gained=make_zero_inventory_counts(),
        scans_extra=0,
        scans_builtin=0,
        equipment_approaches=[],
        equipment_approach_distinct_targets=0,
        equipment_approach_max_repeats=0,
    )


def build_issue_report(source_path: Path) -> IssueReportDict:
    """Build an :class:`IssueReportDict` from a JSONL events artifact.

    Args:
        source_path: Path to a runtime events JSONL artifact.

    Returns:
        Aggregated issue report.

    Raises:
        FileNotFoundError: When ``source_path`` does not exist on disk.
        Exception: Any decode error from
            :func:`tankpit_bot.runtime_logging.decode_runtime_event_record`
            is propagated unchanged so malformed artifacts are surfaced
            instead of silently dropped.
    """
    records = load_event_records(source_path)
    accumulator = _new_accumulator()
    for record in records:
        _route_record(record, accumulator)

    teleport_attempts = accumulator["teleport_attempts"]
    fuel_target_selections = accumulator["fuel_target_selections"]
    wire_completes = accumulator["wire_completes"]
    teleport_success = sum(1 for a in teleport_attempts if a["status"] in _LANDED_STATUSES)
    fuel_selected = sum(1 for s in fuel_target_selections if s["target_present"])
    map_open_completions = sum(1 for w in wire_completes if w["action_kind"] == "map_open")

    return IssueReportDict(
        source_path=str(source_path),
        mode=accumulator["mode"],
        event_count=len(records),
        session_room=accumulator["session_room"],
        teleport_attempts=teleport_attempts,
        map_open_skipped=accumulator["map_open_skipped"],
        fuel_target_selections=fuel_target_selections,
        wire_completes=wire_completes,
        teleport_success_count=teleport_success,
        teleport_failure_count=len(teleport_attempts) - teleport_success,
        fuel_selected_count=fuel_selected,
        fuel_rejected_count=len(fuel_target_selections) - fuel_selected,
        map_open_dispatches=accumulator["map_open_dispatches"],
        map_open_completions=map_open_completions,
        scorecard=_build_session_scorecard(accumulator),
        recovery_boxed_in_count=accumulator["recovery_boxed_in_count"],
    )


__all__ = [
    "build_issue_report",
]
