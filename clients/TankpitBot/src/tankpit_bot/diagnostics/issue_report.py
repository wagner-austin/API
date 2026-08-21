"""Build :class:`IssueReportDict` from a JSONL event artifact.

This module is pure: it takes a path to a JSONL events stream (the
artifact :mod:`tankpit_bot.runtime_logging` writes during ``make bot``
or ``make <name>-probe`` runs), parses every event through the real
:func:`tankpit_bot.runtime_records.decode_runtime_event_record`
decoder, classifies the relevant DIAGNOSTIC / WIRE
events into structured records, and returns the aggregate report.

Categorization rules:

* ``teleport_attempt`` DIAGNOSTIC events (action-lab probes; the
  live bot records teleports as ``action_outcome`` events) become
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
* ``action_outcome`` DIAGNOSTIC events (the ledger's unified
  per-attempt fabric) become :class:`ActionOutcomeRowDict` rows.
* ``WIRE`` channel events whose message starts with ``map_open`` count
  toward the ``map_open_dispatches`` total.
* ``STATE`` transitions, ``shoot(`` WIRE dispatches, and
  ``tank_deactivated`` / ``self_alignment_sample`` DIAGNOSTIC events
  feed the per-run :class:`SessionScorecardDict`.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger
from typing_extensions import TypedDict

from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.diagnostics.issue_report_types import (
    ActionOutcomeRowDict,
    DisplacedTeleportRecordDict,
    FuelTargetSelectionRecordDict,
    IssueReportDict,
    MapOpenSkippedRecordDict,
    SessionRoomRecordDict,
    SuppressedDispatchRecordDict,
    TeleportAttemptRecordDict,
)
from tankpit_bot.diagnostics.session_scorecard import build_session_scorecard
from tankpit_bot.diagnostics.session_scorecard_accumulator import route_scorecard_record
from tankpit_bot.diagnostics.session_scorecard_types import (
    ScorecardAccumulatorDict,
    new_scorecard_accumulator,
)
from tankpit_bot.runtime_records import (
    RuntimeEventRecordDict,
    require_int_field,
    require_str_field,
)

log = get_logger(__name__)


_LANDED_STATUSES: frozenset[str] = frozenset({"landed_exact", "landed_inexact"})

# WIRE command names (the prefix ``command_sender`` stamps as the wire
# event's ``action_kind``) -> the ledger kind that must eventually record
# the completion. Mirrors the executor's ``_LEDGER_KIND_BY_CMD_TYPE``
# with the wire-side spellings (``scope`` for ``scope_shift``). Names
# absent here (``chat``, ``enter_game``, ``equipment``, ...) are
# fire-and-forget or non-decision sends with no ledger contract.
_WIRE_NAME_TO_LEDGER_KIND: dict[str, str] = {
    "move": "move",
    "pickup_fuel": "collect",
    "pickup_equipment": "collect",
    "teleport": "teleport",
    "shoot": "shoot",
    "radar": "scan",
    "scope": "scope",
    "map_open": "map_open",
}


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


def _classify_action_outcome(record: RuntimeEventRecordDict) -> ActionOutcomeRowDict:
    """Build a typed action-outcome row from a DIAGNOSTIC event.

    Args:
        record: Decoded event record whose ``diagnostic_kind`` is
            ``action_outcome``.

    Returns:
        Strict-typed action outcome row.
    """
    fields = record["fields"]
    outcome = require_str_field(fields, "outcome")
    # Every genuine resolution proves a dispatch (the wire answered);
    # only ``superseded`` rows need the executor's explicit mark, and
    # artifacts predating the mark (2026-08-21) read False there --
    # the honest value: whether those closes dispatched is unknowable.
    dispatched = outcome != "superseded" or fields.get("dispatched") is True
    return ActionOutcomeRowDict(
        action_kind=require_str_field(fields, "action_kind"),
        outcome=outcome,
        event_id=require_int_field(fields, "event_id"),
        attempt_id=require_int_field(fields, "attempt_id"),
        duration_ms=require_int_field(fields, "duration_ms"),
        dispatched=dispatched,
        timestamp=record["timestamp"],
    )


class _ReportAccumulatorDict(TypedDict):
    """Mutable scratch space used during :func:`build_issue_report`.

    Composition: the issue report owns report-only buckets
    (teleport_attempts, fuel_target_selections, etc.) and **delegates
    every scorecard-shaped event** to a nested
    :class:`ScorecardAccumulatorDict` consumed by
    :func:`session_scorecard.build_session_scorecard`. The nested
    accumulator is the single source of truth for combat counters,
    fuel samples, dot hops, career stats, pickup tallies, inventory
    samples, and radar/scan tallies -- both the live in-bot scorecard
    and the post-run issue report read the same data from the same
    accumulator type, populated by the same router.

    Attributes:
        teleport_attempts: Teleport attempts observed so far.
        map_open_skipped: ``map_open_skipped_already_open`` events
            observed so far.
        fuel_target_selections: Fuel target selections observed so far.
        action_outcomes: ``action_outcome`` events observed so far.
        session_room: Last ``session_room_joined`` event seen, or None.
        mode: Latest non-empty mode string observed.
        map_open_dispatches: Count of ``WIRE`` events whose message
            starts with ``map_open``.
        suppressed_tallies: Per ``(command, x, y)`` counts of
            ``dispatch_suppressed`` events, with the last predicted
            0x52 code seen for the key.
        displacement_tallies: Per ``(x, y)`` requested-destination
            counts of ``teleport_displacement`` events, with the
            largest Manhattan bounce seen for the key.
        wire_dispatch_tallies: Per-LEDGER-KIND counts of ``WIRE``
            command sends (wire ``action_kind`` mapped through
            ``_WIRE_NAME_TO_LEDGER_KIND``) — the completion audit's
            dispatch side.
        scorecard: Nested scorecard accumulator. Populated by
            :func:`route_scorecard_record` for every event.
    """

    teleport_attempts: list[TeleportAttemptRecordDict]
    map_open_skipped: list[MapOpenSkippedRecordDict]
    fuel_target_selections: list[FuelTargetSelectionRecordDict]
    action_outcomes: list[ActionOutcomeRowDict]
    session_room: SessionRoomRecordDict | None
    mode: str
    map_open_dispatches: int
    suppressed_tallies: dict[tuple[str, int, int], tuple[int, int]]
    displacement_tallies: dict[tuple[int, int], tuple[int, int]]
    wire_dispatch_tallies: dict[str, int]
    scorecard: ScorecardAccumulatorDict


def _new_accumulator() -> _ReportAccumulatorDict:
    """Return a fresh :class:`_ReportAccumulatorDict` with empty collections."""
    return _ReportAccumulatorDict(
        teleport_attempts=[],
        map_open_skipped=[],
        fuel_target_selections=[],
        action_outcomes=[],
        session_room=None,
        mode="unconfigured",
        map_open_dispatches=0,
        suppressed_tallies={},
        displacement_tallies={},
        wire_dispatch_tallies={},
        scorecard=new_scorecard_accumulator(),
    )


def _classify_diagnostic_record(
    record: RuntimeEventRecordDict,
    accumulator: _ReportAccumulatorDict,
) -> None:
    """Route one ``DIAGNOSTIC`` channel record into report-only buckets.

    Scorecard-shaped diagnostic kinds (combat counters, fuel samples,
    dot hops, career stats, pickup tallies, inventory samples, radar
    dispatches) are NOT handled here -- :func:`_route_record` delegates
    every event to
    :func:`session_scorecard.route_scorecard_record`, which is the
    single source of truth for scorecard accumulation. This function
    handles ONLY kinds that don't appear on the scorecard.
    """
    kind = record["fields"].get("diagnostic_kind")
    if kind == "action_outcome":
        accumulator["action_outcomes"].append(_classify_action_outcome(record))
    elif kind == "teleport_attempt":
        accumulator["teleport_attempts"].append(_classify_teleport_attempt(record))
    elif kind == "map_open_skipped_already_open":
        accumulator["map_open_skipped"].append(_classify_map_open_skipped(record))
    elif kind == "fuel_target_selection":
        accumulator["fuel_target_selections"].append(_classify_fuel_target_selection(record))
    elif kind == "session_room_joined":
        accumulator["session_room"] = _classify_session_room(record)
    elif kind == "dispatch_suppressed":
        fields = record["fields"]
        key = (
            require_str_field(fields, "command_name"),
            require_int_field(fields, "target_x"),
            require_int_field(fields, "target_y"),
        )
        count, _ = accumulator["suppressed_tallies"].get(key, (0, 0))
        accumulator["suppressed_tallies"][key] = (
            count + 1,
            require_int_field(fields, "predicted_error_code"),
        )
    elif kind == "teleport_displacement":
        fields = record["fields"]
        tile = (
            require_int_field(fields, "requested_x"),
            require_int_field(fields, "requested_y"),
        )
        count, worst = accumulator["displacement_tallies"].get(tile, (0, 0))
        accumulator["displacement_tallies"][tile] = (
            count + 1,
            max(worst, require_int_field(fields, "displacement")),
        )


def _route_record(
    record: RuntimeEventRecordDict,
    accumulator: _ReportAccumulatorDict,
) -> None:
    """Route a decoded event record into the report accumulator.

    Two-way split: report-only state advances on this accumulator;
    every event (regardless of channel) is also forwarded to the
    nested scorecard accumulator via
    :func:`session_scorecard.route_scorecard_record` so the scorecard
    sees the same stream the report does.
    """
    if record["mode"]:
        accumulator["mode"] = record["mode"]
    channel = record["channel"]
    if channel == "WIRE":
        if record["message"].startswith("map_open"):
            accumulator["map_open_dispatches"] += 1
        wire_name = record["fields"].get("action_kind")
        if isinstance(wire_name, str):
            ledger_kind = _WIRE_NAME_TO_LEDGER_KIND.get(wire_name)
            if ledger_kind is not None:
                tallies = accumulator["wire_dispatch_tallies"]
                tallies[ledger_kind] = tallies.get(ledger_kind, 0) + 1
    elif channel == "DIAGNOSTIC":
        _classify_diagnostic_record(record, accumulator)
    route_scorecard_record(record, accumulator["scorecard"])


# State-budget construction lives in
# :func:`session_scorecard._build_state_budget`, called from
# ``build_session_scorecard``. The issue report just delegates.

# Scorecard construction lives in session_scorecard.build_session_scorecard;
# the issue report just forwards the nested accumulator. That's the
# single source of truth for the SessionScorecardDict shape and every
# field on it -- the report and the in-bot scorecard agree by
# construction.


def _displacement_count_descending(row: DisplacedTeleportRecordDict) -> int:
    """Sort key: highest displacement tally first.

    Args:
        row: One displaced-teleport tally row.

    Returns:
        The negated count.
    """
    return -row["count"]


def _suppression_count_descending(row: SuppressedDispatchRecordDict) -> int:
    """Sort key: highest suppression tally first.

    Args:
        row: One suppressed-dispatch tally row.

    Returns:
        The negated count.
    """
    return -row["count"]


def build_issue_report(source_path: Path) -> IssueReportDict:
    """Build an :class:`IssueReportDict` from a JSONL events artifact.

    Args:
        source_path: Path to a runtime events JSONL artifact.

    Returns:
        Aggregated issue report.

    Raises:
        FileNotFoundError: When ``source_path`` does not exist on disk.
        Exception: Any decode error from
            :func:`tankpit_bot.runtime_records.decode_runtime_event_record`
            is propagated unchanged so malformed artifacts are surfaced
            instead of silently dropped.
    """
    records = load_event_records(source_path)
    accumulator = _new_accumulator()
    for record in records:
        _route_record(record, accumulator)

    teleport_attempts = accumulator["teleport_attempts"]
    fuel_target_selections = accumulator["fuel_target_selections"]
    action_outcomes = accumulator["action_outcomes"]
    teleport_outcomes = [o for o in action_outcomes if o["action_kind"] == "teleport"]
    teleport_success = sum(1 for a in teleport_attempts if a["status"] in _LANDED_STATUSES) + sum(
        1 for o in teleport_outcomes if o["outcome"] in _LANDED_STATUSES
    )
    # ``superseded`` is a re-plan (the decision was replaced before the
    # wire resolved it), not a failed landing -- excluded from failures.
    teleport_superseded = sum(1 for o in teleport_outcomes if o["outcome"] == "superseded")
    teleport_total = len(teleport_attempts) + len(teleport_outcomes) - teleport_superseded
    fuel_selected = sum(1 for s in fuel_target_selections if s["target_present"])
    map_open_completions = sum(
        1
        for o in action_outcomes
        if o["action_kind"] == "map_open" and o["outcome"] == "map_data_processed"
    )
    suppressed_dispatches = sorted(
        (
            SuppressedDispatchRecordDict(
                command_name=command_name,
                target_x=target_x,
                target_y=target_y,
                predicted_error_code=code,
                count=count,
            )
            for (command_name, target_x, target_y), (
                count,
                code,
            ) in accumulator["suppressed_tallies"].items()
        ),
        key=_suppression_count_descending,
    )

    displaced_teleports = sorted(
        (
            DisplacedTeleportRecordDict(
                requested_x=requested_x,
                requested_y=requested_y,
                count=count,
                max_displacement=worst,
            )
            for (requested_x, requested_y), (
                count,
                worst,
            ) in accumulator["displacement_tallies"].items()
        ),
        key=_displacement_count_descending,
    )

    return IssueReportDict(
        source_path=str(source_path),
        mode=accumulator["mode"],
        event_count=len(records),
        session_room=accumulator["session_room"],
        teleport_attempts=teleport_attempts,
        map_open_skipped=accumulator["map_open_skipped"],
        fuel_target_selections=fuel_target_selections,
        action_outcomes=action_outcomes,
        teleport_success_count=teleport_success,
        teleport_failure_count=teleport_total - teleport_success,
        fuel_selected_count=fuel_selected,
        fuel_rejected_count=len(fuel_target_selections) - fuel_selected,
        map_open_dispatches=accumulator["map_open_dispatches"],
        map_open_completions=map_open_completions,
        suppressed_dispatches=suppressed_dispatches,
        displaced_teleports=displaced_teleports,
        wire_dispatches_by_kind=dict(accumulator["wire_dispatch_tallies"]),
        scorecard=build_session_scorecard(accumulator["scorecard"]),
    )


__all__ = [
    "build_issue_report",
]
