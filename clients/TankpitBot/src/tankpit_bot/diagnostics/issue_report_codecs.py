"""Codec for the issue report itself.

The top of the codec family: encodes and decodes the whole report by
composing the record codecs
(:mod:`tankpit_bot.diagnostics.issue_report_codecs_records`) and the
scorecard codecs
(:mod:`tankpit_bot.diagnostics.issue_report_codecs_scorecard`).
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_int,
    require_str,
)

from tankpit_bot.diagnostics.issue_report_codecs_records import (
    _require_object,
    _require_object_list,
    _require_object_or_none,
    decode_action_outcome_row,
    decode_fuel_target_selection_record,
    decode_map_open_skipped_record,
    decode_session_room_record,
    decode_teleport_attempt_record,
    encode_action_outcome_row,
    encode_fuel_target_selection_record,
    encode_map_open_skipped_record,
    encode_session_room_record,
    encode_teleport_attempt_record,
)
from tankpit_bot.diagnostics.issue_report_codecs_scorecard import (
    decode_session_scorecard,
    encode_session_scorecard,
)
from tankpit_bot.diagnostics.issue_report_types import (
    IssueReportDict,
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
        "action_outcomes": [encode_action_outcome_row(r) for r in report["action_outcomes"]],
        "teleport_success_count": report["teleport_success_count"],
        "teleport_failure_count": report["teleport_failure_count"],
        "fuel_selected_count": report["fuel_selected_count"],
        "fuel_rejected_count": report["fuel_rejected_count"],
        "map_open_dispatches": report["map_open_dispatches"],
        "map_open_completions": report["map_open_completions"],
        "scorecard": encode_session_scorecard(report["scorecard"]),
    }


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
        action_outcomes=[
            decode_action_outcome_row(item)
            for item in _require_object_list(data, "action_outcomes")
        ],
        teleport_success_count=require_int(data, "teleport_success_count"),
        teleport_failure_count=require_int(data, "teleport_failure_count"),
        fuel_selected_count=require_int(data, "fuel_selected_count"),
        fuel_rejected_count=require_int(data, "fuel_rejected_count"),
        map_open_dispatches=require_int(data, "map_open_dispatches"),
        map_open_completions=require_int(data, "map_open_completions"),
        scorecard=decode_session_scorecard(_require_object(data, "scorecard")),
    )


__all__ = [
    "decode_issue_report",
    "encode_issue_report",
]
