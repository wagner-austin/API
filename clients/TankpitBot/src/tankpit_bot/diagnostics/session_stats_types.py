"""Typed rows for the cross-session stats report."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict


class SessionStatsRowDict(TypedDict):
    """Per-run statistics derived from one events artifact.

    Attributes:
        run_id: Artifact identifier (``bot-YYYYMMDD-HHMMSS``).
        started: Timestamp of the first event in the run.
        duration_s: Whole seconds between the first and last event.
        events: Total decoded event records.
        kills: Own kills decoded from the wire 0x41 Deactivation.
        teleports_ok: Teleport attempts that landed.
        teleports_failed: Teleport attempts that did not land.
        shots: Shoot commands dispatched on the wire.
        pickups: Pickup commands dispatched on the wire.
        stalls: Actions cleared by the stall timeout.
    """

    run_id: str
    started: str
    duration_s: int
    events: int
    kills: int
    teleports_ok: int
    teleports_failed: int
    shots: int
    pickups: int
    stalls: int


class SessionStatsReportDict(TypedDict):
    """Cross-session stats report over every events artifact in a runs dir.

    Attributes:
        runs_dir: Directory the artifacts were swept from.
        rows: Per-run statistics in chronological (filename) order.
        totals: Aggregate row summing every numeric column; ``run_id``
            is the literal ``TOTAL``, ``started`` is the earliest run
            start, and ``duration_s`` sums all run durations.
    """

    runs_dir: str
    rows: list[SessionStatsRowDict]
    totals: SessionStatsRowDict


def encode_session_stats_row(row: SessionStatsRowDict) -> JSONObject:
    """Encode a stats row to a JSON-serializable dict.

    Args:
        row: Stats row to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "run_id": row["run_id"],
        "started": row["started"],
        "duration_s": row["duration_s"],
        "events": row["events"],
        "kills": row["kills"],
        "teleports_ok": row["teleports_ok"],
        "teleports_failed": row["teleports_failed"],
        "shots": row["shots"],
        "pickups": row["pickups"],
        "stalls": row["stalls"],
    }


def decode_session_stats_row(data: JSONObject) -> SessionStatsRowDict:
    """Decode a stats row with strict validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated stats row.

    Raises:
        JSONTypeError: If required fields are missing or mistyped.
    """
    return SessionStatsRowDict(
        run_id=require_str(data, "run_id"),
        started=require_str(data, "started"),
        duration_s=require_int(data, "duration_s"),
        events=require_int(data, "events"),
        kills=require_int(data, "kills"),
        teleports_ok=require_int(data, "teleports_ok"),
        teleports_failed=require_int(data, "teleports_failed"),
        shots=require_int(data, "shots"),
        pickups=require_int(data, "pickups"),
        stalls=require_int(data, "stalls"),
    )


def encode_session_stats_report(report: SessionStatsReportDict) -> JSONObject:
    """Encode a stats report to a JSON-serializable dict.

    Args:
        report: Stats report to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "runs_dir": report["runs_dir"],
        "rows": [encode_session_stats_row(row) for row in report["rows"]],
        "totals": encode_session_stats_row(report["totals"]),
    }


def decode_session_stats_report(data: JSONObject) -> SessionStatsReportDict:
    """Decode a stats report with strict validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated stats report.

    Raises:
        JSONTypeError: If required fields are missing or mistyped.
        ValueError: If any row entry is not an object.
    """
    rows_raw = require_list(data, "rows")
    rows: list[SessionStatsRowDict] = []
    for index, item in enumerate(rows_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Row at index {index} must be a dict, got {type(item).__name__}")
        rows.append(decode_session_stats_row(item))
    totals_raw = data["totals"]
    if not isinstance(totals_raw, dict):
        raise ValueError(f"totals must be a dict, got {type(totals_raw).__name__}")
    return SessionStatsReportDict(
        runs_dir=require_str(data, "runs_dir"),
        rows=rows,
        totals=decode_session_stats_row(totals_raw),
    )


__all__ = [
    "SessionStatsReportDict",
    "SessionStatsRowDict",
    "decode_session_stats_report",
    "decode_session_stats_row",
    "encode_session_stats_report",
    "encode_session_stats_row",
]
