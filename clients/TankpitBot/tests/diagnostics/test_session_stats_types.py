"""Round-trip and validation tests for session stats TypedDicts."""

from __future__ import annotations

import pytest

from tankpit_bot.diagnostics.session_stats_types import (
    SessionStatsReportDict,
    SessionStatsRowDict,
    decode_session_stats_report,
    decode_session_stats_row,
    encode_session_stats_report,
    encode_session_stats_row,
)


def _row() -> SessionStatsRowDict:
    """Return a fully populated stats row."""
    return SessionStatsRowDict(
        run_id="bot-20260610-100000",
        started="2026-06-10T10:00:00",
        duration_s=480,
        events=2200,
        kills=6,
        teleports_ok=34,
        teleports_failed=0,
        shots=81,
        pickups=33,
        stalls=0,
        feedback_corrections=11,
    )


def test_row_round_trip() -> None:
    """Encode then decode preserves every row field."""
    row = _row()
    assert decode_session_stats_row(encode_session_stats_row(row)) == row


def test_report_round_trip() -> None:
    """Encode then decode preserves rows and totals."""
    report = SessionStatsReportDict(runs_dir="runs/bot", rows=[_row()], totals=_row())
    assert decode_session_stats_report(encode_session_stats_report(report)) == report


def test_decode_report_rejects_non_dict_row() -> None:
    """A non-object row entry is rejected loudly."""
    encoded = encode_session_stats_report(
        SessionStatsReportDict(runs_dir="runs/bot", rows=[_row()], totals=_row())
    )
    encoded["rows"] = ["not-a-dict"]
    with pytest.raises(ValueError, match="Row at index 0"):
        decode_session_stats_report(encoded)


def test_decode_report_rejects_non_dict_totals() -> None:
    """A non-object totals entry is rejected loudly."""
    encoded = encode_session_stats_report(
        SessionStatsReportDict(runs_dir="runs/bot", rows=[], totals=_row())
    )
    encoded["totals"] = 7
    with pytest.raises(ValueError, match="totals must be a dict"):
        decode_session_stats_report(encoded)
