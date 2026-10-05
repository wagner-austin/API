"""The conformance report's records survive a write and a read, and bad ones are refused."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.analysis.types import SessionSkipReason
from tankpit_bot.validate.conformance_types import (
    ConformanceReportDict,
    DivergenceDict,
    GroupTallyDict,
    ReplaySkipReason,
    SessionResultDict,
    SkippedSessionDict,
    decode_report,
    encode_report,
)


def _report() -> ConformanceReportDict:
    """A report holding one of every record."""
    return ConformanceReportDict(
        sessions=[
            SessionResultDict(
                session="runs/bot/a.capture_session.json",
                field="field01_r.gif",
                ticks=10,
                matched=8,
                unmodelled=1,
            )
        ],
        skipped=[
            SkippedSessionDict(
                session="runs/bot/b.capture_session.json",
                reason=ReplaySkipReason.FIELD_MISSING,
                detail="field99.gif",
            )
        ],
        groups=[GroupTallyDict(commands="teleport", ticks=10, matched=8)],
        divergences=[
            DivergenceDict(
                commands="teleport",
                live=["5A", "3Dself", "landed"],
                sim=["3Dself", "landed"],
                count=2,
                example_session="runs/bot/a.capture_session.json",
                example_timestamp_ms=1234,
            )
        ],
    )


def _through_json(data: JSONObject) -> JSONObject:
    """Serialise and re-read, as a report on disk is."""
    return narrow_json_to_dict(load_json_str(dump_json_str(data)))


def test_report_round_trips_through_json() -> None:
    """Every field written is read back identical, the enum included."""
    report = _report()
    decoded = decode_report(_through_json(encode_report(report)))
    assert decoded == report
    assert decoded["skipped"][0]["reason"] is ReplaySkipReason.FIELD_MISSING


def test_scan_reasons_are_carried_by_value() -> None:
    """The two reasons shared with the archive scan read the same as its own."""
    assert ReplaySkipReason.NO_MAGIC.value == SessionSkipReason.NO_MAGIC.value
    assert ReplaySkipReason.UNFRAMED_PAYLOAD.value == SessionSkipReason.UNFRAMED_PAYLOAD.value


def _empty() -> JSONObject:
    """An encoded report with no records, for one list to be replaced."""
    return {"sessions": [], "skipped": [], "groups": [], "divergences": []}


def test_unknown_skip_reason_is_refused() -> None:
    """A reason outside the closed vocabulary names itself and the code."""
    data = _empty()
    data["skipped"] = [{"session": "s", "reason": "felt_like_it", "detail": ""}]
    with pytest.raises(JSONTypeError, match="CONFORMANCE_SKIP_REASON: 'felt_like_it'"):
        decode_report(data)


def test_non_string_token_is_refused() -> None:
    """A divergence's token list must hold strings only."""
    data = _empty()
    data["divergences"] = [
        {
            "commands": "radar",
            "live": ["5A", 7],
            "sim": [],
            "count": 1,
            "example_session": "s",
            "example_timestamp_ms": 1,
        }
    ]
    with pytest.raises(JSONTypeError):
        decode_report(data)


def test_missing_field_is_refused() -> None:
    """A session record without its tick count does not decode."""
    data = _empty()
    data["sessions"] = [{"session": "s", "field": "f", "matched": 0, "unmodelled": 0}]
    with pytest.raises(JSONTypeError):
        decode_report(data)


def test_an_empty_report_decodes_empty() -> None:
    """No records is a valid report."""
    assert decode_report(_empty()) == ConformanceReportDict(
        sessions=[], skipped=[], groups=[], divergences=[]
    )
