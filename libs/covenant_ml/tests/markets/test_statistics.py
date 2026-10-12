"""Economic totals and dependence-aware intervals from actual replay results."""

from __future__ import annotations

from datetime import date

import pytest
from platform_core.json_utils import JSONValue, require_dict

from covenant_ml.evaluation.codec_records import decode_interval
from covenant_ml.markets.replay import replay_maker
from covenant_ml.markets.statistics import decode_summary, encode_summary, summarize_replay

from ._fixtures import history, rule

START = date(2026, 1, 1)
END = date(2026, 1, 14)


def test_shared_event_day_is_one_unit_not_two_independent_bets() -> None:
    results = (
        replay_maker(history(ticker="a", event_day="2026-01-01"), rule()),
        replay_maker(history(ticker="b", event_day="2026-01-01"), rule()),
        replay_maker(history(ticker="c", event_day="2026-01-02", settlement=10000), rule()),
        replay_maker(history(ticker="d", event_day="2026-01-02", settlement=10000), rule()),
    )
    summary = summarize_replay(results, START, END, 100)
    assert (summary["markets"], summary["fills"], summary["event_days"]) == (4, 4, 2)
    assert (summary["pnl"], summary["friction_adjusted_pnl"], summary["max_drawdown"]) == (
        -16400,
        -16800,
        18200,
    )
    encoded = encode_summary(summary)
    per_order = decode_interval(require_dict(encoded, "per_order"))
    per_day = decode_interval(require_dict(encoded, "per_calendar_day"))
    assert per_order["estimate"] == pytest.approx(-0.41)
    assert per_order["std_error"] == pytest.approx(0.5)
    assert per_day["estimate"] == pytest.approx(-1.64 / 14)
    assert decode_summary(encode_summary(summary)) == summary


def test_short_empty_sample_has_explicitly_unavailable_intervals() -> None:
    summary = summarize_replay([], START, START, 0)
    assert (summary["markets"], summary["fills"], summary["contracts"], summary["pnl"]) == (
        0,
        0,
        0,
        0,
    )
    assert summary["per_order"] is summary["per_calendar_day"] is None
    assert decode_summary(encode_summary(summary)) == summary


def test_summary_refuses_empty_positions_and_unsupported_confidence_claims() -> None:
    empty = encode_summary(summarize_replay([], START, END, 0))
    empty["contracts"] = 1
    with pytest.raises(ValueError, match=r"MARKET-014.*must agree"):
        decode_summary(empty)
    filled = replay_maker(history(), rule())
    single = encode_summary(summarize_replay([filled], START, END, 0))
    single["event_days"] = 0
    with pytest.raises(ValueError, match=r"MARKET-014.*must agree"):
        decode_summary(single)
    second = replay_maker(history(ticker="second", event_day="2026-01-02"), rule())
    paired = encode_summary(summarize_replay([filled, second], START, END, 0))
    paired["per_order"] = None
    with pytest.raises(ValueError, match=r"MARKET-014.*at least two"):
        decode_summary(paired)


@pytest.mark.parametrize(
    "end,friction,lags,confidence",
    [
        (date(2025, 12, 31), 0, 7, 0.95),
        (END, -1, 7, 0.95),
        (END, 0, 0, 0.95),
        (END, 0, 7, 1.0),
    ],
)
def test_invalid_summary_assumptions_are_refused(
    end: date, friction: int, lags: int, confidence: float
) -> None:
    with pytest.raises(ValueError, match="MARKET-013"):
        summarize_replay([], START, end, friction, confidence, lags)


def test_summary_refuses_duplicate_contracts_and_outside_dates() -> None:
    result = replay_maker(history(), rule())
    with pytest.raises(ValueError, match="MARKET-013"):
        summarize_replay([result, result], START, END, 0)
    with pytest.raises(ValueError, match="MARKET-013"):
        summarize_replay([result], date(2026, 1, 2), END, 0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("fills", -1),
        ("markets", -1),
        ("contracts", -1),
        ("max_drawdown", -1),
        ("friction_per_contract", -1),
        ("friction_adjusted_pnl", 1),
    ],
)
def test_inconsistent_saved_summary_is_refused(key: str, value: JSONValue) -> None:
    document = encode_summary(summarize_replay([], START, START, 0))
    document[key] = value
    with pytest.raises(ValueError, match="MARKET-014"):
        decode_summary(document)


__all__: list[str] = []
