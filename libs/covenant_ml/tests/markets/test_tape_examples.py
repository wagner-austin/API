"""Recount native records and detect misleading counts and example selection."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_list

from covenant_ml.markets.tape_examples import summarize_tape

from .test_acquisition import TICKER
from .test_public_tape import END, START, tape_market, tape_trade


def example_report() -> JSONObject:
    """Build native records with repeated events and combination legs."""
    return {
        "start_ts_inclusive": START,
        "end_ts_exclusive": END,
        "trade_count": 3,
        "contract_count": 2,
        "event_count": 1,
        "quantity_centi_contracts": 300,
        "block_trade_count": 0,
        "trades": [
            dict(tape_trade(), trade_id="z"),
            dict(tape_trade(), trade_id="a"),
            dict(tape_trade(), trade_id="m", ticker="SECOND-EVENT-CONTRACT"),
        ],
        "markets": [
            dict(tape_market(), mve_selected_legs=[{"event_ticker": "SHARED"}]),
            dict(tape_market(), ticker="SECOND-EVENT-CONTRACT", mve_selected_legs=[]),
        ],
    }


def test_recount_and_deterministic_examples() -> None:
    report = example_report()
    result = summarize_tape(report)
    assert result == {
        "start_ts_inclusive": START,
        "end_ts_exclusive": END,
        "trade_count": 3,
        "contract_count": 2,
        "event_count": 1,
        "quantity_centi_contracts": 300,
        "block_trade_count": 0,
        "contracts_with_combination_legs": 1,
        "distinct_declared_leg_events": 1,
        "namespaces": [
            {
                "namespace": TICKER.split("-", 1)[0],
                "trade_count": 2,
                "contract_count": 1,
                "example_trade": dict(tape_trade(), trade_id="a"),
                "example_market_at_capture": require_list(report, "markets")[0],
            },
            {
                "namespace": "SECOND",
                "trade_count": 1,
                "contract_count": 1,
                "example_trade": dict(tape_trade(), trade_id="m", ticker="SECOND-EVENT-CONTRACT"),
                "example_market_at_capture": require_list(report, "markets")[1],
            },
        ],
        "example_selection": "Smallest trade ID per ticker namespace; no return selection.",
        "interpretation": (
            "Metadata and outcomes are capture-time observations, not historical features. "
            "Event and leg counts do not establish independent observations. "
            "Namespaces are identifiers, not verified asset categories. No profit is estimated."
        ),
    }
    require_list(report, "trades").reverse()
    assert summarize_tape(report) == result


@pytest.mark.parametrize(
    "failure", ["duplicate-market", "duplicate-trade", "missing", "extra", "count", "boundary"]
)
def test_inconsistent_evidence_fails(failure: str) -> None:
    report = example_report()
    trades, markets = require_list(report, "trades"), require_list(report, "markets")
    if failure == "duplicate-market":
        markets.append(markets[0])
    elif failure == "duplicate-trade":
        trades.append(trades[0])
    elif failure == "missing":
        markets.pop()
    elif failure == "extra":
        markets.append(dict(tape_market(), ticker="UNTRADED"))
    elif failure == "count":
        report["trade_count"] = 900000
    else:
        trades[0] = dict(tape_trade(), created_time="2026-01-01T21:01:00Z")
    with pytest.raises(ValueError, match="TAPE-008"):
        summarize_tape(report)


@pytest.mark.parametrize("legs", [None, []])
def test_empty_tape_and_missing_legs_are_counted(legs: JSONValue) -> None:
    report = example_report()
    report["markets"] = [dict(tape_market(), mve_selected_legs=legs)]
    report["trades"] = [tape_trade()]
    report["contract_count"] = report["trade_count"] = 1
    report["quantity_centi_contracts"] = 100
    summary = summarize_tape(report)
    assert summary["contracts_with_combination_legs"] == 0
    assert summary["distinct_declared_leg_events"] == 0
    report["trades"], report["markets"] = [], []
    for name in ("trade_count", "contract_count", "event_count", "quantity_centi_contracts"):
        report[name] = 0
    summary = summarize_tape(report)
    assert summary["namespaces"] == []
    assert summary["trade_count"] == 0


def test_malformed_leg_is_not_silently_counted() -> None:
    report = example_report()
    report["markets"] = [dict(tape_market(), mve_selected_legs=[{"event_ticker": 4}])]
    with pytest.raises(TypeError, match="event_ticker"):
        summarize_tape(report)


@pytest.mark.parametrize("start,end", [(-1, 10), (1, 1), (2, 1), (0, 86401)])
def test_invalid_window_is_rejected(start: int, end: int) -> None:
    report = example_report()
    report["start_ts_inclusive"], report["end_ts_exclusive"] = start, end
    with pytest.raises(ValueError, match=r"TAPE-008.*window"):
        summarize_tape(report)
