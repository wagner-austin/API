"""Chronological separation, exact costs and rejection of malformed observations."""

from __future__ import annotations

from math import nan

import pytest
from platform_core.json_utils import require_dict, require_float, require_list

from covenant_ml.markets.kalshi import MarketSpec
from covenant_ml.markets.sports_strategy import (
    adjusted_probability,
    evaluate_sports,
    quote_decision,
)
from covenant_ml.markets.types import Quote

from ._strategy_fixtures import observation


def test_probability_and_cost_arithmetic() -> None:
    assert adjusted_probability(0.8, 2.0) == pytest.approx(16 / 17)
    assert adjusted_probability(0.8, 1.0) == 0.8
    _, quote = observation(bid=4900, ask=5100)
    assert quote_decision(quote, 0.8, 10000) == {
        "side": "yes",
        "cost_units": 5400,
        "fee_units": 200,
        "profit_units": 4600,
        "double_fee_profit_units": 4400,
    }
    assert quote_decision(quote, 0.2, 10000) == {
        "side": "no",
        "cost_units": 5400,
        "fee_units": 200,
        "profit_units": -5400,
        "double_fee_profit_units": -5600,
    }
    assert quote_decision(quote, 0.5, 0) == {
        "side": "pass",
        "cost_units": 0,
        "fee_units": 0,
        "profit_units": 0,
        "double_fee_profit_units": 0,
    }
    assert quote_decision(Quote(timestamp_ms=0, bid=9950, ask=9990), 0.0, 0)["cost_units"] == 250
    assert quote_decision(Quote(timestamp_ms=0, bid=10, ask=50), 1.0, 10000)["cost_units"] == 250


@pytest.mark.parametrize("probability,exponent", [(0.0, 1.0), (1.0, 1.0), (nan, 1.0), (0.5, 3.0)])
def test_invalid_probability(probability: float, exponent: float) -> None:
    with pytest.raises(ValueError, match="STRATEGY-004"):
        adjusted_probability(probability, exponent)


@pytest.mark.parametrize(
    "bid,ask", [(None, 5000), (5000, None), (0, 5000), (5000, 10000), (5100, 5000), (4000, 5000)]
)
def test_invalid_quotes(bid: int | None, ask: int | None) -> None:
    with pytest.raises(ValueError, match="STRATEGY-005"):
        quote_decision(Quote(timestamp_ms=0, bid=bid, ask=ask), 0.5, 0)


@pytest.mark.parametrize("probability,payout", [(nan, 0), (-0.1, 0), (0.5, 5000)])
def test_invalid_decision_inputs(probability: float, payout: int) -> None:
    with pytest.raises(ValueError, match="STRATEGY-006"):
        quote_decision(observation()[1], probability, payout)


def test_training_is_frozen_when_evaluation_outcomes_change() -> None:
    train = [observation("26JAN01"), observation("26JAN02")]
    test = [observation("26MAR01"), observation("26MAR02", payout=0)]
    report = evaluate_sports([*train, *test])
    reversed_outcomes = evaluate_sports(
        [*train, observation("26MAR01", payout=0), observation("26MAR02")]
    )
    assert report["exponent"] == 2.0
    assert report["training_brier"] == reversed_outcomes["training_brier"]
    assert report["exponent"] == reversed_outcomes["exponent"]
    assert report["trades"] == 2
    assert report["hypothetical_profit_units"] == -4800
    assert report["double_fee_profit_units"] == -5200
    assert report["daily_profit_units"] == {"2026-03-01": 2600, "2026-03-02": -7400}
    assert report["market_brier"] == pytest.approx(0.29)
    assert report["adjusted_brier"] == pytest.approx(((49 / 58 - 1) ** 2 + (49 / 58) ** 2) / 2)
    assert report["advance_to_prospective_demo"] is False
    assert require_dict(report, "gates") == {
        "training_games": False,
        "evaluation_games": False,
        "evaluation_trades": False,
        "improved_brier": False,
        "positive_daily_lower_bound": False,
    }
    interval = require_dict(report, "daily_profit_interval_dollars")
    assert require_float(interval, "estimate") == -0.24
    assert require_float(interval, "ci_low") == pytest.approx(-6.593102368216048)
    assert len(require_list(report, "decisions")) == 2
    assert evaluate_sports(list(reversed([*train, *test]))) == report


def test_tie_prefers_identity_and_abstentions_remain_in_daily_interval() -> None:
    report = evaluate_sports(
        [
            observation("26JAN01", bid=4900, ask=5100),
            observation("26MAR01", bid=4900, ask=5100),
            observation("26MAR02", bid=4900, ask=5100),
        ]
    )
    assert report["exponent"] == 1.0
    assert report["trades"] == 0
    assert report["daily_profit_units"] == {"2026-03-01": 0, "2026-03-02": 0}
    assert require_dict(report, "daily_profit_interval_dollars")["ci_low"] == 0.0


@pytest.mark.parametrize(
    "kind", ["duplicate", "day", "quote_time", "decision_time", "close", "payout", "late_training"]
)
def test_invalid_evaluation_chronology(kind: str) -> None:
    spec, quote = observation()
    changed: MarketSpec = spec
    if kind == "day":
        changed = {**spec, "day": "2025-12-31"}
    elif kind == "quote_time":
        quote = {**quote, "timestamp_ms": 0}
    elif kind == "decision_time":
        changed = {**spec, "decision_ms": 0}
        quote = {**quote, "timestamp_ms": 0}
    elif kind == "close":
        changed = {**spec, "close_ms": 0}
    elif kind == "payout":
        changed = {**spec, "settlement": 5000}
    elif kind == "late_training":
        changed = {**spec, "settled_ms": observation("26MAR01")[0]["settled_ms"]}
    rows = [(changed, quote), observation("26MAR01"), observation("26MAR02")]
    if kind == "duplicate":
        rows.append((spec, quote))
    with pytest.raises(ValueError, match="STRATEGY-007"):
        evaluate_sports(rows)


@pytest.mark.parametrize("training", [False, True])
def test_insufficient_dates(training: bool) -> None:
    rows = [observation()] if training else [observation("26MAR01"), observation("26MAR02")]
    with pytest.raises(ValueError, match="STRATEGY-008"):
        evaluate_sports(rows)
