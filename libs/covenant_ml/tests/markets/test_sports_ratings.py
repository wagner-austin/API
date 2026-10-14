"""Information timing, stable team identities and shared cost evaluation."""

from __future__ import annotations

import pytest
from platform_core.json_utils import require_dict, require_float

from covenant_ml.markets.kalshi import MarketSpec
from covenant_ml.markets.sports_ratings import _blend, evaluate_ratings, rating_predictions
from covenant_ml.markets.sports_strategy import score_sports_predictions
from covenant_ml.markets.types import Quote

from ._rating_fixtures import game, warmup
from ._strategy_fixtures import observation


def test_future_outcome_cannot_enter_its_own_forecast() -> None:
    spec, _ = observation()
    records = [*warmup(), *game("26JAN01")]
    first, counts = rating_predictions(records, [spec])
    reversed_result, _ = rating_predictions([*warmup(), *game("26JAN01", "DAL")], [spec])
    assert first == reversed_result
    assert first[spec["ticker"]] == pytest.approx(0.7113665835137366, abs=1e-14)
    assert counts == {
        "inventory_games": 11,
        "incomplete_pairs": 0,
        "nonbinary_games": 0,
        "unavailable_target": 0,
        "insufficient_history": 0,
        "eligible_forecasts": 1,
    }
    assert rating_predictions(list(reversed(records)), [spec]) == (first, counts)


def test_updates_wait_for_both_contracts_and_strict_cutoff() -> None:
    spec, _ = observation()
    prior = warmup()
    prior[-1]["settlement_ts"] = "2026-01-01T12:00:00Z"
    predictions, counts = rating_predictions([*prior, *game("26JAN01")], [spec])
    assert predictions == {}
    assert counts["insufficient_history"] == 1
    prior[-1]["settlement_ts"] = "2026-01-01T11:59:59Z"
    assert rating_predictions([*prior, *game("26JAN01")], [spec])[0][
        spec["ticker"]
    ] == pytest.approx(0.7113665835137366, abs=1e-14)


def test_exclusions_are_explicit() -> None:
    incomplete = game("25OCT20")[0]
    partial = game("25OCT21")
    partial[0]["settlement_value_dollars"] = "0.5000"
    pending = game("25OCT22")
    pending[0]["status"] = "closed"
    records = [*game("25SEP30"), incomplete, *partial, *pending, *game("26JAN01")]
    forecasts, counts = rating_predictions(records, [observation()[0], observation("26MAR01")[0]])
    assert forecasts == {}
    assert counts == {
        "inventory_games": 4,
        "incomplete_pairs": 1,
        "nonbinary_games": 2,
        "unavailable_target": 1,
        "insufficient_history": 1,
        "eligible_forecasts": 0,
    }


@pytest.mark.parametrize(
    "kind,code",
    [
        ("duplicate", "RATING-001"),
        ("mismatch", "RATING-001"),
        ("same_team", "RATING-001"),
        ("early_settle", "RATING-002"),
        ("duplicate_target", "RATING-003"),
        ("wrong_target", "RATING-003"),
        ("settled_target", "RATING-003"),
    ],
)
def test_invalid_native_and_target_evidence(kind: str, code: str) -> None:
    records = [*warmup(), *game("26JAN01")]
    spec, _ = observation()
    specs = [spec]
    if kind == "duplicate":
        records.append(records[0])
    elif kind == "mismatch":
        records[0]["ticker"] = "other"
    elif kind == "same_team":
        records[0]["custom_strike"] = {"hockey_team": "dallas-id"}
    elif kind == "early_settle":
        records[0]["settlement_ts"] = "2025-10-01T00:00:00Z"
    elif kind == "duplicate_target":
        specs.append(spec)
    elif kind == "wrong_target":
        specs = [{**spec, "ticker": "KXNHLGAME-26JAN01DALCHI-DAL"}]
    elif kind == "settled_target":
        specs = [{**spec, "decision_ms": spec["settled_ms"] + 86400000}]
    with pytest.raises(ValueError, match=code):
        rating_predictions(records, specs)


def test_fit_blend_and_pure_diagnostic_reuse_same_accounting() -> None:
    rows = [observation(), observation("26MAR01"), observation("26MAR02", payout=0)]
    forecasts = {spec["ticker"]: 0.9 for spec, _ in rows}
    report = evaluate_ratings(rows, forecasts)
    assert report["weight"] == 1.0
    assert report["trades"] == 2
    assert report["hypothetical_profit_units"] == -4800
    assert require_dict(report, "pure_rating_diagnostic")["hypothetical_profit_units"] == -4800
    assert report["adjusted_brier"] == pytest.approx(0.41)
    expected = {"0.0": 0.09, "0.25": 0.0625, "0.5": 0.04, "0.75": 0.0225, "1.0": 0.01}
    scores = require_dict(report, "training_brier")
    assert set(scores) == set(expected)
    for weight, value in expected.items():
        assert require_float(scores, weight) == pytest.approx(value)
    assert evaluate_ratings([*rows, observation("26MAR03")], forecasts) == report
    inverted: list[MarketSpec] = [
        {**spec, "settlement": 10000 - spec["settlement"]} for spec, _ in rows[1:]
    ]
    changed = evaluate_ratings(
        [rows[0], (inverted[0], rows[1][1]), (inverted[1], rows[2][1])], forecasts
    )
    assert changed["weight"] == report["weight"]
    assert changed["training_brier"] == report["training_brier"]


def test_blend_tie_prefers_market_and_unknown_forecasts_fail() -> None:
    rows = [observation(), observation("26MAR01"), observation("26MAR02")]
    forecasts = {spec["ticker"]: 0.7 for spec, _ in rows}
    report = evaluate_ratings(rows, forecasts)
    assert report["weight"] == 0.0
    assert report["trades"] == 0
    forecasts["wrong"] = 0.7
    with pytest.raises(ValueError, match="RATING-005"):
        evaluate_ratings(rows, forecasts)
    with pytest.raises(ValueError, match="STRATEGY-010"):
        score_sports_predictions(rows, {"wrong": 0.5})


@pytest.mark.parametrize(
    "bid,ask,rating", [(None, 5000, 0.5), (5000, None, 0.5), (4900, 5100, 1.1)]
)
def test_invalid_blend_inputs(bid: int | None, ask: int | None, rating: float) -> None:
    with pytest.raises(ValueError, match="RATING-004"):
        _blend(Quote(timestamp_ms=0, bid=bid, ask=ask), rating, 0.5)
