"""Actual scoring gives each release equal weight and selects without outcomes."""

from __future__ import annotations

import pytest
from platform_core.json_utils import narrow_json_to_dict, require_list

from covenant_ml.markets.inflation_evaluation import evaluate_inflation
from covenant_ml.markets.inflation_selection import inflation_selection
from covenant_ml.markets.kalshi import MarketSpec, decode_spec, encode_spec
from covenant_ml.markets.types import Quote

from ._inflation_experiment_fixtures import histories, market


def _rows() -> tuple[tuple[MarketSpec, Quote, float], ...]:
    selected, _ = inflation_selection((market(), market(2)), histories())
    return tuple(
        (spec, Quote(timestamp_ms=spec["decision_ms"], bid=4900, ask=5100), probability)
        for spec, probability in selected
    )


def test_release_weighting_selection_and_exact_costs() -> None:
    rows = _rows()
    duplicate_ladder = encode_spec(rows[0][0])
    duplicate_ladder["ticker"] = rows[0][0]["ticker"] + "B"
    losing = encode_spec(rows[0][0])
    losing["ticker"] = rows[0][0]["ticker"] + "C"
    losing["settlement"] = 0
    report = evaluate_inflation(
        (
            *rows,
            (decode_spec(duplicate_ladder), rows[0][1], rows[0][2]),
            (decode_spec(losing), rows[0][1], 0.5),
        )
    )
    assert report["release_dates"] == 2
    assert report["eligible_contracts"] == 4
    assert report["trades"] == 2
    assert report["hypothetical_profit_units"] == 9200
    assert report["double_fee_profit_units"] == 8800
    assert report["market_brier"] == 0.25
    loss = (60.5 / 61 - 1) ** 2
    assert report["model_brier"] == ((loss * 2 + 0.25) / 3 + loss) / 2
    decisions = tuple(narrow_json_to_dict(value) for value in require_list(report, "decisions"))
    assert tuple(row["ticker"] for row in decisions) == tuple(row[0]["ticker"] for row in rows)
    assert report["gates"] == {
        "release_dates": False,
        "trades": False,
        "improved_brier": True,
        "positive_lower_bound": True,
        "positive_double_fee_profit": True,
    }
    assert report["advance_to_prospective_demo"] is False
    assert report["orders_submitted"] == 0
    assert evaluate_inflation(tuple((spec, quote, 0.5) for spec, quote, _ in rows))["trades"] == 0


@pytest.mark.parametrize("kind", ["duplicate", "quote_time", "decision_time"])
def test_invalid_scoring_chronology(kind: str) -> None:
    rows = _rows()
    spec, _quote, probability = rows[0]
    if kind == "duplicate":
        invalid = (*rows, rows[0])
    elif kind == "quote_time":
        invalid = ((spec, Quote(timestamp_ms=1, bid=4900, ask=5100), probability), rows[1])
    else:
        changed = encode_spec(spec)
        changed["decision_ms"] = spec["decision_ms"] + 60000
        invalid = (
            (
                decode_spec(changed),
                Quote(timestamp_ms=spec["decision_ms"] + 60000, bid=4900, ask=5100),
                probability,
            ),
            rows[1],
        )
    with pytest.raises(ValueError, match="INFLATION-014"):
        evaluate_inflation(invalid)


def test_insufficient_release_dates() -> None:
    with pytest.raises(ValueError, match="INFLATION-015"):
        evaluate_inflation(_rows()[:1])
