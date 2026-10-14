"""Causal quote arithmetic, explicit coverage and exhaustive outcome partitioning."""

from __future__ import annotations

from datetime import date
from math import sqrt

import pytest
from platform_core.json_utils import (
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_int,
    require_list,
)

from covenant_ml.markets.codecs import decode_history, encode_history
from covenant_ml.markets.experiment import decode_plan, evaluate_panel
from covenant_ml.markets.regime_records import HOUR_MS, encode_quote_regime
from covenant_ml.markets.regimes import pre_entry_regime, regime_label, summarize_regimes
from covenant_ml.markets.replay import replay_markets
from covenant_ml.markets.statistics import encode_summary, summarize_replay
from covenant_ml.markets.types import MarketHistory

from ._fixtures import history, rule
from .test_acquisition import SERIES
from .test_experiment import plan_document


def grid_history(values: tuple[int, ...] = (600, 700, 600, 700, 600, 700)) -> MarketHistory:
    """Build six real-code input closes and a later matching public trade."""
    quotes: list[JSONValue] = [
        {"timestamp_ms": 20_000_000 - (5 - index) * HOUR_MS, "bid": value - 100, "ask": value + 100}
        for index, value in enumerate(values)
    ]
    return history(
        series=SERIES,
        decision_ms=20_000_000,
        close_ms=30_000_000,
        quotes=quotes,
        trades=[
            {
                "trade_id": "later",
                "timestamp_ms": 20_002_000,
                "price": 1100,
                "quantity": 100,
                "taker_yes": True,
                "block": False,
            }
        ],
    )


def test_exact_population_arithmetic_and_monotonic_trend() -> None:
    measured = pre_entry_regime(grid_history())
    assert encode_quote_regime(measured) == {
        "usable_quotes": 6,
        "hourly_change_std": sqrt(9600),
        "net_change": 100.0,
    }
    assert regime_label(measured) == "low_rising"
    trend = pre_entry_regime(grid_history((200, 300, 400, 500, 600, 700)))
    assert encode_quote_regime(trend) == {
        "usable_quotes": 6,
        "hourly_change_std": 0.0,
        "net_change": 500.0,
    }
    assert regime_label(trend) == "low_rising"


@pytest.mark.parametrize("change", ["gap", "bid", "ask", "off_grid"])
def test_missing_grid_or_side_never_filled_forward(change: str) -> None:
    source = encode_history(grid_history())
    quotes = require_list(source, "quotes")
    if change == "gap":
        del quotes[2]
    elif change == "off_grid":
        narrow_json_to_dict(quotes[2])["timestamp_ms"] = 20_000_000 - 3 * HOUR_MS + 1
    else:
        narrow_json_to_dict(quotes[2])[change] = None
    result = pre_entry_regime(decode_history(source))
    assert encode_quote_regime(result) == {
        "usable_quotes": 5,
        "hourly_change_std": None,
        "net_change": None,
    }
    assert regime_label(result) == "unclassified"


def test_groups_partition_all_results_and_preserve_existing_statistics() -> None:
    first = grid_history()
    second = history(series=SERIES, ticker="missing", event_day="2026-01-02")
    rows = (first, second)
    plan = decode_plan(plan_document())
    report = evaluate_panel(rows, plan)
    entries = require_list(require_dict(report, "pre_entry_regimes"), "contracts")
    assert [
        require_dict(narrow_json_to_dict(row), "measurements")["usable_quotes"] for row in entries
    ] == [6, 1]
    for value in require_list(report, "results"):
        arm = narrow_json_to_dict(value)
        groups = require_dict(arm, "by_regime")
        expected = replay_markets(rows, rule(through=require_int(arm, "through")))
        start, end = date(2026, 1, 1), date(2026, 1, 14)
        assert arm["summary"] == encode_summary(summarize_replay(expected, start, end, 100))
        assert groups["low_rising"] == encode_summary(
            summarize_replay(expected[:1], start, end, 100)
        )
        assert groups["unclassified"] == encode_summary(
            summarize_replay(expected[1:], start, end, 100)
        )
        for field in ("markets", "fills", "contracts", "pnl", "friction_adjusted_pnl"):
            assert sum(
                require_int(narrow_json_to_dict(group), field) for group in groups.values()
            ) == require_int(require_dict(arm, "summary"), field)
        assert len(groups) == 7


@pytest.mark.parametrize("change", ["missing", "wrong", "duplicate"])
def test_group_identity_must_match_exactly(change: str) -> None:
    item = grid_history()
    results = replay_markets([item], rule())
    features = {item["ticker"]: pre_entry_regime(item)}
    if change == "missing":
        features.clear()
    elif change == "wrong":
        features = {"other": pre_entry_regime(item)}
    else:
        results = (*results, *results)
    with pytest.raises(ValueError, match="REGIME-003"):
        summarize_regimes(results, features, date(2026, 1, 1), date(2026, 1, 14), 100)
