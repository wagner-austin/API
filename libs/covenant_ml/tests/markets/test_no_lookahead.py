"""Future mutations preserve decisions; past controls prove the inputs are used."""

from __future__ import annotations

import pytest
from platform_core.json_utils import narrow_json_to_dict, require_list

from covenant_ml.markets.codecs import decode_history, encode_history
from covenant_ml.markets.forecast_model import chronological_split, fit_logit, predict_logit
from covenant_ml.markets.forecast_records import decode_forecast_row, encode_forecast_row
from covenant_ml.markets.regimes import pre_entry_regime
from covenant_ml.markets.replay import replay_maker
from covenant_ml.markets.types import ReplayStatus

from ._fixtures import rule
from ._forecast_fixtures import comparison_plan, comparison_rows
from .test_regimes import grid_history


@pytest.mark.parametrize("mutation", ["quotes", "trades", "settlement", "remove_future", "reverse"])
def test_future_mutation_cannot_change_past_features_or_placement(mutation: str) -> None:
    baseline = grid_history()
    source = encode_history(baseline)
    quotes = require_list(source, "quotes")
    quotes.append({"timestamp_ms": 20_000_001, "bid": 0, "ask": 10000})
    if mutation == "quotes":
        quotes.append({"timestamp_ms": 20_003_000, "bid": 9900, "ask": 10000})
    elif mutation == "trades":
        narrow_json_to_dict(require_list(source, "trades")[0])["price"] = 0
    elif mutation == "settlement":
        source["settlement"] = 10000
    elif mutation == "remove_future":
        source["trades"] = []
        del quotes[-1]
    else:
        quotes.reverse()
        require_list(source, "trades").reverse()
    changed = decode_history(source)
    assert pre_entry_regime(changed) == pre_entry_regime(baseline)
    original, result = replay_maker(baseline, rule()), replay_maker(changed, rule())
    assert original["offer"] == result["offer"] == 700
    assert original["status"] is ReplayStatus.FILLED
    assert original["pnl"] == 700
    if mutation in ("trades", "remove_future"):
        assert result["status"] is ReplayStatus.UNFILLED
        assert result["pnl"] == 0
    elif mutation == "settlement":
        assert result["status"] is ReplayStatus.FILLED
        assert result["pnl"] == -9300
    else:
        assert result == original


def test_past_quote_control_changes_features_and_order_price() -> None:
    baseline = grid_history()
    source = encode_history(baseline)
    last = narrow_json_to_dict(require_list(source, "quotes")[-1])
    last["bid"], last["ask"] = 700, 900
    changed = decode_history(source)
    before, after = pre_entry_regime(baseline), pre_entry_regime(changed)
    assert before["net_change"] == 100.0
    assert after["net_change"] == 200.0
    assert before["hourly_change_std"] != after["hourly_change_std"]
    assert replay_maker(baseline, rule())["offer"] == 700
    assert replay_maker(changed, rule())["offer"] == 800


def test_future_weather_labels_cannot_change_fitted_forecast() -> None:
    rows = comparison_rows()
    plan = comparison_plan()
    train, test, _ = chronological_split(rows, plan)
    future_tickers = {row["ticker"] for row in test}
    documents = [encode_forecast_row(row) for row in rows]
    for row, document in zip(rows, documents, strict=True):
        if row["ticker"] in future_tickers:
            document["yes"] = 1 - row["yes"]
    changed = tuple(decode_forecast_row(document) for document in documents)
    new_train, new_test, _ = chronological_split(changed, plan)
    assert new_train == train
    assert tuple(row["yes"] for row in new_test) == tuple(1 - row["yes"] for row in test)
    original = fit_logit(train, True)
    rebuilt = fit_logit(new_train, True)
    assert rebuilt == original
    assert predict_logit(rebuilt, new_test) == predict_logit(original, test)
    past_documents = [encode_forecast_row(row) for row in train]
    for row, document in zip(train, past_documents, strict=True):
        document["yes"] = 1 - row["yes"]
    past_changed = tuple(decode_forecast_row(document) for document in past_documents)
    control = fit_logit(past_changed, True)
    assert control["coefficients"] != original["coefficients"]
    assert predict_logit(control, test) != predict_logit(original, test)
