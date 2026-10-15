"""Real fits, independent arithmetic and temporal falsifiers for crypto comparisons."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_list,
)

from covenant_ml.markets.interval_comparison import compare_intervals
from covenant_ml.markets.interval_inputs import interval_observations, split_interval_rows
from covenant_ml.markets.interval_model import fit_interval_model, predict_interval_model
from covenant_ml.markets.interval_records import (
    decode_interval_fit,
    decode_interval_row,
    encode_interval_fit,
    encode_interval_row,
)
from covenant_ml.markets.interval_scoring import interval_daily_estimate, interval_quoted_policy

from .test_interval_quotes import START, cohort_document, plan_document


def comparison_documents() -> tuple[JSONObject, JSONObject, JSONObject]:
    """Construct an explicit sixteen-date binary experiment, with no estimator fakes."""
    cohort, quote_plan = cohort_document(), plan_document()
    cohort["end_ts_exclusive"] = START + 16 * 86400
    quote_plan["end_ts"] = START + 16 * 86400
    families: list[JSONValue] = []
    attachments: list[JSONValue] = []
    for asset, series in enumerate(("KXBTC15M", "KXETH15M")):
        markets: list[JSONValue] = []
        for day in range(16):
            opened = START + day * 86400
            event = f"{series}-D{day}"
            ticker = event + "-A"
            result = "yes" if (day + asset) % 2 else "no"
            markets.append(
                {
                    "ticker": ticker,
                    "event_ticker": event,
                    "result": result,
                    "open_time": datetime.fromtimestamp(opened, UTC).isoformat(),
                    "close_time": datetime.fromtimestamp(opened + 900, UTC).isoformat(),
                    "settlement_ts": datetime.fromtimestamp(opened + 960, UTC).isoformat(),
                }
            )
            attachments.append(
                {
                    "ticker": ticker,
                    "event_ticker": event,
                    "series": series,
                    "decision_ms": (opened + 300) * 1000,
                    "quote_age_ms": 60000,
                    "status": "two_sided",
                    "quote": {"timestamp_ms": (opened + 240) * 1000, "bid": 4000, "ask": 4200},
                }
            )
        families.append({"series": series, "markets": markets})
    cohort["families"] = families
    quotes: JSONObject = {"plan": quote_plan, "contracts": attachments}
    plan: JSONObject = {
        "schema_version": 1,
        "study_kind": "exploratory_chronological_development",
        "train_end_ts": START + 6 * 86400,
        "fee_coefficient": 700,
        "edge_hurdle": 200,
        "extra_friction": [0, 100, 200],
        "serial_lags": 7,
        "inverse_regularization": 1.0,
        "declared_at": "2026-01-20T00:00:00Z",
        "models": ["raw_market", "earlier_asset_base_rate", "logistic_recalibration"],
        "features": ["prior_market_logit", "ether_indicator"],
    }
    return cohort, quotes, plan


def test_real_fit_future_outcomes_and_later_features_do_not_change_earlier_predictions() -> None:
    cohort, quotes, plan = comparison_documents()
    report = compare_intervals(cohort, quotes, plan)
    assert report["training_count"] == 12
    assert report["evaluation_count"] == 20
    assert report["exclusions"] == []
    assert report["purged_training_labels"] == []
    fitted = decode_interval_fit(require_dict(report, "model"))
    assert fitted["bitcoin_base"] == fitted["ether_base"] == 0.5
    assert encode_interval_fit(fitted) == report["model"]
    changed = deepcopy(cohort)
    for value in require_list(changed, "families"):
        family = narrow_json_to_dict(value)
        for item in require_list(family, "markets")[6:]:
            market = narrow_json_to_dict(item)
            market["result"] = "yes"
    altered = compare_intervals(changed, quotes, plan)
    assert altered["model"] == report["model"]
    for name in ("raw_market", "earlier_asset_base_rate", "logistic_recalibration"):
        before = require_dict(require_dict(report, "arms"), name)
        after = require_dict(require_dict(altered, "arms"), name)
        assert before["predictions"] == after["predictions"]
        for a, b in zip(
            require_list(before, "cost_scenarios"),
            require_list(after, "cost_scenarios"),
            strict=True,
        ):
            assert narrow_json_to_dict(a)["actions"] == narrow_json_to_dict(b)["actions"]
    changed_quotes = deepcopy(quotes)
    last = narrow_json_to_dict(require_list(changed_quotes, "contracts")[-1])
    require_dict(last, "quote").update(bid=9000, ask=9200)
    future = compare_intervals(cohort, changed_quotes, plan)
    assert future["model"] == report["model"]
    assert (
        require_list(require_dict(require_dict(future, "arms"), "raw_market"), "predictions")[-1]
        == 0.91
    )
    assert (
        require_dict(require_dict(report, "arms"), "raw_market")["brier"]
        != (require_dict(require_dict(altered, "arms"), "raw_market")["brier"])
    )
    before_predictions = require_list(
        require_dict(require_dict(report, "arms"), "logistic_recalibration"), "predictions"
    )
    future_predictions = require_list(
        require_dict(require_dict(future, "arms"), "logistic_recalibration"), "predictions"
    )
    assert before_predictions[:-1] == future_predictions[:-1]


def test_exact_cost_arithmetic_and_calendar_weighting() -> None:
    cohort, quotes, _ = comparison_documents()
    rows, _ = interval_observations(cohort, quotes)
    _, test, _ = split_interval_rows(rows, START + 6 * 86400)
    for probability, action, expected in (
        (1.0, "buy_yes", [row["yes"] * 10000 - 4500 for row in test]),
        (0.0, "buy_no", [(1 - row["yes"]) * 10000 - 6300 for row in test]),
        (0.41, "pass", [0 for row in test]),
    ):
        policy = interval_quoted_policy(test, tuple(probability for _ in test), 100)
        assert policy["actions"] == [action] * 20
        assert policy["fees"] == [0 if action == "pass" else 200] * 20
        assert policy["pnl_by_row"] == expected
        assert policy["pnl"] == sum(expected)
        assert policy["trades"] == (0 if action == "pass" else 20)
    estimate = interval_daily_estimate(tuple(float(row["yes"]) for row in test), test)
    assert [narrow_json_to_dict(day)["mean"] for day in require_list(estimate, "days")] == [
        0.5
    ] * 10
    assert encode_interval_row(decode_interval_row(encode_interval_row(test[0]))) == (
        encode_interval_row(test[0])
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("ticker", "OTHER-A"),
        ("series", "OTHER"),
        ("bid", 4300),
        ("decision_ms", 2**62),
        ("settled_ms", 1),
    ],
)
def test_invalid_row(field: str, value: JSONValue) -> None:
    cohort, quotes, _ = comparison_documents()
    rows, _ = interval_observations(cohort, quotes)
    document = encode_interval_row(rows[0])
    document[field] = value
    with pytest.raises(ValueError, match="IFORE-002"):
        decode_interval_row(document)


@pytest.mark.parametrize(
    "field,value",
    [("market_weight", float("inf")), ("bitcoin_base", -0.1), ("ether_base", float("nan"))],
)
def test_invalid_fit(field: str, value: float) -> None:
    cohort, quotes, _ = comparison_documents()
    rows, _ = interval_observations(cohort, quotes)
    document = encode_interval_fit(fit_interval_model(rows))
    document[field] = value
    with pytest.raises(ValueError, match="IFORE-004"):
        decode_interval_fit(document)


def test_actual_nonconvergence_and_stable_extreme_prediction() -> None:
    cohort, quotes, _ = comparison_documents()
    rows, _ = interval_observations(cohort, quotes)
    varying = tuple(
        decode_interval_row(
            dict(
                encode_interval_row(row),
                bid=8000 if row["yes"] else 1000,
                ask=8200 if row["yes"] else 1200,
            )
        )
        for row in rows
    )
    with (
        pytest.warns(Warning, match="failed to converge"),
        pytest.raises(ValueError, match="IFORE-004"),
    ):
        fit_interval_model(varying, 1)
    with pytest.raises(ValueError, match="IFORE-004"):
        fit_interval_model(rows, 0)
    with pytest.raises(ValueError, match="IFORE-003"):
        fit_interval_model(rows[:1])
    for intercept, expected in ((1000.0, 1.0), (-1000.0, 0.0)):
        model = decode_interval_fit(
            dict(encode_interval_fit(fit_interval_model(rows)), intercept=intercept)
        )
        assert predict_interval_model(model, rows[0]) == expected
