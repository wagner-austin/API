"""Real logistic fitting, chronological isolation and exact quoted policy accounting."""

from __future__ import annotations

from datetime import date

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_dict,
    require_float,
    require_list,
)

from covenant_ml.markets.forecast_evaluation import evaluate_forecasts
from covenant_ml.markets.forecast_model import (
    chronological_split,
    decode_fitted_logit,
    encode_fitted_logit,
    features,
    fit_logit,
    logit,
    predict_logit,
)
from covenant_ml.markets.forecast_policy import quoted_policy
from covenant_ml.markets.forecast_records import (
    decode_comparison_plan,
    decode_forecast_row,
    encode_comparison_plan,
    encode_forecast_row,
)

from ._forecast_fixtures import comparison_plan, comparison_rows, forecast_row


def test_chronological_split_excludes_unpublished_labels() -> None:
    rows = comparison_rows()
    train, test, excluded = chronological_split(rows, comparison_plan())
    assert len(train) == 10
    assert len(test) == 20
    assert excluded == 2
    assert max(r["settled_ms"] for r in train) < min(r["decision_ms"] for r in test)
    assert {r["event_day"] for r in train}.isdisjoint({r["event_day"] for r in test})
    plan = encode_comparison_plan(comparison_plan())
    plan["excluded_days"] = [rows[0]["event_day"]]
    with pytest.raises(ValueError, match="FORECAST-007"):
        chronological_split(rows, decode_comparison_plan(plan))


@pytest.mark.parametrize("case", ["duplicate", "outside", "no_test", "no_train"])
def test_split_refuses_invalid_panels(case: str) -> None:
    rows = comparison_rows()
    if case == "duplicate":
        rows = (*rows, rows[0])
    elif case == "outside":
        rows = (*rows, forecast_row(date(2026, 4, 19)))
    elif case == "no_test":
        rows = rows[:2]
    else:
        rows = rows[-2:]
    with pytest.raises(ValueError, match="FORECAST-007"):
        chronological_split(rows, comparison_plan())


def test_real_fit_portable_roundtrip_and_nested_features() -> None:
    train, test, _ = chronological_split(comparison_rows(), comparison_plan())
    baseline, augmented = fit_logit(train, False), fit_logit(train, True)
    assert decode_fitted_logit(encode_fitted_logit(augmented)) == augmented
    assert len(features(train[0], False)) == 3
    assert len(features(train[0], True)) == 4
    assert predict_logit(baseline, test) == pytest.approx((0.5,) * len(test), abs=1e-12)
    assert baseline["iterations"] == 0
    predictions = predict_logit(augmented, test)
    assert test[0]["yes"] == 1
    assert test[1]["yes"] == 0
    assert predictions[0] == pytest.approx(0.8907891481, abs=1e-7)
    assert predictions[1] == pytest.approx(0.1092108519, abs=1e-7)
    assert logit(0.0) == pytest.approx(-9.21024036697585)
    assert logit(1.0) == pytest.approx(9.21024036697596)


def test_real_solver_nonconvergence_is_not_silenced() -> None:
    with (
        pytest.warns(UserWarning, match="failed to converge"),
        pytest.raises(ValueError, match="FORECAST-008"),
    ):
        fit_logit(comparison_rows(), True, max_iter=1)
    with pytest.raises(ValueError, match="FORECAST-008"):
        fit_logit((forecast_row(),), False)
    with pytest.raises(ValueError, match="FORECAST-008"):
        fit_logit(comparison_rows(), False, max_iter=0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("features", ["wrong"]),
        ("coefficients", [1.0]),
        ("coefficients", [float("inf"), 0.0, 0.0]),
        ("intercept", float("nan")),
    ],
)
def test_model_codec_refuses_invalid_parameters(key: str, value: JSONValue) -> None:
    document: JSONObject = {
        "features": ["market_logit", "boston", "philadelphia"],
        "coefficients": [0.0, 0.0, 0.0],
        "intercept": 0.0,
        "iterations": 1,
    }
    # Encode the actual immutable shape before injecting malformed JSON.
    native = encode_fitted_logit(decode_fitted_logit(document))
    native[key] = value
    with pytest.raises(ValueError, match="FORECAST-006"):
        decode_fitted_logit(native)


def test_policy_exact_payout_fee_hurdle_and_unavailable_extreme_prices() -> None:
    rows = (
        forecast_row(date(2026, 4, 3), 1),
        forecast_row(date(2026, 4, 4), 0),
        forecast_row(date(2026, 4, 5), 1),
    )
    report = quoted_policy(rows, (0.95, 0.05, 0.5))
    assert report["actions"] == ["buy_yes", "buy_no", "pass"]
    assert report["pnl_by_row"] == [3800, 3800, 0]
    assert report["fees_by_row"] == [200, 200, 0]
    assert report["trades"] == 2
    assert report["pnl"] == 7600
    assert report["extra_cent_pnl"] == 7400
    extreme = encode_forecast_row(rows[0])
    extreme["bid"], extreme["ask"] = 0, 0
    assert quoted_policy((decode_forecast_row(extreme), rows[1]), (1.0, 0.5))["trades"] == 0
    with pytest.raises(ValueError, match="FORECAST-009"):
        quoted_policy(rows, (0.5,))
    with pytest.raises(ValueError, match="FORECAST-009"):
        quoted_policy(rows, (0.5, 0.5, float("nan")))


def test_evaluation_never_fits_to_held_out_labels() -> None:
    rows = comparison_rows()
    report = evaluate_forecasts(rows, comparison_plan())
    changed = []
    for row in rows:
        document = encode_forecast_row(row)
        if row["event_day"] > comparison_plan()["train_end"]:
            document["yes"] = 1 - row["yes"]
        changed.append(decode_forecast_row(document))
    opposite = evaluate_forecasts(changed, comparison_plan())
    assert report["models"] == opposite["models"]
    original_arms, opposite_arms = require_dict(report, "arms"), require_dict(opposite, "arms")
    for name in original_arms:
        assert (
            require_dict(original_arms, name)["predictions"]
            == require_dict(opposite_arms, name)["predictions"]
        )
    primary = require_dict(require_dict(report, "comparisons"), "market_calibrated")
    assert primary["candidate_gate"] is True
    assert require_float(require_dict(primary, "brier_difference"), "ci_high") < 0
    serial = require_dict(primary, "daily_newey_west")
    assert serial["method"] == "newey_west"
    assert require_float(serial, "ci_high") < 0
    assert report["unavailable_training_labels"] == 2
    assert len(require_list(report, "training_rows")) == 10
    assert len(require_list(report, "evaluation_rows")) == 20


def test_short_panel_reports_absent_serial_interval() -> None:
    document = encode_comparison_plan(comparison_plan())
    document["end"] = "2026-04-10"
    plan = decode_comparison_plan(document)
    report = evaluate_forecasts(comparison_rows()[:16], plan)
    primary = require_dict(require_dict(report, "comparisons"), "market_calibrated")
    assert primary["daily_newey_west"] is None
