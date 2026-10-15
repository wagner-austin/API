"""Actual fitted forecasts and altered records exercise the offline evidence audit."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import pytest
from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)

from covenant_ml.markets.baseline_audit import audit_baseline
from covenant_ml.markets.boosting_evaluation import evaluate_boosting
from covenant_ml.markets.forecast_records import (
    ForecastRow,
    decode_comparison_plan,
    decode_forecast_row,
    encode_comparison_plan,
    encode_forecast_row,
)

from ._forecast_fixtures import comparison_plan
from .test_boosting_model import training_rows


def fitted_report(tmp_path: Path, rows: tuple[ForecastRow, ...]) -> JSONObject:
    plan = encode_comparison_plan(comparison_plan())
    plan.update(start="2026-01-01", train_end="2026-03-01", end="2026-03-21")
    return evaluate_boosting(
        rows,
        decode_comparison_plan(plan),
        ("2026-03-11", "2026-03-21"),
        tmp_path / "models",
        "fixed",
    )


def test_actual_fits_replay_and_match_independent_loss_arithmetic(tmp_path: Path) -> None:
    source = fitted_report(tmp_path, training_rows())
    result = audit_baseline(source)
    assert result["contracts"] == 40
    assert len(require_list(result, "event_dates")) == 20
    assert result["verified_scores"] == source["scores"]
    differences: list[float] = []
    for raw in require_list(source, "folds"):
        fold = narrow_json_to_dict(raw)
        arms = require_dict(require_dict(fold, "scores"), "arms")
        for row, prediction in zip(
            require_list(fold, "evaluation_rows"),
            require_list(require_dict(arms, "logistic"), "predictions"),
            strict=True,
        ):
            observation = narrow_json_to_dict(row)
            label = require_int(observation, "yes")
            probability = require_float({"p": prediction}, "p")
            market = (require_int(observation, "bid") + require_int(observation, "ask")) / 20000
            differences.append((probability - label) ** 2 - (market - label) ** 2)
    comparison = require_dict(result, "logistic_minus_raw_market")
    assert require_float(comparison, "mean") == pytest.approx(mean(differences), abs=1e-14)
    assert require_float(comparison, "mean") < -0.1
    policies = require_dict(result, "policies")
    assert require_dict(policies, "raw_market")["trade_dates"] == []
    assert require_dict(policies, "logistic")["trade_dates"] == result["event_dates"]
    assert require_dict(require_dict(policies, "logistic"), "policy")["trades"] == 40


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("no_folds", "002"),
        ("no_rows", "005"),
        ("duplicate", "005"),
        ("overlap", "005"),
        ("day", "005"),
        ("cutoff", "005"),
        ("late_label", "005"),
        ("market", "003"),
        ("logistic", "003"),
        ("fold_score", "004"),
        ("pooled_score", "004"),
    ],
)
def test_evidence_mutations_are_rejected(tmp_path: Path, mutation: str, code: str) -> None:
    source = fitted_report(tmp_path, training_rows())
    folds = require_list(source, "folds")
    first = narrow_json_to_dict(folds[0])
    rows = require_list(first, "evaluation_rows")
    if mutation == "no_folds":
        source["folds"] = []
    elif mutation == "no_rows":
        first["evaluation_rows"] = []
    elif mutation == "duplicate":
        first["evaluation_rows"] = [*rows, rows[0]]
    elif mutation == "overlap":
        require_dict(narrow_json_to_dict(folds[1]), "plan")["train_end"] = "2026-03-10"
    elif mutation == "day":
        require_dict(first, "plan")["train_end"] = "2026-03-02"
    elif mutation == "cutoff":
        first["earliest_evaluation_decision_ms"] = 0
    elif mutation == "late_label":
        first["latest_training_settlement_ms"] = first["earliest_evaluation_decision_ms"]
    elif mutation in ("market", "logistic"):
        name = "raw_market" if mutation == "market" else mutation
        arm = require_dict(require_dict(require_dict(first, "scores"), "arms"), name)
        predictions = require_list(arm, "predictions")
        arm["predictions"] = [0.123, *predictions[1:]]
    else:
        scores = require_dict(first if mutation == "fold_score" else source, "scores")
        require_dict(require_dict(scores, "arms"), "logistic")["date_mean_brier"] = 0.0
    with pytest.raises(ValueError, match="BASELINE-" + code):
        audit_baseline(source)


def test_unequal_contract_counts_cannot_change_the_date_weighting(tmp_path: Path) -> None:
    """Distinguish row weighting from date weighting and independently check paired error."""
    inputs: list[ForecastRow] = []
    for row in training_rows():
        if "2026-03-02" <= row["event_day"] <= "2026-03-06":
            if row["yes"] == 0:
                continue
            document = encode_forecast_row(row)
            document["weather_p"] = 0.5
            inputs.append(decode_forecast_row(document))
        else:
            inputs.append(row)
    source = fitted_report(tmp_path, tuple(inputs))
    result = audit_baseline(source)
    per_date: dict[str, list[float]] = defaultdict(list)
    per_row: list[float] = []
    for value in require_list(source, "folds"):
        fold = narrow_json_to_dict(value)
        arms = require_dict(require_dict(fold, "scores"), "arms")
        for raw, probability in zip(
            require_list(fold, "evaluation_rows"),
            require_list(require_dict(arms, "logistic"), "predictions"),
            strict=True,
        ):
            observation = narrow_json_to_dict(raw)
            label = require_int(observation, "yes")
            forecast = require_float({"p": probability}, "p")
            market = (require_int(observation, "bid") + require_int(observation, "ask")) / 20000
            difference = (forecast - label) ** 2 - (market - label) ** 2
            per_date[require_str(observation, "event_day")].append(difference)
            per_row.append(difference)
    daily = tuple(mean(per_date[day]) for day in sorted(per_date))
    expected = mean(daily)
    error = stdev(daily) / sqrt(20)
    # Independent reference for the two-sided 95% Student-t critical value, df=19.
    half_width = 2.093024054408263 * error
    comparison = require_dict(result, "logistic_minus_raw_market")
    interval = require_dict(comparison, "date_clustered")
    assert result["contracts"] == 35
    assert len(daily) == 20
    assert sorted(len(values) for values in per_date.values()) == [1] * 5 + [2] * 15
    assert abs(expected - mean(per_row)) > 0.01
    assert require_list(comparison, "daily_differences") == pytest.approx(daily, abs=1e-14)
    assert require_float(comparison, "mean") == pytest.approx(expected, abs=1e-14)
    assert require_float(interval, "std_error") == pytest.approx(error, abs=1e-14)
    assert require_float(interval, "ci_low") == pytest.approx(expected - half_width, abs=1e-12)
    assert require_float(interval, "ci_high") == pytest.approx(expected + half_width, abs=1e-12)
    assert interval["n_units"] == 20
    assert interval["confidence"] == 0.95
    for name in ("raw_market", "logistic", "boosting"):
        policy = require_dict(require_dict(result, "policies"), name)
        original = require_dict(require_dict(require_dict(source, "scores"), "arms"), name)
        assert policy["policy"] == original["policy"]
