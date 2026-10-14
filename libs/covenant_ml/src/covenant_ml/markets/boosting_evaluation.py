"""Expanding chronological fits with complete predictions and model artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_bool,
    require_dict,
    require_float,
    require_int,
)

from .boosting_model import boosting_parameters, feature_matrix, fit_boosting
from .boosting_scores import score_predictions
from .boosting_selection import select_boosting
from .forecast_model import (
    chronological_split,
    encode_fitted_logit,
    fit_logit,
    float_values,
    predict_logit,
)
from .forecast_records import (
    ComparisonPlan,
    ForecastRow,
    decode_comparison_plan,
    encode_comparison_plan,
    encode_forecast_row,
)
from .validation import require_day


def evaluate_boosting(
    rows: Sequence[ForecastRow],
    plan: ComparisonPlan,
    fold_ends: tuple[str, str],
    artifacts: Path,
    study: Literal["fixed", "selection"],
) -> JSONObject:
    """Fit fixed models on earlier available labels and score two later periods.

    Args:
        rows: Verified source report's unique saved observations.
        plan: Original experiment's date window and initial training boundary.
        fold_ends: Two ordered evaluation endpoints, ending at the source plan end.
        artifacts: New experiment directory, reserved exclusively by this call.
        study: Fixed candidate or earlier-only capacity selection experiment.

    Returns:
        Fold identities, fitted logistic models, native booster names and pooled scores.

    Raises:
        ValueError: BOOST-004 for invalid folds, or shared row/split validation errors.
        FileExistsError: Artifact directory already exists.
        OSError: Evidence cannot be saved.
    """
    first, last = (require_day({"day": value}, "day") for value in fold_ends)
    if not plan["train_end"] < first < last or last != plan["end"]:
        raise ValueError("[BOOST-004] require two later folds covering the source evaluation")
    chronological_split(rows, plan)
    artifacts.mkdir(parents=True, exist_ok=False)
    folds: list[JSONValue] = []
    pooled_rows: list[ForecastRow] = []
    pooled_predictions: dict[str, tuple[float, ...]] = {
        "raw_market": (),
        "logistic": (),
        "boosting": (),
    }
    fold_means: list[float] = []
    train_end = plan["train_end"]
    for index, end in enumerate(fold_ends):
        document = encode_comparison_plan(plan)
        document["train_end"], document["end"] = train_end, end
        document["excluded_days"] = [day for day in plan["excluded_days"] if day <= end]
        fold_plan = decode_comparison_plan(document)
        train, test, excluded = chronological_split(
            tuple(row for row in rows if row["event_day"] <= end), fold_plan
        )
        baseline = fit_logit(train, True)
        model_name = f"fold-{index + 1}.txt"
        selection: JSONObject = {}
        weather = True
        if study == "selection":
            boosted, weather, selection = select_boosting(train, artifacts / model_name)
        else:
            boosted = fit_boosting(train, artifacts / model_name, 20, 4, True)
        probabilities = boosted.booster_.predict(feature_matrix(test, weather))
        predictions = {
            "raw_market": tuple((r["bid"] + r["ask"]) / 20000 for r in test),
            "logistic": predict_logit(baseline, test),
            "boosting": float_values(probabilities),
        }
        scores = score_predictions(test, predictions)
        comparison = require_dict(require_dict(scores, "comparisons"), "logistic")
        fold_means.append(require_float(comparison, "mean"))
        ordered_dates = sorted({row["event_day"] for row in train})
        training_dates: list[JSONValue] = list(ordered_dates)
        folds.append(
            {
                "plan": document,
                "training_tickers": [row["ticker"] for row in train],
                "training_dates": training_dates,
                "unavailable_training_labels": excluded,
                "latest_training_settlement_ms": max(row["settled_ms"] for row in train),
                "earliest_evaluation_decision_ms": min(row["decision_ms"] for row in test),
                "evaluation_rows": [encode_forecast_row(row) for row in test],
                "logistic_model": encode_fitted_logit(baseline),
                "boosting_model": model_name,
                "selection": selection,
                "scores": scores,
            }
        )
        pooled_rows.extend(test)
        for name in predictions:
            pooled_predictions[name] += predictions[name]
        train_end = end
    scores = score_predictions(pooled_rows, pooled_predictions)
    primary = require_dict(require_dict(scores, "comparisons"), "logistic")
    policy = require_dict(require_dict(require_dict(scores, "arms"), "boosting"), "policy")
    criteria = {
        "improves_both_folds": all(value < 0 for value in fold_means),
        "both_pooled_upper_bounds_negative": require_bool(primary, "both_upper_bounds_negative"),
        "extra_cent_profit_positive": require_int(policy, "extra_cent_pnl") > 0,
    }
    return {
        "schema_version": 1,
        "study": study,
        "hyperparameters": boosting_parameters(20, 4) if study == "fixed" else {},
        "folds": folds,
        "scores": scores,
        "criteria": dict(criteria),
        "development_screen_passed": all(criteria.values()),
        "interpretation": "Reused dates; hypothetical fills; no funded-trading approval.",
    }
