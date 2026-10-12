"""Nested chronological forecast comparison with fixed primary and economic diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, JSONValue, require_float

from ..evaluation.codec_records import encode_interval
from ..evaluation.paired import paired_mean_difference
from ..evaluation.resampling import build_resampling
from ..evaluation.types import ResamplingUnit
from ..metrics import brier_per_row, log_loss_per_row
from .forecast_model import (
    chronological_split,
    encode_fitted_logit,
    fit_logit,
    float_values,
    predict_logit,
)
from .forecast_policy import cluster_interval, quoted_policy
from .forecast_records import (
    ComparisonPlan,
    ForecastRow,
    encode_comparison_plan,
    encode_forecast_row,
)


def _daily_interval(differences: Sequence[float], rows: Sequence[ForecastRow]) -> JSONObject | None:
    """Aggregate paired losses by represented date and apply seven observed-date lags.

    Args:
        differences: Per-row paired loss differences.
        rows: Held-out observations in the same order.

    Returns:
        Newey-West interval, or explicit null when fewer than nine dates exist.
    """
    by_date: dict[str, list[float]] = defaultdict(list)
    for value, row in zip(differences, rows, strict=True):
        by_date[row["event_day"]].append(value)
    if len(by_date) < 9:
        return None
    means: list[float] = [sum(by_date[day]) / len(by_date[day]) for day in sorted(by_date)]
    values: NDArray[np.float64] = np.asarray(means, dtype=np.float64)
    indices: NDArray[np.int64] = np.arange(len(means), dtype=np.int64)
    units = build_resampling(ResamplingUnit.BLOCK, len(means), indices, 8)
    return encode_interval(paired_mean_difference(values, units, 0.95))


def evaluate_forecasts(rows: Sequence[ForecastRow], plan: ComparisonPlan) -> JSONObject:
    """Fit two nested earlier-period models and evaluate all frozen held-out predictions.

    Args:
        rows: Complete normalized comparison panel.
        plan: Frozen date partitions and protocol identity.

    Returns:
        Parameters, predictions, losses, paired intervals and quoted execution diagnostics.

    Raises:
        ValueError: Invalid splits, unavailable labels, insufficient classes or dates.
    """
    train, test, unavailable = chronological_split(rows, plan)
    baseline, augmented = fit_logit(train, False), fit_logit(train, True)
    predictions: dict[str, tuple[float, ...]] = {
        "raw_market": tuple((row["bid"] + row["ask"]) / 20000 for row in test),
        "raw_forecast": tuple(row["weather_p"] for row in test),
        "market_calibrated": predict_logit(baseline, test),
        "market_plus_forecast": predict_logit(augmented, test),
    }
    label_values: list[int] = [row["yes"] for row in test]
    labels: NDArray[np.int64] = np.asarray(label_values, dtype=np.int64)
    losses: dict[str, tuple[float, ...]] = {}
    arms: JSONObject = {}
    for name, probabilities in predictions.items():
        values: NDArray[np.float64] = np.asarray(probabilities, dtype=np.float64)
        brier = brier_per_row(labels, values)
        losses[name] = float_values(brier)
        encoded_predictions: list[JSONValue] = list(probabilities)
        arms[name] = {
            "brier": sum(losses[name]) / len(test),
            "log_loss": sum(float_values(log_loss_per_row(labels, values))) / len(test),
            "predictions": encoded_predictions,
            "policy": quoted_policy(test, probabilities),
        }
    comparisons: JSONObject = {}
    for reference in ("market_calibrated", "raw_market"):
        difference = tuple(
            a - b for a, b in zip(losses["market_plus_forecast"], losses[reference], strict=True)
        )
        interval = cluster_interval(difference, test)
        comparisons[reference] = {
            "brier_difference": interval,
            "daily_newey_west": _daily_interval(difference, test),
            "candidate_gate": require_float(interval, "ci_high") < 0,
        }
    encoded_train: list[JSONValue] = [encode_forecast_row(row) for row in train]
    encoded_test: list[JSONValue] = [encode_forecast_row(row) for row in test]
    return {
        "schema_version": 1,
        "plan": encode_comparison_plan(plan),
        "training_rows": encoded_train,
        "evaluation_rows": encoded_test,
        "unavailable_training_labels": unavailable,
        "models": {
            "market_calibrated": encode_fitted_logit(baseline),
            "market_plus_forecast": encode_fitted_logit(augmented),
        },
        "arms": arms,
        "comparisons": comparisons,
        "primary_reference": "market_calibrated",
        "interpretation": (
            "Historical chronological validation; assumed publication lag and hypothetical fills."
        ),
    }


__all__ = ["evaluate_forecasts"]
