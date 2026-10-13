"""Exploratory chronological comparison of fixed and city-calibrated weather models."""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.json_utils import JSONObject, require_dict, require_float, require_list

from .calibration_model import (
    BIAS_GRID,
    MINIMUM_DATES,
    SIGMA_GRID,
    apply_city_calibration,
    encode_city_calibration,
    fit_city_calibration,
)
from .forecast_evaluation import evaluate_forecasts
from .forecast_model import chronological_split
from .forecast_policy import cluster_interval
from .forecast_records import CITY_COORDINATES, ComparisonPlan, ForecastRow


def evaluate_city_calibration(
    rows: Sequence[ForecastRow],
    plan: ComparisonPlan,
    bounds: dict[str, tuple[float | None, float | None]],
) -> JSONObject:
    """Fit city error distributions on earlier dates and reuse the complete evaluator.

    Args:
        rows: Verified historical observations including held-out later dates.
        plan: Original chronological split, unchanged by this experiment.
        bounds: Verified native bracket boundaries for every observation.

    Returns:
        Both complete evaluations, fitted city parameters and paired loss comparisons.

    Raises:
        ValueError: Insufficient training dates, classes or invalid chronology.
        KeyError: A required native bracket is missing.
    """
    train, test, _ = chronological_split(rows, plan)
    models = {series: fit_city_calibration(train, bounds, series) for series in CITY_COORDINATES}
    fixed = evaluate_forecasts(rows, plan)
    calibrated = evaluate_forecasts(apply_city_calibration(rows, bounds, models), plan)
    fixed_arms, calibrated_arms = require_dict(fixed, "arms"), require_dict(calibrated, "arms")
    comparisons: JSONObject = {}
    for name in ("raw_forecast", "market_plus_forecast"):
        before = require_list(require_dict(fixed_arms, name), "predictions")
        after = require_list(require_dict(calibrated_arms, name), "predictions")
        differences = tuple(
            (require_float({"p": new}, "p") - row["yes"]) ** 2
            - (require_float({"p": old}, "p") - row["yes"]) ** 2
            for old, new, row in zip(before, after, test, strict=True)
        )
        comparisons[name] = cluster_interval(differences, test)
    return {
        "schema_version": 1,
        "bias_grid_f": list(BIAS_GRID),
        "sigma_grid_f": list(SIGMA_GRID),
        "minimum_training_dates": MINIMUM_DATES,
        "training_objective": "equal-date mean binary-bracket Brier per city",
        "tie_break": "smaller bias, then smaller sigma",
        "city_models": {series: encode_city_calibration(model) for series, model in models.items()},
        "fixed": fixed,
        "city_calibrated": calibrated,
        "calibrated_minus_fixed": comparisons,
        "interpretation": (
            "Exploratory chronological validation on previously examined later dates; "
            "not an untouched confirmation or executable-profit claim. "
            "Binary bracket labels do not reveal exact observed temperatures. "
            "Original prospective study remains frozen."
        ),
    }
