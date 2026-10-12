"""Chronological logistic calibration using the existing strict learner factory."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, JSONValue, require_float, require_list, require_str
from typing_extensions import ReadOnly, TypedDict

from ..backends.logreg.backend import _create_logreg_model
from .forecast_records import ComparisonPlan, ForecastFeatures, ForecastRow
from .validation import require_bounded_int

MARKET_FEATURES = ("market_logit", "boston", "philadelphia")
WEATHER_FEATURES = (*MARKET_FEATURES, "forecast_logit")


class FittedLogit(TypedDict):
    """Portable fitted parameters, independent of mutable sklearn model state."""

    features: ReadOnly[tuple[str, ...]]
    coefficients: ReadOnly[tuple[float, ...]]
    intercept: ReadOnly[float]
    iterations: ReadOnly[int]


def encode_fitted_logit(model: FittedLogit) -> JSONObject:
    """Encode fitted weights and the exact feature ordering.

    Args:
        model: Validated fitted parameters.

    Returns:
        JSON parameter record.
    """
    features: list[JSONValue] = list(model["features"])
    coefficients: list[JSONValue] = list(model["coefficients"])
    return {
        "features": features,
        "coefficients": coefficients,
        "intercept": model["intercept"],
        "iterations": model["iterations"],
    }


def decode_fitted_logit(document: JSONObject) -> FittedLogit:
    """Validate feature identities, finite weights and convergence metadata.

    Args:
        document: Decoded fitted parameter record.

    Returns:
        Immutable fitted parameters.

    Raises:
        ValueError: FORECAST-006 for invalid features or coefficients.
        TypeError: Incorrect native field types.
    """
    features = tuple(
        require_str({"feature": v}, "feature") for v in require_list(document, "features")
    )
    coefficients = tuple(
        require_float({"weight": v}, "weight") for v in require_list(document, "coefficients")
    )
    intercept = require_float(document, "intercept")
    if features not in (MARKET_FEATURES, WEATHER_FEATURES) or len(features) != len(coefficients):
        raise ValueError("[FORECAST-006] invalid logistic feature contract")
    if not all(math.isfinite(v) for v in (*coefficients, intercept)):
        raise ValueError("[FORECAST-006] fitted coefficients must be finite")
    return FittedLogit(
        features=features,
        coefficients=coefficients,
        intercept=intercept,
        iterations=require_bounded_int(document, "iterations", 0, 1999),
    )


def logit(probability: float) -> float:
    """Apply the protocol's fixed clipping before log odds.

    Args:
        probability: Validated probability in the unit interval.

    Returns:
        Finite log odds with a fixed 0.0001 tail floor.
    """
    clipped = min(0.9999, max(0.0001, probability))
    return math.log(clipped / (1 - clipped))


def features(row: ForecastFeatures, weather: bool) -> tuple[float, ...]:
    """Build the fixed nested baseline or forecast-augmented feature vector.

    Args:
        row: Validated observation.
        weather: Include the single additional weather logit.

    Returns:
        Feature values in the registered order.
    """
    base = (
        logit((row["bid"] + row["ask"]) / 20000),
        float(row["series"] == "KXHIGHTBOS"),
        float(row["series"] == "KXHIGHPHIL"),
    )
    return (*base, logit(row["weather_p"])) if weather else base


def chronological_split(
    rows: Sequence[ForecastRow],
    plan: ComparisonPlan,
) -> tuple[tuple[ForecastRow, ...], tuple[ForecastRow, ...], int]:
    """Keep whole later dates out and exclude labels unavailable at first prediction.

    Args:
        rows: Complete unique normalized observations.
        plan: Fixed date partitions.

    Returns:
        Training rows, evaluation rows, unavailable-training-label count.

    Raises:
        ValueError: FORECAST-007 for empty partitions, duplicate IDs or off-window rows.
    """
    if len({r["ticker"] for r in rows}) != len(rows):
        raise ValueError("[FORECAST-007] duplicated comparison contract")
    if any(
        not plan["start"] <= r["event_day"] <= plan["end"]
        or r["event_day"] in plan["excluded_days"]
        for r in rows
    ):
        raise ValueError("[FORECAST-007] row outside frozen comparison window")

    def chronological_key(row: ForecastRow) -> tuple[str, str]:
        """Return event date and ticker as the deterministic ordering key.

        Args:
            row: Validated observation.

        Returns:
            Chronological key with an identity tie breaker.
        """
        return row["event_day"], row["ticker"]

    ordered = sorted(rows, key=chronological_key)
    test = tuple(r for r in ordered if r["event_day"] > plan["train_end"])
    if not test:
        raise ValueError("[FORECAST-007] no held-out observations")
    first_decision = min(r["decision_ms"] for r in test)
    candidates = tuple(r for r in ordered if r["event_day"] <= plan["train_end"])
    train = tuple(r for r in candidates if r["settled_ms"] < first_decision)
    if not train:
        raise ValueError("[FORECAST-007] no training labels available before evaluation")
    return train, test, len(candidates) - len(train)


def fit_logit(rows: Sequence[ForecastRow], weather: bool, max_iter: int = 2000) -> FittedLogit:
    """Fit the predeclared L2 learner exclusively on supplied earlier observations.

    Args:
        rows: Chronologically filtered training observations.
        weather: Whether the weather logit is included.
        max_iter: Iteration limit; tests exercise real nonconvergence at one iteration.

    Returns:
        Immutable fitted coefficients and actual convergence iteration count.

    Raises:
        ValueError: FORECAST-008 for missing classes or iteration-limit exhaustion.
    """
    if {row["yes"] for row in rows} != {0, 1} or not 1 <= max_iter <= 2000:
        raise ValueError("[FORECAST-008] require both classes and a valid iteration limit")
    feature_values: list[tuple[float, ...]] = [features(row, weather) for row in rows]
    label_values: list[int] = [row["yes"] for row in rows]
    x: NDArray[np.float64] = np.asarray(feature_values, dtype=np.float64)
    y: NDArray[np.int64] = np.asarray(label_values, dtype=np.int64)
    learner = _create_logreg_model(
        penalty="l2",
        inverse_reg_strength=1.0,
        solver="lbfgs",
        max_iter=max_iter,
        tol=1e-10,
        random_state=0,
        class_weight=None,
        l1_ratio=None,
    )
    learner.fit(x, y)
    iterations = int(np.max(learner.n_iter_))
    if iterations >= max_iter:
        raise ValueError("[FORECAST-008] logistic solver exhausted its iteration limit")
    names = WEATHER_FEATURES if weather else MARKET_FEATURES
    return decode_fitted_logit(
        encode_fitted_logit(
            FittedLogit(
                features=names,
                coefficients=float_values(learner.coef_.ravel()),
                intercept=float_values(learner.intercept_)[0],
                iterations=iterations,
            )
        )
    )


def float_values(values: NDArray[np.float64]) -> tuple[float, ...]:
    """Convert numeric scalar elements to serializable Python floats.

    Args:
        values: Numeric array whose flattened order must be retained.

    Returns:
        Exactly typed immutable scalar values.
    """
    result: list[float] = []
    for index in range(values.size):
        value: np.float64 = values.flat[index]
        result.append(float(value))
    return tuple(result)


def predict_logit(model: FittedLogit, rows: Sequence[ForecastFeatures]) -> tuple[float, ...]:
    """Evaluate saved coefficients without access to outcomes or sklearn state.

    Args:
        model: Validated fitted coefficients.
        rows: Observations whose features were available at their decisions.

    Returns:
        Positive-class probabilities in input order.
    """
    values: list[float] = []
    for row in rows:
        vector = features(row, model["features"] == WEATHER_FEATURES)
        score = model["intercept"] + sum(
            w * v for w, v in zip(model["coefficients"], vector, strict=True)
        )
        exponential = math.exp(-abs(score))
        values.append(1 / (1 + exponential) if score >= 0 else exponential / (1 + exponential))
    return tuple(values)


__all__ = [
    "FittedLogit",
    "chronological_split",
    "decode_fitted_logit",
    "encode_fitted_logit",
    "features",
    "fit_logit",
    "logit",
    "predict_logit",
]
