"""Earlier-only logistic recalibration with fixed regularization and feature selection."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..backends.logreg.backend import _create_logreg_model
from .forecast_model import float_values, logit
from .interval_quotes import SERIES
from .interval_records import IntervalFit, IntervalRow, decode_interval_fit, encode_interval_fit


def interval_features(row: IntervalRow) -> tuple[float, float]:
    """Select prior market log odds and asset identity, never settlement data.

    Args:
        row: Validated comparison observation.

    Returns:
        Fixed two-feature vector.
    """
    return logit((row["bid"] + row["ask"]) / 20000), float(row["series"] == "KXETH15M")


def fit_interval_model(rows: Sequence[IntervalRow], max_iter: int = 2000) -> IntervalFit:
    """Fit the shared real learner on an explicitly selected earlier partition.

    Args:
        rows: Training observations with labels already available at the cutoff.
        max_iter: Solver limit; real nonconvergence is exercised in tests.

    Returns:
        Portable coefficients and per-asset smoothed base rates.

    Raises:
        ValueError: IFORE-003 for absent classes/assets; IFORE-004 for nonconvergence.
    """
    if {row["series"] for row in rows} != set(SERIES) or {row["yes"] for row in rows} != {0, 1}:
        raise ValueError("[IFORE-003] fitting requires both assets and both binary classes")
    if not 1 <= max_iter <= 2000:
        raise ValueError("[IFORE-004] solver iteration budget must lie in [1, 2000]")
    features: list[tuple[float, float]] = [interval_features(row) for row in rows]
    labels_all: list[int] = [row["yes"] for row in rows]
    x: NDArray[np.float64] = np.asarray(features, dtype=np.float64)
    y: NDArray[np.int64] = np.asarray(labels_all, dtype=np.int64)
    model = _create_logreg_model(
        penalty="l2",
        inverse_reg_strength=1.0,
        solver="lbfgs",
        max_iter=max_iter,
        tol=1e-10,
        random_state=0,
        class_weight=None,
        l1_ratio=None,
    )
    model.fit(x, y)
    iterations = int(np.max(model.n_iter_))
    if iterations >= max_iter:
        raise ValueError("[IFORE-004] logistic solver exhausted its iteration budget")
    rates: list[float] = []
    for series in SERIES:
        labels = [row["yes"] for row in rows if row["series"] == series]
        rates.append((sum(labels) + 0.5) / (len(labels) + 1))
    weights = float_values(model.coef_.ravel())
    return decode_interval_fit(
        encode_interval_fit(
            IntervalFit(
                intercept=float_values(model.intercept_)[0],
                market_weight=weights[0],
                ether_weight=weights[1],
                bitcoin_base=rates[0],
                ether_base=rates[1],
                iterations=iterations,
            )
        )
    )


def predict_interval_model(model: IntervalFit, row: IntervalRow) -> float:
    """Predict from saved coefficients and the two permitted input features.

    Args:
        model: Validated frozen parameter record.
        row: Observation; labels and settlement times are not read.

    Returns:
        Stable logistic probability.
    """
    market, ether = interval_features(row)
    score = model["intercept"] + model["market_weight"] * market + model["ether_weight"] * ether
    exponential = math.exp(-abs(score))
    return 1 / (1 + exponential) if score >= 0 else exponential / (1 + exponential)
