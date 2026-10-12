"""Turning a point estimate and its spread into an :class:`IntervalEstimate`.

Every interval in this package is built here, by one of three shapes:

* a Student's t interval, when the standard error is a sample estimate over a
  known number of independent units (rows, clusters, or seeds);
* a normal interval, when the variance is asymptotic (DeLong, Newey-West);
* a percentile interval, when the uncertainty is a bootstrap distribution.

The t quantile and survival function are ``platform_core.power_distributions``',
the same ones the power module certifies, so this package and the minimum
detectable effect it is read beside agree on what "significant" means.

No ``_test_hooks.py``: every function is a pure transformation of its
arguments, with no clock, file, network or unseeded randomness behind it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray
from platform_core.power_distributions import t_critical, two_sided_t_survival

from .types import (
    ERR_INVALID_CONFIDENCE,
    ERR_INVALID_RESAMPLES,
    ERR_LENGTH_MISMATCH,
    ERR_TOO_FEW_SEEDS,
    IntervalEstimate,
    IntervalMethod,
)

_STANDARD_NORMAL = NormalDist()


def require_confidence(confidence: float) -> None:
    """Refuse a coverage outside ``(0, 1)``.

    Args:
        confidence: Candidate coverage, e.g. 0.95.

    Raises:
        ValueError: If ``confidence`` is not strictly between 0 and 1.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"[{ERR_INVALID_CONFIDENCE}] confidence must be in (0, 1), got {confidence}"
        )


def _degenerate(
    method: IntervalMethod,
    estimate: float,
    confidence: float,
    n_units: int,
) -> IntervalEstimate:
    """The interval of an estimate with zero spread.

    Every unit agreed exactly, so the interval collapses onto the estimate:
    a nonzero estimate with no spread is certain (p = 0), and a zero one is
    no evidence of anything (p = 1).

    Args:
        method: How the spread was measured.
        estimate: The point estimate.
        confidence: Requested coverage.
        n_units: Independent units behind it.

    Returns:
        The collapsed interval.
    """
    return {
        "method": method,
        "estimate": estimate,
        "std_error": 0.0,
        "ci_low": estimate,
        "ci_high": estimate,
        "confidence": confidence,
        "p_value": 1.0 if estimate == 0.0 else 0.0,
        "n_units": n_units,
    }


def t_interval(
    method: IntervalMethod,
    estimate: float,
    std_error: float,
    degrees_of_freedom: int,
    confidence: float,
    n_units: int,
) -> IntervalEstimate:
    """Build a Student's t interval.

    Args:
        method: How the standard error was computed.
        estimate: The point estimate.
        std_error: Its standard error, non-negative.
        degrees_of_freedom: Degrees of freedom, at least 1.
        confidence: Coverage in ``(0, 1)``.
        n_units: Independent units behind the estimate.

    Returns:
        The interval, with a two-sided p-value against zero.

    Raises:
        ValueError: If ``confidence`` is outside ``(0, 1)``.
        AppError: If ``degrees_of_freedom`` is below 1, from the shared t
            quantile.
    """
    require_confidence(confidence)
    if std_error == 0.0:
        return _degenerate(method, estimate, confidence, n_units)
    critical = t_critical(degrees_of_freedom, 1.0 - confidence)
    statistic = abs(estimate) / std_error
    return {
        "method": method,
        "estimate": estimate,
        "std_error": std_error,
        "ci_low": estimate - critical * std_error,
        "ci_high": estimate + critical * std_error,
        "confidence": confidence,
        "p_value": two_sided_t_survival(statistic, degrees_of_freedom),
        "n_units": n_units,
    }


def normal_interval(
    method: IntervalMethod,
    estimate: float,
    std_error: float,
    confidence: float,
    n_units: int,
) -> IntervalEstimate:
    """Build a normal interval from an asymptotic standard error.

    Args:
        method: How the standard error was computed.
        estimate: The point estimate.
        std_error: Its standard error, non-negative.
        confidence: Coverage in ``(0, 1)``.
        n_units: Independent units behind the estimate.

    Returns:
        The interval, with a two-sided p-value against zero.

    Raises:
        ValueError: If ``confidence`` is outside ``(0, 1)``.
    """
    require_confidence(confidence)
    if std_error == 0.0:
        return _degenerate(method, estimate, confidence, n_units)
    critical = _STANDARD_NORMAL.inv_cdf(0.5 + confidence / 2.0)
    statistic = abs(estimate) / std_error
    return {
        "method": method,
        "estimate": estimate,
        "std_error": std_error,
        "ci_low": estimate - critical * std_error,
        "ci_high": estimate + critical * std_error,
        "confidence": confidence,
        "p_value": 2.0 * (1.0 - _STANDARD_NORMAL.cdf(statistic)),
        "n_units": n_units,
    }


def _quantile(ordered: NDArray[np.float64], probability: float) -> float:
    """Read a quantile off sorted draws by linear interpolation.

    The interpolation is numpy's default (``index = q * (n - 1)``), written
    out so the result is a plain float.

    Args:
        ordered: Draws sorted ascending, at least one.
        probability: Quantile in ``[0, 1]``.

    Returns:
        The interpolated quantile.
    """
    position = probability * (len(ordered) - 1)
    below = math.floor(position)
    above = min(below + 1, len(ordered) - 1)
    weight = position - below
    low = float(ordered.flat[below].item())
    high = float(ordered.flat[above].item())
    return low + weight * (high - low)


def percentile_interval(
    estimate: float,
    draws: NDArray[np.float64],
    confidence: float,
    n_units: int,
) -> IntervalEstimate:
    """Build a percentile interval from a bootstrap distribution.

    The p-value counts the draws on each side of zero with the add-one
    correction, ``(count + 1) / (B + 1)``, so a bootstrap of B draws never
    reports a p below ``2 / (B + 1)``: the resolution the draws actually
    have, rather than a zero they cannot support.

    Args:
        estimate: The point estimate on the original sample.
        draws: The statistic on each resample.
        confidence: Coverage in ``(0, 1)``.
        n_units: Independent units each resample drew.

    Returns:
        The interval.

    Raises:
        ValueError: If ``confidence`` is outside ``(0, 1)``, or fewer than two
            draws were given, which leaves no spread to measure.
    """
    require_confidence(confidence)
    n_draws = len(draws)
    if n_draws < 2:
        raise ValueError(
            f"[{ERR_INVALID_RESAMPLES}] a percentile interval needs at least 2 draws, got {n_draws}"
        )
    ordered: NDArray[np.float64] = np.sort(draws)
    mean = float(np.sum(draws)) / n_draws
    centred: NDArray[np.float64] = draws - mean
    std_error = math.sqrt(float(np.sum(centred * centred)) / (n_draws - 1))
    at_or_below = int(np.sum(draws <= 0.0))
    at_or_above = int(np.sum(draws >= 0.0))
    tail = min(at_or_below, at_or_above)
    alpha = 1.0 - confidence
    return {
        "method": IntervalMethod.UNIT_BOOTSTRAP,
        "estimate": estimate,
        "std_error": std_error,
        "ci_low": _quantile(ordered, alpha / 2.0),
        "ci_high": _quantile(ordered, 1.0 - alpha / 2.0),
        "confidence": confidence,
        "p_value": min(1.0, 2.0 * (tail + 1) / (n_draws + 1)),
        "n_units": n_units,
    }


def pool_across_seeds(
    per_seed: Sequence[float],
    within_seed_errors: Sequence[float],
    test_train_ratio: float,
    confidence: float,
) -> IntervalEstimate:
    """Carry one difference per seed into a t interval over seeds.

    THE NAIVE t OVER SEEDS IS TOO NARROW, measured on the first real run of
    this package (2026-10-12, US bankruptcy corpus, two seeds): XGBoost's AUC
    lead over LightGBM was +0.0024 and +0.0025 at the two seeds, each with a
    within-seed interval of about +/-0.007, and the naive pooled t read p =
    0.0097 from two numbers that happened to agree. Two seeds' held-out sets
    share companies, so their differences are correlated and their spread
    understates the uncertainty of their mean. Two corrections, both applied:

    * Nadeau and Bengio (2003, "Inference for the Generalization Error"): for
      ``k`` random splits with ``n_test / n_train`` held out, the variance of
      the mean is ``(1/k + n_test/n_train) * s**2``, not ``s**2 / k``.
    * A floor at the mean within-seed standard error. Where the partition is
      the SAME at every seed (a chronological split), the seeds vary only
      the model, so their spread carries none of the held-out sampling
      uncertainty at all; no pooled interval may be narrower than the seeds'
      own intervals, on average, say it is.

    Args:
        per_seed: The within-seed point difference at each seed.
        within_seed_errors: Each seed's own standard error of that difference.
        test_train_ratio: Held-out rows over training rows, averaged over
            seeds.
        confidence: Coverage in ``(0, 1)``.

    Returns:
        The mean difference with a t interval on ``len(per_seed) - 1`` degrees
        of freedom.

    Raises:
        ValueError: If fewer than two seeds were given, or the two sequences
            differ in length.
    """
    n_seeds = len(per_seed)
    if n_seeds < 2:
        raise ValueError(
            f"[{ERR_TOO_FEW_SEEDS}] pooling across seeds needs at least 2 seeds, got {n_seeds}"
        )
    if len(within_seed_errors) != n_seeds:
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] one within-seed error per seed is needed, got "
            f"{len(within_seed_errors)} for {n_seeds} seeds"
        )
    mean = sum(per_seed) / n_seeds
    variance = sum((value - mean) ** 2 for value in per_seed) / (n_seeds - 1)
    corrected = math.sqrt((1.0 / n_seeds + test_train_ratio) * variance)
    floor = sum(within_seed_errors) / n_seeds
    return t_interval(
        IntervalMethod.ACROSS_SEEDS,
        mean,
        max(corrected, floor),
        n_seeds - 1,
        confidence,
        n_seeds,
    )


__all__ = [
    "normal_interval",
    "percentile_interval",
    "pool_across_seeds",
    "require_confidence",
    "t_interval",
]
