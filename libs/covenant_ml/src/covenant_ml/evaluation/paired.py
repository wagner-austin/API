"""Paired differences of per-row quantities, with unit-aware standard errors.

A loss or a realised payoff is a MEAN over rows, so two models' difference is
the mean of their per-row differences ``d_i = a_i - b_i``. What changes with
the data is only the standard error of that mean:

``row``
    Rows independent: ``sd(d) / sqrt(n)``, Student's t on ``n - 1`` df.
``cluster``
    Rows within a cluster move together: the variance is summed within each
    cluster first (the cluster-robust "sandwich" estimator with the
    ``G / (G - 1)`` small-sample factor), t on ``G - 1`` df.
``block``
    Rows are a time series: the Newey-West long-run variance with a Bartlett
    kernel over ``block_length - 1`` lags, normal reference. This is the
    Diebold-Mariano test of equal forecast accuracy, which is what a market
    forecast is compared against another with.

AUC is not a mean over rows, so it is handled by :mod:`.auc` instead, and
:func:`compare_probabilities` routes each metric to the right estimator.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..metrics import brier_per_row, log_loss_per_row
from .auc import bootstrap_auc_difference, delong_auc_difference
from .inputs import require_observations
from .intervals import normal_interval, t_interval
from .resampling import Resampling, unit_sums
from .types import (
    ERR_INVALID_ARGUMENT,
    ERR_LENGTH_MISMATCH,
    ERR_TOO_FEW_UNITS,
    ComparisonMetric,
    IntervalEstimate,
    IntervalMethod,
    PairedComparison,
    ResamplingUnit,
)


def _mean(values: NDArray[np.float64]) -> float:
    """Arithmetic mean as a plain float.

    Args:
        values: At least one value.

    Returns:
        The mean.
    """
    return float(np.sum(values)) / len(values)


def _cluster_robust_error(
    differences: NDArray[np.float64],
    mean: float,
    resampling: Resampling,
) -> float:
    """Standard error of a mean whose rows are clustered.

    Args:
        differences: Per-row values.
        mean: Their mean.
        resampling: The cluster of each row.

    Returns:
        The cluster-robust standard error.
    """
    n_rows = len(differences)
    n_units = resampling.n_units
    sums = unit_sums(differences, resampling)
    sizes = unit_sums(np.ones(n_rows, dtype=np.float64), resampling)
    residual: NDArray[np.float64] = sums - sizes * mean
    variance = (n_units / (n_units - 1)) * float(np.sum(residual * residual)) / (n_rows * n_rows)
    return math.sqrt(variance)


def _newey_west_error(differences: NDArray[np.float64], mean: float, lags: int) -> float:
    """Standard error of a time-series mean from its long-run variance.

    The Bartlett weights ``1 - k / (lags + 1)`` keep the estimate
    non-negative, which a plain sum of autocovariances does not.

    Args:
        differences: Per-row values in time order.
        mean: Their mean.
        lags: Autocovariance lags to include, at least 1.

    Returns:
        The Newey-West standard error.
    """
    n_rows = len(differences)
    centred: NDArray[np.float64] = differences - mean
    long_run = float(np.sum(centred * centred)) / n_rows
    for lag in range(1, lags + 1):
        lead: NDArray[np.float64] = centred[lag:]
        lagged: NDArray[np.float64] = centred[:-lag]
        autocovariance = float(np.sum(lead * lagged)) / n_rows
        long_run += 2.0 * (1.0 - lag / (lags + 1)) * autocovariance
    return math.sqrt(long_run / n_rows)


def paired_mean_difference(
    differences: NDArray[np.float64],
    resampling: Resampling,
    confidence: float,
) -> IntervalEstimate:
    """Interval the mean of per-row differences.

    Args:
        differences: ``a_i - b_i`` for every held-out row, in row order.
        resampling: Which rows form one independent unit.
        confidence: Coverage of the interval.

    Returns:
        The mean difference with an interval built for the unit.

    Raises:
        ValueError: If ``differences`` does not cover the resampling's rows.
    """
    require_observations(differences, "paired differences")
    n_rows = len(differences)
    if n_rows != len(resampling.groups):
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] differences must cover {len(resampling.groups)} rows, "
            f"got {n_rows}"
        )
    if n_rows < 2:
        raise ValueError(f"[{ERR_TOO_FEW_UNITS}] paired inference needs at least two rows")
    mean = _mean(differences)
    if resampling.unit is ResamplingUnit.CLUSTER:
        return t_interval(
            IntervalMethod.CLUSTER_ROBUST,
            mean,
            _cluster_robust_error(differences, mean, resampling),
            resampling.n_units - 1,
            confidence,
            resampling.n_units,
        )
    if resampling.unit is ResamplingUnit.BLOCK:
        return normal_interval(
            IntervalMethod.NEWEY_WEST,
            mean,
            _newey_west_error(differences, mean, resampling.block_length - 1),
            confidence,
            resampling.n_units,
        )
    centred: NDArray[np.float64] = differences - mean
    variance = float(np.sum(centred * centred)) / (n_rows - 1)
    return t_interval(
        IntervalMethod.PAIRED_T,
        mean,
        math.sqrt(variance / n_rows),
        n_rows - 1,
        confidence,
        n_rows,
    )


def compare_per_row(
    metric: ComparisonMetric,
    arm: str,
    reference: str,
    values_arm: NDArray[np.float64],
    values_reference: NDArray[np.float64],
    resampling: Resampling,
    confidence: float,
) -> PairedComparison:
    """Compare two arms on a quantity that is a mean over rows.

    Args:
        metric: What the per-row values are.
        arm: The arm under test.
        reference: The arm it is measured against.
        values_arm: The arm's per-row values.
        values_reference: The reference's per-row values.
        resampling: Which rows form one independent unit.
        confidence: Coverage of the interval.

    Returns:
        The comparison, differenced as ``arm - reference``.

    Raises:
        ValueError: If the two arrays differ in length or do not cover the
            resampling's rows.
    """
    require_observations(values_arm, "arm observations")
    require_observations(values_reference, "reference observations")
    if len(values_arm) != len(values_reference):
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] both arms must be scored on the same rows, got "
            f"{len(values_arm)} and {len(values_reference)}"
        )
    difference = paired_mean_difference(values_arm - values_reference, resampling, confidence)
    return {
        "metric": metric,
        "arm": arm,
        "reference": reference,
        "value_arm": _mean(values_arm),
        "value_reference": _mean(values_reference),
        "difference": difference,
        "n_rows": len(values_arm),
    }


def compare_probabilities(
    labels: NDArray[np.int64],
    proba_arm: NDArray[np.float64],
    proba_reference: NDArray[np.float64],
    arm: str,
    reference: str,
    resampling: Resampling,
    n_resamples: int,
    resample_seed: int,
    confidence: float,
) -> list[PairedComparison]:
    """Compare two arms' probabilities on AUC, log loss and Brier score.

    The AUC interval is DeLong's where rows are independent and a unit
    bootstrap otherwise; the two losses use :func:`paired_mean_difference`.

    Args:
        labels: Binary labels of the held-out rows.
        proba_arm: The arm's positive-class probabilities.
        proba_reference: The reference's positive-class probabilities.
        arm: The arm under test.
        reference: The arm it is measured against.
        resampling: Which rows form one independent unit.
        n_resamples: Bootstrap resamples, used only off the ``row`` unit.
        resample_seed: Seed of the bootstrap draws.
        confidence: Coverage of every interval.

    Returns:
        Three comparisons: AUC, log loss, Brier score.

    Raises:
        ValueError: If the arrays are misaligned, a class has too few rows,
            or a bootstrap resample draws one class only.
    """
    for probabilities in (proba_arm, proba_reference):
        require_observations(probabilities, "probabilities")
        if bool(np.any((probabilities < 0.0) | (probabilities > 1.0))):
            raise ValueError(f"[{ERR_INVALID_ARGUMENT}] probabilities must lie in [0, 1]")
    if resampling.unit is ResamplingUnit.ROW:
        auc = delong_auc_difference(labels, proba_arm, proba_reference, confidence)
    else:
        auc = bootstrap_auc_difference(
            labels,
            proba_arm,
            proba_reference,
            resampling,
            n_resamples,
            resample_seed,
            confidence,
        )
    auc_comparison: PairedComparison = {
        "metric": ComparisonMetric.AUC,
        "arm": arm,
        "reference": reference,
        "value_arm": auc.auc_arm,
        "value_reference": auc.auc_reference,
        "difference": auc.difference,
        "n_rows": len(labels),
    }
    return [
        auc_comparison,
        compare_per_row(
            ComparisonMetric.LOG_LOSS,
            arm,
            reference,
            log_loss_per_row(labels, proba_arm),
            log_loss_per_row(labels, proba_reference),
            resampling,
            confidence,
        ),
        compare_per_row(
            ComparisonMetric.BRIER,
            arm,
            reference,
            brier_per_row(labels, proba_arm),
            brier_per_row(labels, proba_reference),
            resampling,
            confidence,
        ),
    ]


__all__ = ["compare_per_row", "compare_probabilities", "paired_mean_difference"]
