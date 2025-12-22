"""Out-of-fold (OOF) prediction utilities.

Provides functions for working with OOF predictions from cross-validation,
including metrics computation and stacking preparation.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from covenant_ml.metrics import compute_all_metrics, compute_auc
from covenant_ml.types import EvalMetrics
from covenant_ml.validation.types import CVResult


class OOFMetrics(TypedDict, total=True):
    """Metrics computed from out-of-fold predictions.

    OOF metrics provide an unbiased estimate of model performance
    since each sample is predicted by a model that never saw it.

    Attributes:
        oof_auc: AUC computed from all OOF predictions.
        mean_fold_auc: Mean of per-fold validation AUCs.
        std_fold_auc: Standard deviation of per-fold validation AUCs.
        eval_metrics: Full evaluation metrics from OOF predictions.
    """

    oof_auc: float
    mean_fold_auc: float
    std_fold_auc: float
    eval_metrics: EvalMetrics


def compute_oof_auc(
    y_true: NDArray[np.int64],
    oof_predictions: NDArray[np.float64],
) -> float:
    """Compute AUC from out-of-fold predictions.

    Args:
        y_true: True labels of shape (n_samples,).
        oof_predictions: OOF predictions of shape (n_samples,).

    Returns:
        AUC score computed on all samples using OOF predictions.
    """
    return compute_auc(y_true, oof_predictions)


def compute_oof_metrics(
    y_true: NDArray[np.int64],
    cv_result: CVResult,
) -> OOFMetrics:
    """Compute comprehensive metrics from cross-validation results.

    Uses OOF predictions for unbiased evaluation and compares with
    per-fold metrics for consistency checking.

    Args:
        y_true: True labels of shape (n_samples,).
        cv_result: Complete cross-validation result.

    Returns:
        OOFMetrics with AUC, fold statistics, and full eval metrics.
    """
    oof_predictions = cv_result["oof_predictions"]

    # Compute overall OOF AUC
    oof_auc = compute_auc(y_true, oof_predictions)

    # Compute full metrics on OOF predictions
    eval_metrics = compute_all_metrics(y_true, oof_predictions)

    return OOFMetrics(
        oof_auc=oof_auc,
        mean_fold_auc=cv_result["mean_val_auc"],
        std_fold_auc=cv_result["std_val_auc"],
        eval_metrics=eval_metrics,
    )


def validate_oof_coverage(
    n_samples: int,
    cv_result: CVResult,
) -> bool:
    """Validate that OOF predictions cover all samples exactly once.

    In proper k-fold CV, each sample should appear in exactly one
    validation fold, so OOF predictions should cover all samples.

    Args:
        n_samples: Expected total number of samples.
        cv_result: Cross-validation result to validate.

    Returns:
        True if coverage is valid, False otherwise.
    """
    if len(cv_result["oof_predictions"]) != n_samples:
        return False

    # Check each sample appears in exactly one validation fold
    covered: NDArray[np.bool_] = np.zeros(n_samples, dtype=np.bool_)

    for fold_result in cv_result["fold_results"]:
        val_indices = fold_result["val_indices"]
        for i in range(len(val_indices)):
            idx_int = int(val_indices.item(i))
            already_covered = bool(covered.item(idx_int))
            if already_covered:
                # Sample appears in multiple validation folds
                return False
            covered[idx_int] = True

    # All samples should be covered
    all_result: np.bool_ = np.all(covered)
    return bool(all_result)


def get_oof_for_stacking(
    cv_result: CVResult,
) -> NDArray[np.float64]:
    """Extract OOF predictions ready for use as stacking features.

    OOF predictions can be used as meta-features for stacking ensembles.
    Each prediction was made by a model that never saw that sample,
    making them suitable as unbiased input features.

    Args:
        cv_result: Cross-validation result containing OOF predictions.

    Returns:
        OOF predictions of shape (n_samples,) suitable for stacking.
    """
    return cv_result["oof_predictions"]


def combine_oof_predictions(
    oof_arrays: tuple[NDArray[np.float64], ...],
) -> NDArray[np.float64]:
    """Combine OOF predictions from multiple models for stacking.

    Stacks predictions from multiple models as columns, creating
    a feature matrix for meta-learner training.

    Args:
        oof_arrays: Tuple of OOF prediction arrays, each of shape (n_samples,).

    Returns:
        Combined array of shape (n_samples, n_models).

    Raises:
        ValueError: If arrays have different lengths.
    """
    if len(oof_arrays) == 0:
        raise ValueError("At least one OOF array required")

    n_samples = len(oof_arrays[0])

    for i, arr in enumerate(oof_arrays):
        if len(arr) != n_samples:
            raise ValueError(f"OOF array {i} has length {len(arr)}, expected {n_samples}")

    # Stack as columns
    result: NDArray[np.float64] = np.column_stack(oof_arrays)
    return result


__all__ = [
    "OOFMetrics",
    "combine_oof_predictions",
    "compute_oof_auc",
    "compute_oof_metrics",
    "get_oof_for_stacking",
    "validate_oof_coverage",
]
