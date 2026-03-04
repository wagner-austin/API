"""Cross-validation types for regression model evaluation.

Parallel to types.py (classification). Key differences:
- RegressionFoldResult stores RMSE (not AUC)
- RegressionCVResult aggregates RMSE across folds
- val_predictions are continuous values (not probabilities)
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class RegressionFoldResult(TypedDict, total=True):
    """Result from a single regression cross-validation fold.

    Contains metrics and predictions for one fold of k-fold CV.

    Attributes:
        fold_number: Zero-indexed fold number (0 to n_folds-1).
        train_rmse: RMSE on training portion of this fold.
        val_rmse: RMSE on validation portion of this fold.
        val_indices: Indices of validation samples in original dataset.
        val_predictions: Predicted continuous values for validation samples.
    """

    fold_number: int
    train_rmse: float
    val_rmse: float
    val_indices: NDArray[np.intp]
    val_predictions: NDArray[np.float64]


class RegressionCVResult(TypedDict, total=True):
    """Complete regression cross-validation result across all folds.

    Contains aggregated RMSE metrics and out-of-fold predictions.

    Attributes:
        n_folds: Number of folds used in cross-validation.
        fold_results: Tuple of per-fold results.
        mean_val_rmse: Mean validation RMSE across all folds.
        std_val_rmse: Standard deviation of validation RMSE across folds.
        oof_predictions: Out-of-fold predictions for all samples.
            Shape (n_samples,) where each sample has prediction from
            the fold where it was in the validation set.
    """

    n_folds: int
    fold_results: tuple[RegressionFoldResult, ...]
    mean_val_rmse: float
    std_val_rmse: float
    oof_predictions: NDArray[np.float64]


__all__ = [
    "RegressionCVResult",
    "RegressionFoldResult",
]
