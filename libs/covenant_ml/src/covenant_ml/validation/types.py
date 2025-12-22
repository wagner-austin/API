"""Cross-validation types for model evaluation.

Provides TypedDicts for cross-validation results, fold results, and OOF predictions.
All types are immutable (total=True) and strictly typed.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class FoldResult(TypedDict, total=True):
    """Result from a single cross-validation fold.

    Contains metrics and predictions for one fold of k-fold CV.

    Attributes:
        fold_number: Zero-indexed fold number (0 to n_folds-1).
        train_auc: AUC on training portion of this fold.
        val_auc: AUC on validation portion of this fold.
        val_indices: Indices of validation samples in original dataset.
        val_predictions: Predicted probabilities for validation samples.
    """

    fold_number: int
    train_auc: float
    val_auc: float
    val_indices: NDArray[np.intp]
    val_predictions: NDArray[np.float64]


class CVResult(TypedDict, total=True):
    """Complete cross-validation result across all folds.

    Contains aggregated metrics and out-of-fold predictions for stacking.

    Attributes:
        n_folds: Number of folds used in cross-validation.
        fold_results: Tuple of per-fold results.
        mean_val_auc: Mean validation AUC across all folds.
        std_val_auc: Standard deviation of validation AUC across folds.
        oof_predictions: Out-of-fold predictions for all samples.
            Shape (n_samples,) where each sample has prediction from
            the fold where it was in validation set.
    """

    n_folds: int
    fold_results: tuple[FoldResult, ...]
    mean_val_auc: float
    std_val_auc: float
    oof_predictions: NDArray[np.float64]


class CVSplit(TypedDict, total=True):
    """A single train/validation split from k-fold CV.

    Used internally by splitter to yield splits one at a time.

    Attributes:
        fold_number: Zero-indexed fold number.
        train_indices: Indices of training samples.
        val_indices: Indices of validation samples.
    """

    fold_number: int
    train_indices: NDArray[np.intp]
    val_indices: NDArray[np.intp]


class CVSplitInfo(TypedDict, total=True):
    """Information about a complete k-fold split setup.

    Contains all folds and metadata about the split configuration.

    Attributes:
        n_folds: Number of folds.
        n_samples: Total number of samples.
        folds: Tuple of all fold splits.
    """

    n_folds: int
    n_samples: int
    folds: tuple[CVSplit, ...]


__all__ = [
    "CVResult",
    "CVSplit",
    "CVSplitInfo",
    "FoldResult",
]
