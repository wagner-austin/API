"""Regression cross-validation runner with preprocessing isolation.

Parallel to runner.py (classification). Key differences:
- Uses kfold_split (not stratified_kfold_split — regression has no classes)
- Metrics: RMSE (not AUC)
- y: float64 continuous targets (not int64 binary labels)
- predict() returns 1D continuous values (not predict_proba() class probabilities)
- FoldRegressorTrainer/TrainedRegressor protocols (not FoldTrainer/TrainedModel)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.metrics_regression import compute_rmse
from covenant_ml.preprocessing import AutoPreprocessor, PreprocessingState
from covenant_ml.validation.regression_types import RegressionCVResult, RegressionFoldResult
from covenant_ml.validation.types import CVSplit, CVSplitInfo

_log = get_logger(__name__)


class TrainedRegressor(Protocol):
    """Protocol for a trained regression model that can predict continuous values."""

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict continuous values.

        Args:
            x: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples,).
        """
        ...


class FoldRegressorTrainer(Protocol):
    """Protocol for training a regressor on a single fold.

    Implementations should train a model and return it for prediction.
    The trainer is called with already preprocessed data.
    """

    def __call__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        fold_number: int,
    ) -> TrainedRegressor:
        """Train a regressor on one fold.

        Args:
            x_train: Preprocessed training features.
            y_train: Training targets (float64 continuous).
            x_val: Preprocessed validation features.
            y_val: Validation targets (float64 continuous).
            fold_number: Zero-indexed fold number.

        Returns:
            Trained regressor implementing predict.
        """
        ...


# =============================================================================
# Statistics helpers (pure Python to avoid np.mean Any typing)
# =============================================================================


def _compute_std(values: tuple[float, ...]) -> float:
    """Compute population standard deviation of values.

    Args:
        values: Tuple of float values.

    Returns:
        Population standard deviation.
    """
    n = len(values)
    if n <= 1:
        return 0.0

    total = 0.0
    for v in values:
        total += v
    mean = total / n

    var_sum = 0.0
    for v in values:
        diff = v - mean
        var_sum += diff * diff

    return math.sqrt(var_sum / n)


def _compute_mean(values: tuple[float, ...]) -> float:
    """Compute mean of values.

    Args:
        values: Tuple of float values.

    Returns:
        Mean value, or 0.0 if empty.
    """
    n = len(values)
    if n == 0:
        return 0.0
    total = 0.0
    for v in values:
        total += v
    return total / n


# =============================================================================
# Splitting
# =============================================================================


def kfold_split(
    n_samples: int,
    n_folds: int,
    random_state: int,
) -> CVSplitInfo:
    """Create random k-fold cross-validation splits for regression.

    Unlike stratified_kfold_split, this does not attempt to balance classes
    across folds. It simply shuffles all indices and splits into n_folds
    approximately equal parts. Appropriate for regression where y is continuous.

    Args:
        n_samples: Total number of samples.
        n_folds: Number of folds (must be >= 2).
        random_state: Random seed for reproducibility.

    Returns:
        CVSplitInfo containing all fold splits and metadata.

    Raises:
        ValueError: If n_folds < 2 or n_samples < n_folds.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_samples < n_folds:
        raise ValueError(
            f"Not enough samples ({n_samples}) for {n_folds} folds. "
            f"Need at least {n_folds} samples."
        )

    rng = np.random.default_rng(random_state)
    all_indices: NDArray[np.intp] = np.arange(n_samples, dtype=np.intp)
    rng.shuffle(all_indices)

    # Split into approximately equal parts
    fold_arrays: list[NDArray[np.intp]] = []
    raw_splits = np.array_split(all_indices, n_folds)
    for split in raw_splits:
        typed: NDArray[np.intp] = np.asarray(split, dtype=np.intp)
        fold_arrays.append(typed)

    # Build CVSplits
    folds: list[CVSplit] = []
    for fold_num in range(n_folds):
        val_indices = fold_arrays[fold_num]

        # Training is all other folds concatenated
        train_parts: list[NDArray[np.intp]] = []
        for other in range(n_folds):
            if other != fold_num:
                train_parts.append(fold_arrays[other])
        train_indices: NDArray[np.intp] = np.concatenate(train_parts)

        # Shuffle for randomness
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        folds.append(
            CVSplit(
                fold_number=fold_num,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

    _log.info(
        "Created k-fold splits for regression",
        extra={
            "n_folds": n_folds,
            "n_samples": n_samples,
        },
    )

    return CVSplitInfo(
        n_folds=n_folds,
        n_samples=n_samples,
        folds=tuple(folds),
    )


def get_regression_fold_data(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    split: CVSplit,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Extract train and validation data for a single regression fold.

    Args:
        x: Feature matrix of shape (n_samples, n_features).
        y: Continuous targets of shape (n_samples,).
        split: A single fold split with train and val indices.

    Returns:
        Tuple of (x_train, y_train, x_val, y_val) with float64 targets.
    """
    x_train: NDArray[np.float64] = x[split["train_indices"]]
    y_train: NDArray[np.float64] = y[split["train_indices"]]
    x_val: NDArray[np.float64] = x[split["val_indices"]]
    y_val: NDArray[np.float64] = y[split["val_indices"]]

    return x_train, y_train, x_val, y_val


# =============================================================================
# Runner
# =============================================================================


def run_regression_cross_validation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_folds: int,
    random_state: int,
    trainer: FoldRegressorTrainer,
    progress_callback: Callable[[int, int], None] | None = None,
) -> RegressionCVResult:
    """Run k-fold cross-validation for regression with preprocessing isolation.

    For each fold:
    1. Split data into train/val using random (non-stratified) sampling
    2. Fit preprocessing on training data only
    3. Transform both train and val with fitted preprocessing
    4. Train model on preprocessed training data
    5. Predict on validation data
    6. Collect metrics and OOF predictions

    Args:
        x: Feature matrix of shape (n_samples, n_features).
        y: Continuous targets of shape (n_samples,).
        n_folds: Number of folds (must be >= 2).
        random_state: Random seed for reproducibility.
        trainer: Function to train regressor on one fold.
        progress_callback: Optional callback(fold_number, n_folds) for progress.

    Returns:
        RegressionCVResult with per-fold results, aggregated metrics, and
        OOF predictions.

    Raises:
        ValueError: If n_folds < 2 or insufficient samples.
    """
    n_samples = len(y)

    # Create random splits (not stratified — regression has no classes)
    split_info: CVSplitInfo = kfold_split(n_samples, n_folds, random_state)

    # Initialize OOF predictions array
    oof_predictions: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)

    # Process each fold
    fold_results: list[RegressionFoldResult] = []

    for fold_num, split in enumerate(split_info["folds"]):
        if progress_callback is not None:
            progress_callback(fold_num, n_folds)

        # Get raw data for this fold
        x_train_raw, y_train, x_val_raw, y_val = get_regression_fold_data(x, y, split)

        # Fit preprocessing on training data only
        preprocessor = AutoPreprocessor()
        dummy_y_int: NDArray[np.int64] = np.zeros(len(y_train), dtype=np.int64)
        state: PreprocessingState = preprocessor.fit(x_train_raw, dummy_y_int)

        # Transform both train and val with fitted preprocessing
        x_train: NDArray[np.float64] = preprocessor.transform(x_train_raw, state)
        x_val: NDArray[np.float64] = preprocessor.transform(x_val_raw, state)

        # Train model on this fold
        model: TrainedRegressor = trainer(x_train, y_train, x_val, y_val, fold_num)

        # Get predictions
        train_preds: NDArray[np.float64] = model.predict(x_train)
        val_preds: NDArray[np.float64] = model.predict(x_val)

        # Compute RMSE metrics
        train_rmse = compute_rmse(y_train, train_preds)
        val_rmse = compute_rmse(y_val, val_preds)

        # Store OOF predictions at correct indices
        val_indices = split["val_indices"]
        for i in range(len(val_indices)):
            idx = int(val_indices.item(i))
            oof_predictions[idx] = float(val_preds.item(i))

        # Record fold result
        fold_results.append(
            RegressionFoldResult(
                fold_number=fold_num,
                train_rmse=train_rmse,
                val_rmse=val_rmse,
                val_indices=val_indices,
                val_predictions=val_preds,
            )
        )

        _log.info(
            "Completed regression fold",
            extra={
                "fold_number": fold_num,
                "n_folds": n_folds,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "n_train": len(y_train),
                "n_val": len(y_val),
            },
        )

    # Aggregate metrics
    val_rmses: tuple[float, ...] = tuple(fr["val_rmse"] for fr in fold_results)
    mean_val_rmse = _compute_mean(val_rmses)
    std_val_rmse = _compute_std(val_rmses)

    _log.info(
        "Regression cross-validation complete",
        extra={
            "n_folds": n_folds,
            "mean_val_rmse": mean_val_rmse,
            "std_val_rmse": std_val_rmse,
            "fold_rmses": val_rmses,
        },
    )

    return RegressionCVResult(
        n_folds=n_folds,
        fold_results=tuple(fold_results),
        mean_val_rmse=mean_val_rmse,
        std_val_rmse=std_val_rmse,
        oof_predictions=oof_predictions,
    )


__all__ = [
    "FoldRegressorTrainer",
    "TrainedRegressor",
    "get_regression_fold_data",
    "kfold_split",
    "run_regression_cross_validation",
]
