"""Cross-validation runner with preprocessing isolation.

Runs k-fold cross-validation ensuring preprocessing is fit only on
training data within each fold to prevent data leakage.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.metrics import compute_auc
from covenant_ml.preprocessing import AutoPreprocessor, PreprocessingState
from covenant_ml.validation.splitter import (
    get_fold_data,
    group_stratified_kfold_split,
    stratified_kfold_split,
)
from covenant_ml.validation.types import CVResult, CVSplitInfo, FoldResult

_log = get_logger(__name__)


class TrainedModel(Protocol):
    """Protocol for a trained model that can predict probabilities."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            x: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability predictions of shape (n_samples,) for binary class 1.
        """
        ...


class FoldTrainer(Protocol):
    """Protocol for training a model on a single fold.

    Implementations should train a model and return it for prediction.
    The trainer is called with already preprocessed data.
    """

    def __call__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        fold_number: int,
    ) -> TrainedModel:
        """Train a model on one fold.

        Args:
            x_train: Preprocessed training features.
            y_train: Training labels.
            x_val: Preprocessed validation features.
            y_val: Validation labels.
            fold_number: Zero-indexed fold number.

        Returns:
            Trained model implementing predict_proba.
        """
        ...


def _compute_std(values: tuple[float, ...]) -> float:
    """Compute standard deviation of values.

    Args:
        values: Tuple of float values.

    Returns:
        Standard deviation (population std, not sample std).
    """
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0

    # Compute mean
    total = 0.0
    for v in values:
        total += v
    mean = total / n

    # Compute variance
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


def run_cross_validation(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    n_folds: int,
    random_state: int,
    trainer: FoldTrainer,
    progress_callback: Callable[[int, int], None] | None = None,
) -> CVResult:
    """Run stratified k-fold cross-validation with preprocessing isolation.

    For each fold:
    1. Split data into train/val using stratified sampling
    2. Fit preprocessing on training data only
    3. Transform both train and val with fitted preprocessing
    4. Train model on preprocessed training data
    5. Predict on validation data
    6. Collect metrics and OOF predictions

    Args:
        x: Feature matrix of shape (n_samples, n_features).
        y: Binary labels of shape (n_samples,).
        n_folds: Number of folds (must be >= 2).
        random_state: Random seed for reproducibility.
        trainer: Function to train model on one fold.
        progress_callback: Optional callback(fold_number, n_folds) for progress.

    Returns:
        CVResult with per-fold results, aggregated metrics, and OOF predictions.

    Raises:
        ValueError: If n_folds < 2 or insufficient samples per class.
    """
    n_samples = len(y)

    # Create stratified splits
    split_info: CVSplitInfo = stratified_kfold_split(y, n_folds, random_state)

    # Initialize OOF predictions array
    oof_predictions: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)

    # Process each fold
    fold_results: list[FoldResult] = []

    for fold_num, split in enumerate(split_info["folds"]):
        if progress_callback is not None:
            progress_callback(fold_num, n_folds)

        # Get raw data for this fold
        x_train_raw, y_train, x_val_raw, y_val = get_fold_data(x, y, split)

        # Fit preprocessing on training data only
        preprocessor = AutoPreprocessor()
        state: PreprocessingState = preprocessor.fit(x_train_raw, y_train)

        # Transform both train and val with fitted preprocessing
        x_train: NDArray[np.float64] = preprocessor.transform(x_train_raw, state)
        x_val: NDArray[np.float64] = preprocessor.transform(x_val_raw, state)

        # Train model on this fold
        model: TrainedModel = trainer(x_train, y_train, x_val, y_val, fold_num)

        # Get predictions
        train_proba: NDArray[np.float64] = model.predict_proba(x_train)
        val_proba: NDArray[np.float64] = model.predict_proba(x_val)

        # Compute metrics
        train_auc = compute_auc(y_train, train_proba)
        val_auc = compute_auc(y_val, val_proba)

        # Store OOF predictions at correct indices
        val_indices = split["val_indices"]
        for i in range(len(val_indices)):
            idx = int(val_indices.item(i))
            oof_predictions[idx] = float(val_proba.item(i))

        # Record fold result
        fold_results.append(
            FoldResult(
                fold_number=fold_num,
                train_auc=train_auc,
                val_auc=val_auc,
                val_indices=val_indices,
                val_predictions=val_proba,
            )
        )

        _log.info(
            "Completed fold",
            extra={
                "fold_number": fold_num,
                "n_folds": n_folds,
                "train_auc": train_auc,
                "val_auc": val_auc,
                "n_train": len(y_train),
                "n_val": len(y_val),
            },
        )

    # Aggregate metrics
    val_aucs: tuple[float, ...] = tuple(fr["val_auc"] for fr in fold_results)
    mean_val_auc = _compute_mean(val_aucs)
    std_val_auc = _compute_std(val_aucs)

    _log.info(
        "Cross-validation complete",
        extra={
            "n_folds": n_folds,
            "mean_val_auc": mean_val_auc,
            "std_val_auc": std_val_auc,
            "fold_aucs": val_aucs,
        },
    )

    return CVResult(
        n_folds=n_folds,
        fold_results=tuple(fold_results),
        mean_val_auc=mean_val_auc,
        std_val_auc=std_val_auc,
        oof_predictions=oof_predictions,
    )


def run_group_cross_validation(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    groups: NDArray[np.int64],
    n_folds: int,
    random_state: int,
    trainer: FoldTrainer,
    progress_callback: Callable[[int, int], None] | None = None,
) -> CVResult:
    """Run group-stratified k-fold cross-validation with preprocessing isolation.

    Similar to run_cross_validation but ensures that all samples from the same
    group (e.g., customer_ID) stay together in the same fold. This is critical
    for time-series data where multiple observations per entity must not leak
    between train and validation sets.

    For each fold:
    1. Split data by groups using stratified sampling (groups, not samples)
    2. Fit preprocessing on training data only
    3. Transform both train and val with fitted preprocessing
    4. Train model on preprocessed training data
    5. Predict on validation data
    6. Collect metrics and OOF predictions

    Args:
        x: Feature matrix of shape (n_samples, n_features).
        y: Binary labels of shape (n_samples,).
        groups: Group IDs of shape (n_samples,). All samples with the same
            group ID will be assigned to the same fold.
        n_folds: Number of folds (must be >= 2).
        random_state: Random seed for reproducibility.
        trainer: Function to train model on one fold.
        progress_callback: Optional callback(fold_number, n_folds) for progress.

    Returns:
        CVResult with per-fold results, aggregated metrics, and OOF predictions.

    Raises:
        ValueError: If n_folds < 2, insufficient groups per class, or length mismatch.
    """
    n_samples = len(y)

    # Create group-stratified splits
    split_info: CVSplitInfo = group_stratified_kfold_split(y, groups, n_folds, random_state)

    # Initialize OOF predictions array
    oof_predictions: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)

    # Process each fold
    fold_results: list[FoldResult] = []

    for fold_num, split in enumerate(split_info["folds"]):
        if progress_callback is not None:
            progress_callback(fold_num, n_folds)

        # Get raw data for this fold
        x_train_raw, y_train, x_val_raw, y_val = get_fold_data(x, y, split)

        # Fit preprocessing on training data only
        preprocessor = AutoPreprocessor()
        state: PreprocessingState = preprocessor.fit(x_train_raw, y_train)

        # Transform both train and val with fitted preprocessing
        x_train: NDArray[np.float64] = preprocessor.transform(x_train_raw, state)
        x_val: NDArray[np.float64] = preprocessor.transform(x_val_raw, state)

        # Train model on this fold
        model: TrainedModel = trainer(x_train, y_train, x_val, y_val, fold_num)

        # Get predictions
        train_proba: NDArray[np.float64] = model.predict_proba(x_train)
        val_proba: NDArray[np.float64] = model.predict_proba(x_val)

        # Compute metrics
        train_auc = compute_auc(y_train, train_proba)
        val_auc = compute_auc(y_val, val_proba)

        # Store OOF predictions at correct indices
        val_indices = split["val_indices"]
        for i in range(len(val_indices)):
            idx = int(val_indices.item(i))
            oof_predictions[idx] = float(val_proba.item(i))

        # Record fold result
        fold_results.append(
            FoldResult(
                fold_number=fold_num,
                train_auc=train_auc,
                val_auc=val_auc,
                val_indices=val_indices,
                val_predictions=val_proba,
            )
        )

        _log.info(
            "Completed group fold",
            extra={
                "fold_number": fold_num,
                "n_folds": n_folds,
                "train_auc": train_auc,
                "val_auc": val_auc,
                "n_train": len(y_train),
                "n_val": len(y_val),
            },
        )

    # Aggregate metrics
    val_aucs: tuple[float, ...] = tuple(fr["val_auc"] for fr in fold_results)
    mean_val_auc = _compute_mean(val_aucs)
    std_val_auc = _compute_std(val_aucs)

    _log.info(
        "Group cross-validation complete",
        extra={
            "n_folds": n_folds,
            "mean_val_auc": mean_val_auc,
            "std_val_auc": std_val_auc,
            "fold_aucs": val_aucs,
        },
    )

    return CVResult(
        n_folds=n_folds,
        fold_results=tuple(fold_results),
        mean_val_auc=mean_val_auc,
        std_val_auc=std_val_auc,
        oof_predictions=oof_predictions,
    )


__all__ = [
    "FoldTrainer",
    "TrainedModel",
    "run_cross_validation",
    "run_group_cross_validation",
]
