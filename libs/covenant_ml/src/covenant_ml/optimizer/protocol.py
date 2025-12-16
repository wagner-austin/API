"""Protocols for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
Defines the optimizer interface that implementations must satisfy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .types import OptimizationConfig, OptimizationSummary, TrialResult, XGBoostSearchSpace


class TrialCallbackProtocol(Protocol):
    """Protocol for trial completion callbacks."""

    def __call__(self, result: TrialResult) -> None:
        """Called after each trial completes.

        Args:
            result: The trial result with parameters and objective value.
        """
        ...


class XGBoostObjectiveProtocol(Protocol):
    """Protocol for XGBoost objective functions.

    The optimizer calls this function with sampled hyperparameters.
    The function trains a model and returns the validation AUC to maximize.
    """

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        max_depth: int,
        n_estimators: int,
        learning_rate: float,
        reg_alpha: float,
        reg_lambda: float,
        subsample: float,
        colsample_bytree: float,
        random_state: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> float:
        """Train XGBoost with given hyperparameters and return validation AUC.

        Args:
            x_features: Feature matrix (n_samples, n_features)
            y_labels: Binary labels (n_samples,)
            feature_names: Names for each feature column
            max_depth: Maximum tree depth
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate (eta)
            reg_alpha: L1 regularization weight
            reg_lambda: L2 regularization weight
            subsample: Row subsampling ratio
            colsample_bytree: Column subsampling ratio
            random_state: Random seed for reproducibility
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for test (unused during optimization)

        Returns:
            Validation AUC score to maximize
        """
        ...


class XGBoostOptimizerProtocol(Protocol):
    """Protocol for XGBoost hyperparameter optimizers.

    Implementations must provide a method to run optimization given
    data, search space, and configuration.
    """

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: XGBoostSearchSpace,
        config: OptimizationConfig,
        objective: XGBoostObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization.

        Args:
            x_features: Feature matrix (n_samples, n_features)
            y_labels: Binary labels (n_samples,)
            feature_names: Names for each feature column
            search_space: Parameter ranges to search
            config: Optimization settings (n_trials, timeout, etc.)
            objective: Function to evaluate hyperparameters
            trial_callback: Optional callback after each trial

        Returns:
            Summary with best hyperparameters and trial statistics
        """
        ...


# Type alias for the objective function callable signature
XGBoostObjectiveCallable = Callable[
    [
        NDArray[np.float64],  # x_features
        NDArray[np.int64],  # y_labels
        list[str],  # feature_names
        int,  # max_depth
        int,  # n_estimators
        float,  # learning_rate
        float,  # reg_alpha
        float,  # reg_lambda
        float,  # subsample
        float,  # colsample_bytree
        int,  # random_state
        float,  # train_ratio
        float,  # val_ratio
        float,  # test_ratio
    ],
    float,  # returns validation AUC
]


__all__ = [
    "TrialCallbackProtocol",
    "XGBoostObjectiveCallable",
    "XGBoostObjectiveProtocol",
    "XGBoostOptimizerProtocol",
]
