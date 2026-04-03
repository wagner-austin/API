"""Protocols for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
Defines the optimizer interface that implementations must satisfy.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .types import (
    ClearGBMSearchSpace,
    LightGBMSearchSpace,
    LogRegSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    RandomForestSearchSpace,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    TrialResult,
    XGBoostSearchSpace,
)

# =============================================================================
# Trial Callback Protocol
# =============================================================================


class TrialCallbackProtocol(Protocol):
    """Protocol for trial completion callbacks."""

    def __call__(self, result: TrialResult) -> None:
        """Called after each trial completes.

        Args:
            result: The trial result with parameters and objective value.
        """
        ...


# =============================================================================
# Objective Function Protocols
# =============================================================================


class ObjectiveProtocol(Protocol):
    """Protocol for objective functions.

    The optimizer calls this function with sampled hyperparameters.
    The function trains a model and returns the validation AUC to maximize.
    """

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        int_params: SampledIntParams,
        float_params: SampledFloatParams,
        string_params: SampledStringParams,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ) -> float:
        """Train model with given hyperparameters and return validation AUC.

        Args:
            x_features: Feature matrix (n_samples, n_features)
            y_labels: Binary labels (n_samples,)
            feature_names: Names for each feature column
            int_params: Integer hyperparameters
            float_params: Float hyperparameters
            string_params: String hyperparameters (boosting_type, booster)
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for test (unused during optimization)
            random_state: Random seed for reproducibility

        Returns:
            Validation AUC score to maximize
        """
        ...


# =============================================================================
# Optimizer Protocols
# =============================================================================


class XGBoostOptimizerProtocol(Protocol):
    """Protocol for XGBoost hyperparameter optimizers."""

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: XGBoostSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
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


class MLPOptimizerProtocol(Protocol):
    """Protocol for MLP hyperparameter optimizers."""

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: MLPSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization for MLP."""
        ...


class LSTMOptimizerProtocol(Protocol):
    """Protocol for LSTM hyperparameter optimizers."""

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: LSTMSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization for LSTM."""
        ...


class LightGBMOptimizerProtocol(Protocol):
    """Protocol for LightGBM hyperparameter optimizers."""

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: LightGBMSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization for LightGBM."""
        ...


class ClearGBMOptimizerProtocol(Protocol):
    """Protocol for ClearGBM hyperparameter optimizers."""

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: ClearGBMSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization for ClearGBM."""
        ...


class LogRegOptimizerProtocol(Protocol):
    """Protocol for Logistic Regression hyperparameter optimizers."""

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: LogRegSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization for Logistic Regression."""
        ...


class RandomForestOptimizerProtocol(Protocol):
    """Protocol for Random Forest hyperparameter optimizers."""

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: RandomForestSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization for Random Forest."""
        ...


__all__ = [
    "ClearGBMOptimizerProtocol",
    "LSTMOptimizerProtocol",
    "LightGBMOptimizerProtocol",
    "LogRegOptimizerProtocol",
    "MLPOptimizerProtocol",
    "ObjectiveProtocol",
    "RandomForestOptimizerProtocol",
    "TrialCallbackProtocol",
    "XGBoostOptimizerProtocol",
]
