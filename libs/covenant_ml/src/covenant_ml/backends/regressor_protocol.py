"""Protocols and capabilities for pluggable tabular regressors.

Parallel to protocol.py for classifiers. Reuses BackendCapabilities.
Strict typing only: no Any, no dataclasses, no stubs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from covenant_ml.types import FeatureImportance
from covenant_ml.types_regression import (
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressionTrainProgress,
    RegressorBackendName,
    RegressorTrainConfig,
)

from ..optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SearchSpace,
)
from .protocol import BackendCapabilities

RegressorProgressCallback = Callable[[RegressionTrainProgress], None]


class PreparedRegressor(Protocol):
    """A trained regressor ready for inference.

    Returns 1D continuous predictions, not class probabilities.
    """

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict continuous values.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted values, shape (n_samples,).
        """
        ...


class RegressorBackend(Protocol):
    """Protocol for pluggable regressor backends.

    Parallel to ClassifierBackend. Key differences:
    - predict() returns 1D float64 instead of predict_proba() returning 2D
    - y_targets is float64 instead of y_labels int64
    - evaluate() returns RegressionMetrics instead of EvalMetrics
    - train() returns RegressionTrainOutcome instead of TrainOutcome
    - No n_classes in prepare() (regression is always 1 output)
    """

    def backend_name(self) -> RegressorBackendName:
        """Return the backend identifier.

        Returns:
            The backend name literal.
        """
        ...

    def capabilities(self) -> BackendCapabilities:
        """Return capability flags for this backend.

        Returns:
            BackendCapabilities describing what this backend supports.
        """
        ...

    def prepare(
        self,
        *,
        n_features: int,
        feature_names: list[str] | None,
    ) -> PreparedRegressor:
        """Prepare a regressor for inference from saved state.

        Args:
            n_features: Number of input features.
            feature_names: Optional feature names for the model.

        Returns:
            A prepared regressor ready for predict().
        """
        ...

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str] | None,
        config: RegressorTrainConfig,
        output_dir: Path,
        progress: RegressorProgressCallback | None,
    ) -> RegressionTrainOutcome:
        """Train a regression model.

        Args:
            x_features: Feature matrix, shape (n_samples, n_features).
            y_targets: Continuous target values, shape (n_samples,).
            feature_names: Optional feature names.
            config: Regressor training configuration.
            output_dir: Directory to save model artifacts.
            progress: Optional callback for training progress.

        Returns:
            Complete training outcome with metrics from all splits.
        """
        ...

    def evaluate(
        self,
        *,
        model: PreparedRegressor,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> RegressionMetrics:
        """Evaluate a trained regressor on data.

        Args:
            model: A trained regressor.
            x: Feature matrix, shape (n_samples, n_features).
            y: True continuous target values, shape (n_samples,).

        Returns:
            RegressionMetrics with mse, rmse, mae, r_squared, mape.
        """
        ...

    def save(self, *, model: PreparedRegressor, path: str) -> None:
        """Save a trained regressor to disk.

        Args:
            model: The regressor to save.
            path: File path to save to.
        """
        ...

    def load(self, *, path: str) -> PreparedRegressor:
        """Load a trained regressor from disk.

        Args:
            path: File path to load from.

        Returns:
            A prepared regressor ready for predict().
        """
        ...

    def get_feature_importances(
        self,
        *,
        model: PreparedRegressor,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Extract feature importances from a trained regressor.

        Args:
            model: A trained regressor.
            feature_names: Optional feature names for labeling.

        Returns:
            Sorted list of feature importances, or None if unsupported.
        """
        ...

    def get_default_search_space(self) -> SearchSpace:
        """Return the backend's default hyperparameter search space.

        Returns:
            Backend-specific SearchSpace with sensible default ranges.
        """
        ...

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Return a narrowed search space for fine-tuning around prior best params.

        Args:
            best_int_params: Best integer params from prior optimization.
            best_float_params: Best float params from prior optimization.

        Returns:
            Backend-specific SearchSpace with narrowed ranges.
        """
        ...


__all__ = [
    "PreparedRegressor",
    "RegressorBackend",
    "RegressorProgressCallback",
]
