"""Protocols and capabilities for pluggable tabular classifiers.

Strict typing only: no Any, no dataclasses, no stubs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from ..optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SearchSpace,
)
from ..types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    TrainOutcome,
    TrainProgress,
)


class BackendCapabilities(TypedDict, total=True):
    """Describes supported features of a backend implementation."""

    supports_train: bool
    supports_gpu: bool
    supports_early_stopping: bool
    supports_feature_importance: bool
    model_format: str  # e.g., "ubj" for XGBoost booster, "pt" for torch


ProgressCallback = Callable[[TrainProgress], None]


class PreparedClassifier(Protocol):
    """A trained-or-prepared classifier ready for inference or further training."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


class ClassifierBackend(Protocol):
    """Protocol for pluggable classifier backends (e.g., XGBoost, MLP)."""

    def backend_name(self) -> BackendName: ...

    def capabilities(self) -> BackendCapabilities: ...

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier: ...

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: ProgressCallback | None,
        groups: NDArray[np.int64] | None = None,
    ) -> TrainOutcome: ...

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics: ...

    def save(self, *, model: PreparedClassifier, path: str) -> None: ...

    def load(self, *, path: str) -> PreparedClassifier: ...

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None: ...

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
    "BackendCapabilities",
    "ClassifierBackend",
    "PreparedClassifier",
    "ProgressCallback",
]
