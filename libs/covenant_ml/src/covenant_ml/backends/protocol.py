"""Protocols and capabilities for pluggable tabular classifiers.

Strict typing only: no Any, no dataclasses, no stubs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

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


__all__ = [
    "BackendCapabilities",
    "ClassifierBackend",
    "PreparedClassifier",
    "ProgressCallback",
]
