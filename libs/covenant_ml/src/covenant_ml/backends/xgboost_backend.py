"""XGBoost backend wrapping the existing trainer API.

Provides a ClassifierBackend implementation that defers to
train_model_with_validation and preserves existing behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeGuard

import numpy as np
from numpy.typing import NDArray

from ..metrics import compute_all_metrics
from ..trainer import train_model_with_validation
from ..types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    TrainConfig,
    TrainOutcome,
)
from .protocol import BackendCapabilities, ClassifierBackend, PreparedClassifier, ProgressCallback


class _XGBPrepared:
    """Prepared placeholder; real training returns outcomes with persisted model."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise RuntimeError("XGBoost backend prepared model not available in this context")


XGBOOST_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": True,
    "supports_early_stopping": True,
    "supports_feature_importance": True,
    "model_format": "ubj",
}


class XGBoostBackend(ClassifierBackend):
    """Backend that wraps covenant_ml.trainer XGBoost implementation."""

    def backend_name(self) -> BackendName:
        return "xgboost"

    def capabilities(self) -> BackendCapabilities:
        return XGBOOST_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        # XGBoost uses on-demand training; preparation returns a placeholder.
        return _XGBPrepared()

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: ProgressCallback | None,
    ) -> TrainOutcome:
        # Delegate to existing trainer (expects TrainConfig)
        def _is_train_config(cfg: ClassifierTrainConfig) -> TypeGuard[TrainConfig]:
            return isinstance(cfg, dict) and "n_estimators" in cfg

        if not _is_train_config(config):
            raise RuntimeError("XGBoostBackend requires TrainConfig (found MLPConfig)")
        cfg = config
        # feature_names required by trainer for importances
        if feature_names is None:
            count = int(x_features.shape[1])
            names = [f"f{i}" for i in range(count)]
        else:
            names = feature_names
        return train_model_with_validation(
            x_features=x_features,
            y_labels=y_labels,
            config=cfg,
            output_dir=output_dir,
            feature_names=names,
            progress_callback=progress,
        )

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        proba = model.predict_proba(x)
        # positive class probability is column 1
        pos = proba[:, 1]
        return compute_all_metrics(y, pos)

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        # Intentionally no-op here; saving handled in train via trainer.save_model
        # Consumers use TrainOutcome.model_path.
        with open(path, "wb") as f:
            f.write(b"")

    def load(self, *, path: str) -> PreparedClassifier:
        # Not supported in this backend wrapper; prediction uses different pipeline.
        return _XGBPrepared()

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        # Importances are provided by the higher-level outcome in the trainer path.
        return None


def create_xgboost_backend() -> ClassifierBackend:
    return XGBoostBackend()


__all__ = [
    "XGBOOST_CAPABILITIES",
    "XGBoostBackend",
    "create_xgboost_backend",
]
