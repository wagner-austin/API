"""XGBoost backend wrapping the existing trainer API.

Provides a ClassifierBackend implementation that defers to
train_model_with_validation and preserves existing behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeGuard

import numpy as np
from numpy.typing import NDArray

from covenant_ml.trainer_fit import train_model_with_validation

from ...metrics import compute_all_metrics
from ...optimizer.search_spaces import make_xgboost_default_space, make_xgboost_focused_space
from ...optimizer.types import SampledFloatParams, SampledIntParams, SearchSpace
from ...types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    TrainConfig,
    TrainOutcome,
    TrainProgress,
)
from ..protocol import BackendCapabilities, ClassifierBackend, PreparedClassifier


class _XGBClassifierProtocol(Protocol):
    """Protocol for a fitted XGBClassifier."""

    def load_model(self, fname: str) -> None: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


class _XGBClassifierCtor(Protocol):
    """Protocol for the XGBClassifier constructor."""

    def __call__(self) -> _XGBClassifierProtocol: ...


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
        """Prepare is not supported.

        XGBoost uses on-demand training via train(). Use train() to create
        a fitted model, then load() to get a PreparedClassifier for inference.

        Raises:
            RuntimeError: Always, as prepare is not supported.
        """
        raise RuntimeError(
            "XGBoostBackend.prepare not supported; train() then load() for inference."
        )

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: Callable[[TrainProgress], None] | None,
        groups: NDArray[np.int64] | None = None,
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
            groups=groups,
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
        """Save is not supported; the trainer persists the model.

        Args:
            model: Unused; this backend has no in-memory model to write.
            path: Unused.

        Raises:
            RuntimeError: Always. Use TrainOutcome.model_path.
        """
        raise RuntimeError("XGBoostBackend.save not supported; use TrainOutcome.model_path.")

    def load(self, *, path: str) -> PreparedClassifier:
        """Load a fitted XGBoost model from a persisted booster file.

        Args:
            path: Path to the saved model file (.ubj format).

        The fitted classifier is returned as-is rather than wrapped. It
            already satisfies PreparedClassifier, and SHAP's TreeExplainer
            introspects the native object -- it rejects a wrapper with
            "Model type not yet supported by TreeExplainer".

        Returns:
            The loaded classifier, which implements PreparedClassifier.
        """
        xgb_module = __import__("xgboost")
        classifier_ctor: _XGBClassifierCtor = xgb_module.XGBClassifier
        model = classifier_ctor()
        model.load_model(path)
        return model

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        # Importances are provided by the higher-level outcome in the trainer path.
        return None

    def get_default_search_space(self) -> SearchSpace:
        """Return default XGBoost search space with DART support.

        Returns:
            XGBoostSearchSpace with sensible default ranges.
        """
        return make_xgboost_default_space()

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Return focused XGBoost search space around prior best params.

        Args:
            best_int_params: Best integer params (reads max_depth).
            best_float_params: Best float params (reads learning_rate).

        Returns:
            XGBoostSearchSpace with narrowed ranges.
        """
        return make_xgboost_focused_space(
            best_max_depth=best_int_params["max_depth"],
            best_learning_rate=best_float_params["learning_rate"],
        )


def create_xgboost_backend() -> ClassifierBackend:
    return XGBoostBackend()


__all__ = [
    "XGBOOST_CAPABILITIES",
    "XGBoostBackend",
    "create_xgboost_backend",
]
