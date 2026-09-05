"""Shared fixtures and helpers for test_pipeline splits."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    ClassifierBackend,
    PreparedClassifier,
)
from covenant_ml.optimizer import SearchSpace
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    TrainOutcome,
    TrainProgress,
)
from numpy.typing import NDArray


class Fake2DClassifier:
    """Fake classifier that returns 2D predictions like sklearn."""

    def __init__(self, rng_seed: int = 42) -> None:
        """Initialize with random generator.

        Args:
            rng_seed: Random seed.
        """
        self._rng = np.random.default_rng(rng_seed)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return 2D probabilities like sklearn classifiers.

        Args:
            x: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with class probabilities.
        """
        n_samples = x.shape[0]
        # Build 2D array without column_stack to avoid Any type issues
        result: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        probs_1 = self._rng.uniform(0.0, 1.0, size=n_samples)
        result[:, 1] = probs_1
        result[:, 0] = 1.0 - probs_1
        return result


class Fake2DBackend:
    """Fake backend returning 2D-prediction classifier."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize with output directory.

        Args:
            output_dir: Output directory path.
        """
        self._output_dir = output_dir

    def backend_name(self) -> BackendName:
        """Get backend name."""
        return "lightgbm"

    def capabilities(self) -> BackendCapabilities:
        """Get capabilities."""
        return BackendCapabilities(
            supports_train=True,
            supports_gpu=False,
            supports_early_stopping=False,
            supports_feature_importance=False,
            model_format="pkl",
        )

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> Fake2DClassifier:
        """Prepare classifier.

        Args:
            n_features: Number of features.
            n_classes: Number of classes.
            feature_names: Feature names.

        Returns:
            Fake 2D classifier.
        """
        _ = n_features, n_classes, feature_names
        return Fake2DClassifier()

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
        """Train classifier.

        Args:
            x_features: Features.
            y_labels: Labels.
            feature_names: Feature names.
            config: Config.
            output_dir: Output directory.
            progress: Progress callback.

        Returns:
            Train outcome.
        """
        _ = progress, y_labels, feature_names

        model_path = output_dir / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("fake 2d model")

        n_samples = int(x_features.shape[0])
        n_train = int(n_samples * 0.7)
        n_val = int(n_samples * 0.15)
        n_test = n_samples - n_train - n_val

        fake_metrics = EvalMetrics(
            loss=0.3,
            ppl=1.35,
            auc=0.85,
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1_score=0.72,
        )

        return TrainOutcome(
            model_path=str(model_path),
            model_id="fake_2d_model",
            samples_total=n_samples,
            samples_train=n_train,
            samples_val=n_val,
            samples_test=n_test,
            train_metrics=fake_metrics,
            val_metrics=fake_metrics,
            test_metrics=fake_metrics,
            best_val_auc=0.85,
            best_round=10,
            total_rounds=10,
            early_stopped=False,
            config=config,
            feature_importances=[],
            scale_pos_weight_computed=1.0,
        )

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        """Evaluate model.

        Args:
            model: Model.
            x: Features.
            y: Labels.

        Returns:
            Eval metrics.
        """
        _ = model, x, y
        return EvalMetrics(
            loss=0.3,
            ppl=1.35,
            auc=0.85,
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1_score=0.72,
        )

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        """Save model.

        Args:
            model: Model.
            path: Path.
        """
        _ = model, path

    def load(self, *, path: str) -> Fake2DClassifier:
        """Load model.

        Args:
            path: Path.

        Returns:
            Fake 2D classifier.
        """
        _ = path
        return Fake2DClassifier()

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Get feature importances.

        Args:
            model: Model.
            feature_names: Feature names.

        Returns:
            Empty list.
        """
        _ = model, feature_names
        return []

    def get_default_search_space(self) -> SearchSpace:
        """Not used in pipeline tests."""
        raise NotImplementedError

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Not used in pipeline tests."""
        raise NotImplementedError


class Fake2DRegistry:
    """Registry returning 2D-prediction backend."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize with output directory.

        Args:
            output_dir: Output directory.
        """
        self._output_dir = output_dir

    def get(self, name: BackendName) -> ClassifierBackend:
        """Get backend.

        Args:
            name: Backend name.

        Returns:
            Fake 2D backend.
        """
        _ = name
        backend: ClassifierBackend = Fake2DBackend(self._output_dir)
        return backend
