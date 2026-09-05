"""Fake implementations for amex pipeline tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    ClassifierBackend,
    PreparedClassifier,
)
from covenant_ml.datasets import LoadedDataset, TimeSeriesDatasetConfig
from covenant_ml.datasets.types import DatasetMeta
from covenant_ml.ensemble.types import (
    EnsembleOOFData,
    EnsembleWeights,
    OptimizationConfig,
    OptimizationResult,
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

import scripts.amex._hooks as hooks_module
from scripts.amex._hook_protocols import (
    FakeDatasetSpec,
)


class FakeConsole:
    """Fake console that captures output for testing."""

    def __init__(self) -> None:
        """Initialize fake console."""
        self._messages: list[str] = []

    def write(self, message: str) -> None:
        """Capture message.

        Args:
            message: Message to capture.
        """
        self._messages.append(message)

    @property
    def messages(self) -> tuple[str, ...]:
        """Get captured messages."""
        return tuple(self._messages)


def make_fake_dataset(
    n_samples: int,
    n_features: int,
    positive_ratio: float,
    name: str,
    random_state: int = 42,
) -> LoadedDataset:
    """Create a fake loaded dataset for testing.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        positive_ratio: Fraction of positive samples.
        name: Dataset name.
        random_state: Random seed.

    Returns:
        Fake LoadedDataset.
    """
    rng = np.random.default_rng(random_state)

    # Generate features
    x: NDArray[np.float64] = rng.standard_normal((n_samples, n_features)).astype(np.float64)

    # Generate labels
    n_positive = int(n_samples * positive_ratio)
    labels_list: list[int] = [1] * n_positive + [0] * (n_samples - n_positive)
    rng.shuffle(labels_list)
    y: NDArray[np.int64] = np.array(labels_list, dtype=np.int64)

    # Generate feature names
    feature_names: tuple[str, ...] = tuple(f"feature_{i}" for i in range(n_features))

    return LoadedDataset(
        groups=None,
        meta=DatasetMeta(
            name=name,
            n_samples=n_samples,
            n_features=n_features,
            n_positive=n_positive,
            n_negative=n_samples - n_positive,
            positive_ratio=positive_ratio,
            feature_names=feature_names,
            categorical_encodings=(),
        ),
        x=x,
        y=y,
    )


class FakeTimeseriesLoader:
    """Fake time-series loader for testing."""

    def __init__(
        self,
        train_spec: FakeDatasetSpec,
        test_spec: FakeDatasetSpec,
    ) -> None:
        """Initialize fake loader.

        Args:
            train_spec: Specification for training dataset.
            test_spec: Specification for test dataset.
        """
        self._train_spec = train_spec
        self._test_spec = test_spec

    def __call__(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load a fake dataset.

        Args:
            config: Dataset configuration.
            external_dir: Ignored in fake.

        Returns:
            Fake LoadedDataset.
        """
        # Determine if this is train or test based on config name
        spec = self._train_spec if "train" in config["name"] else self._test_spec

        return make_fake_dataset(
            n_samples=spec["n_samples"],
            n_features=spec["n_features"],
            positive_ratio=spec["positive_ratio"],
            name=config["name"],
        )


class FakePreparedClassifier:
    """Fake prepared classifier for testing."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialize fake classifier.

        Args:
            random_state: Random seed.
        """
        self._rng = np.random.default_rng(random_state)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return random probabilities.

        Args:
            x: Feature matrix.

        Returns:
            Random probabilities of shape (n_samples,).
        """
        n_samples = x.shape[0]
        # Generate random values - mypy needs explicit array conversion
        random_values = self._rng.uniform(0.0, 1.0, size=n_samples)
        random_probs: NDArray[np.float64] = np.asarray(random_values, dtype=np.float64)
        return random_probs


class FakeBackend:
    """Fake ML backend for testing.

    Implements full ClassifierBackend protocol with keyword-only arguments.
    """

    def __init__(self, output_dir: Path, random_state: int = 42) -> None:
        """Initialize fake backend.

        Args:
            output_dir: Directory to save fake model.
            random_state: Random seed.
        """
        self._output_dir = output_dir
        self._random_state = random_state

    def backend_name(self) -> BackendName:
        """Get backend name.

        Returns:
            Backend name as BackendName literal.
        """
        return "lightgbm"

    def capabilities(self) -> BackendCapabilities:
        """Get backend capabilities.

        Returns:
            BackendCapabilities TypedDict.
        """
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
    ) -> FakePreparedClassifier:
        """Prepare classifier.

        Args:
            n_features: Number of features.
            n_classes: Number of classes.
            feature_names: Feature names.

        Returns:
            Fake prepared classifier.
        """
        _ = n_features
        _ = n_classes
        _ = feature_names
        return FakePreparedClassifier(self._random_state)

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
        """Fake training.

        Args:
            x_features: Features.
            y_labels: Labels.
            feature_names: Feature names.
            config: Config.
            output_dir: Output directory.
            progress: Progress callback (ignored).

        Returns:
            TrainOutcome TypedDict.
        """
        _ = progress
        _ = y_labels

        # Create a fake model file
        model_path = output_dir / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("fake model")

        n_samples: int = int(x_features.shape[0])
        n_train: int = int(n_samples * 0.7)
        n_val: int = int(n_samples * 0.15)
        n_test: int = n_samples - n_train - n_val

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
            model_id="fake_model_001",
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
            feature_importances=_make_fake_importances(feature_names),
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
            EvalMetrics TypedDict.
        """
        _ = model
        _ = x
        _ = y
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
        _ = model
        _ = path

    def load(self, *, path: str) -> FakePreparedClassifier:
        """Load fake model.

        Args:
            path: Model path (ignored).

        Returns:
            Fake prepared classifier.
        """
        _ = path
        return FakePreparedClassifier(self._random_state)

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
            List of feature importances or None.
        """
        _ = model
        return _make_fake_importances(feature_names)

    def get_default_search_space(self) -> SearchSpace:
        """Not used in amex pipeline."""
        raise NotImplementedError

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Not used in amex pipeline."""
        raise NotImplementedError


def _make_fake_importances(
    feature_names: list[str] | None,
) -> list[FeatureImportance]:
    """Create fake feature importances.

    Args:
        feature_names: Feature names or None.

    Returns:
        List of fake FeatureImportance entries.
    """
    if feature_names is None:
        return []
    result: list[FeatureImportance] = []
    for i, name in enumerate(feature_names):
        result.append(
            FeatureImportance(
                name=name,
                importance=1.0 / (i + 1),
                rank=i + 1,
            )
        )
    return result


class FakeRegistry:
    """Fake classifier registry for testing."""

    def __init__(self, output_dir: Path, random_state: int = 42) -> None:
        """Initialize fake registry.

        Args:
            output_dir: Directory for model outputs.
            random_state: Random seed.
        """
        self._output_dir = output_dir
        self._random_state = random_state

    def get(self, name: BackendName) -> ClassifierBackend:
        """Get fake backend.

        Args:
            name: Backend name (ignored).

        Returns:
            Fake backend that implements ClassifierBackend protocol.
        """
        _ = name
        backend: ClassifierBackend = FakeBackend(self._output_dir, self._random_state)
        return backend


def make_fake_optimizer(random_state: int = 42) -> hooks_module.EnsembleOptimizerCallable:
    """Create a fake ensemble optimizer.

    Args:
        random_state: Random seed.

    Returns:
        Fake optimizer callable.
    """

    def fake_optimizer(
        oof_data: EnsembleOOFData,
        config: OptimizationConfig,
    ) -> OptimizationResult:
        """Fake optimizer that returns equal weights.

        Args:
            oof_data: OOF data.
            config: Config.

        Returns:
            Fake optimization result with equal weights.
        """
        n_models = oof_data["n_models"]
        model_names = tuple(p["model_name"] for p in oof_data["model_predictions"])

        weights: NDArray[np.float64] = np.full(n_models, 1.0 / n_models, dtype=np.float64)

        return OptimizationResult(
            weights=EnsembleWeights(
                weights=weights,
                model_names=model_names,
            ),
            best_score=0.82,
            n_iterations=10,
            converged=True,
            initial_score=0.80,
        )

    return fake_optimizer


__all__ = [
    "FakeBackend",
    "FakeConsole",
    "FakePreparedClassifier",
    "FakeRegistry",
    "FakeTimeseriesLoader",
    "make_fake_dataset",
    "make_fake_optimizer",
]
