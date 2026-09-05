"""Shared fixtures and helpers for test_model_saver splits."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    PreparedClassifier,
)
from covenant_ml.datasets import DatasetConfig, DatasetMeta, LoadedDataset
from covenant_ml.optimizer.types import (
    FloatRangeSpec,
    IntRangeSpec,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    SearchSpace,
    XGBoostSearchSpace,
)
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    TrainOutcome,
    TrainProgress,
)
from numpy.typing import NDArray
from scripts.optimize.model_saver import (
    MODEL_EXTENSIONS,
)

from covenant_radar_api.worker.optimize_result_types import (
    UnifiedOptimizationResult,
)


class FakePreparedClassifier:
    """Fake classifier for testing that returns constant predictions."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return constant probability predictions.

        Args:
            x: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, 2).
        """
        n_samples: int = int(x.shape[0])
        col_0: NDArray[np.float64] = np.full(n_samples, 0.3, dtype=np.float64)
        col_1: NDArray[np.float64] = np.full(n_samples, 0.7, dtype=np.float64)
        result: NDArray[np.float64] = np.column_stack([col_0, col_1])
        return result


class FakeClassifierBackend:
    """Fake backend for testing model saving workflow.

    Returns deterministic training outcomes without actual ML training.
    """

    def __init__(
        self,
        backend_name_val: BackendName = "xgboost",
        output_filename_override: str | None = None,
    ) -> None:
        """Initialize with configurable backend name and output filename.

        Args:
            backend_name_val: Backend name to return.
            output_filename_override: Optional override for output model filename.
                If set, uses this exact filename instead of default pattern.
        """
        self._backend_name = backend_name_val
        self._train_call_count = 0
        self._output_filename_override = output_filename_override

    def backend_name(self) -> BackendName:
        """Return the configured backend name.

        Returns:
            Backend name string.
        """
        return self._backend_name

    def capabilities(self) -> BackendCapabilities:
        """Return fake backend capabilities.

        Returns:
            BackendCapabilities with all features enabled.
        """
        return {
            "supports_train": True,
            "supports_gpu": False,
            "supports_early_stopping": True,
            "supports_feature_importance": True,
            "model_format": MODEL_EXTENSIONS[self._backend_name],
        }

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        """Prepare a fake classifier.

        Args:
            n_features: Number of input features.
            n_classes: Number of output classes.
            feature_names: Optional feature names.

        Returns:
            FakePreparedClassifier instance.
        """
        return FakePreparedClassifier()

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
        """Return a fake training outcome without actual training.

        Args:
            x_features: Feature matrix.
            y_labels: Label vector.
            feature_names: Optional feature names.
            config: Training configuration.
            output_dir: Output directory for model.
            progress: Optional progress callback.

        Returns:
            TrainOutcome with deterministic values.
        """
        self._train_call_count += 1
        ext = MODEL_EXTENSIONS[self._backend_name]

        if self._output_filename_override is not None:
            model_path = output_dir / self._output_filename_override
        else:
            model_path = output_dir / f"model_{self._train_call_count}.{ext}"

        model_path.write_bytes(b"fake model data")

        n_samples: int = int(x_features.shape[0])
        n_train: int = int(n_samples * 0.7)
        n_val: int = int(n_samples * 0.15)
        n_test: int = n_samples - n_train - n_val

        fake_metrics: EvalMetrics = {
            "loss": 0.35,
            "ppl": 1.42,
            "auc": 0.88,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
        }

        n_features_count: int = int(x_features.shape[1])
        fake_importances: list[FeatureImportance] = [
            {"name": f"feature_{i}", "importance": 1.0 / n_features_count, "rank": i + 1}
            for i in range(n_features_count)
        ]

        train_outcome: TrainOutcome = {
            "model_path": str(model_path),
            "model_id": f"fake-model-{self._train_call_count}",
            "samples_total": n_samples,
            "samples_train": n_train,
            "samples_val": n_val,
            "samples_test": n_test,
            "train_metrics": fake_metrics,
            "val_metrics": fake_metrics,
            "test_metrics": fake_metrics,
            "best_val_auc": 0.88,
            "best_round": 50,
            "total_rounds": 100,
            "early_stopped": True,
            "config": config,
            "feature_importances": fake_importances,
            "scale_pos_weight_computed": 1.0,
        }
        return train_outcome

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        """Return fake evaluation metrics.

        Args:
            model: Prepared classifier.
            x: Feature matrix.
            y: Label vector.

        Returns:
            EvalMetrics with deterministic values.
        """
        return {
            "loss": 0.35,
            "ppl": 1.42,
            "auc": 0.88,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
        }

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        """Save fake model to path.

        Args:
            model: Prepared classifier.
            path: Output path.
        """
        Path(path).write_bytes(b"fake model data")

    def load(self, *, path: str) -> PreparedClassifier:
        """Load fake model from path.

        Args:
            path: Model path.

        Returns:
            FakePreparedClassifier instance.
        """
        return FakePreparedClassifier()

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Return fake feature importances.

        Args:
            model: Prepared classifier.
            feature_names: Optional feature names.

        Returns:
            List of fake feature importances.
        """
        if feature_names is None:
            return None
        n_features = len(feature_names)
        return [
            {"name": name, "importance": 1.0 / n_features, "rank": i + 1}
            for i, name in enumerate(feature_names)
        ]

    def get_default_search_space(self) -> SearchSpace:
        """Return fake default search space.

        Returns:
            XGBoostSearchSpace with minimal ranges.
        """
        return XGBoostSearchSpace(
            max_depth=IntRangeSpec(param_type="int", low=3, high=10, log_scale=False),
            n_estimators=IntRangeSpec(param_type="int", low=50, high=500, log_scale=False),
            learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
            reg_alpha=FloatRangeSpec(param_type="float", low=1e-8, high=10.0, log_scale=True),
            reg_lambda=FloatRangeSpec(param_type="float", low=1e-8, high=10.0, log_scale=True),
            subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
            colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        )

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Return fake focused search space.

        Args:
            best_int_params: Best integer parameters from previous run.
            best_float_params: Best float parameters from previous run.

        Returns:
            XGBoostSearchSpace with minimal ranges.
        """
        return self.get_default_search_space()


def _make_fake_dataset_config(name: str = "taiwan") -> DatasetConfig:
    """Create a fake dataset config for testing.

    Args:
        name: Dataset name.

    Returns:
        DatasetConfig with minimal valid values.
    """
    return {
        "name": name,
        "display_name": f"Fake {name.title()} Dataset",
        "folder": f"{name}_data",
        "file_name": "data.csv",
        "file_format": "csv",
        "encoding": "utf-8",
        "target": {
            "column_name": "target",
            "label_type": "binary_int",
            "positive_values": (1,),
            "negative_values": (0,),
        },
        "exclude_columns": (),
        "n_samples_expected": 100,
        "n_features_expected": 10,
        "positive_class_ratio_expected": 0.3,
    }


def _make_fake_loaded_dataset(n_samples: int = 100, n_features: int = 10) -> LoadedDataset:
    """Create a fake loaded dataset for testing.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.

    Returns:
        LoadedDataset with random data.
    """
    rng = np.random.default_rng(42)
    x = rng.random((n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples).astype(np.int64)
    n_positive = int(np.sum(y))

    meta: DatasetMeta = {
        "name": "fake_dataset",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_positive": n_positive,
        "n_negative": n_samples - n_positive,
        "positive_ratio": n_positive / n_samples,
        "feature_names": tuple(f"feature_{i}" for i in range(n_features)),
        "categorical_encodings": (),
    }

    return {
        "meta": meta,
        "x": x,
        "y": y,
        "groups": None,
    }


def _make_fake_optimization_result(
    best_value: float = 0.85,
    dataset: str = "taiwan",
) -> UnifiedOptimizationResult:
    """Create a fake unified optimization result for testing.

    Args:
        best_value: Best validation AUC.
        dataset: Dataset name.

    Returns:
        UnifiedOptimizationResult with specified values.
    """
    return UnifiedOptimizationResult(
        backend="xgboost",
        status="complete",
        dataset=dataset,
        n_samples=1000,
        n_features=100,
        feature_preset="full",
        n_trials_complete=10,
        n_trials_pruned=2,
        n_trials_failed=0,
        best_trial_number=5,
        best_value=best_value,
        best_int_params=SampledIntParams(max_depth=6, n_estimators=100),
        best_float_params=SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.01,
            reg_lambda=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        best_string_params=SampledStringParams(),
        duration_seconds=10.0,
    )
