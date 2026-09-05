"""Shared fixtures for optimize script tests.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import scripts._test_hooks as _hooks_module
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
from platform_core.rich_logging import setup_rich_logging

from covenant_radar_api.worker.optimize_types import UnifiedOptimizationResult

FeaturePresetLiteral = Literal["none", "log_only", "ratios_only", "full"]


# =============================================================================
# Autouse Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _setup_rich_logging_for_tests() -> None:
    """Set up rich logging before each test that needs it."""
    setup_rich_logging(level="WARNING", show_time=False)


@pytest.fixture(autouse=True)
def _isolate_project_root(tmp_path: Path) -> Generator[None, None, None]:
    """Isolate tests from real project root to prevent history file corruption.

    Overrides the project_root_hook so that all optimize tests use a temporary
    directory instead of the real project root. This prevents xdist race
    conditions on the shared optimization_history.jsonl file.

    Args:
        tmp_path: Pytest temporary directory unique to each test.

    Yields:
        None after setting up hook, restores after test.
    """
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "data" / "external").mkdir(parents=True, exist_ok=True)

    original = _hooks_module.project_root_hook

    def _test_project_root() -> Path:
        return tmp_path

    _hooks_module.project_root_hook = _test_project_root
    yield
    _hooks_module.project_root_hook = original


# =============================================================================
# Fake Result Factories
# =============================================================================


def make_fake_result(
    backend: BackendName = "xgboost",
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_value: float = 0.85,
    n_features: int = 100,
    best_int_params: SampledIntParams | None = None,
    best_float_params: SampledFloatParams | None = None,
    best_string_params: SampledStringParams | None = None,
) -> UnifiedOptimizationResult:
    """Create a fake optimization result for testing.

    Args:
        backend: Backend name.
        dataset: Dataset name.
        feature_preset: Feature preset.
        best_value: Best validation AUC.
        n_features: Number of features.
        best_int_params: Optional int params override.
        best_float_params: Optional float params override.
        best_string_params: Optional string params override.

    Returns:
        UnifiedOptimizationResult with test values.
    """
    if best_int_params is None:
        best_int_params = SampledIntParams(max_depth=6, n_estimators=100)
    if best_float_params is None:
        best_float_params = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.01,
            reg_lambda=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    if best_string_params is None:
        best_string_params = SampledStringParams()

    return UnifiedOptimizationResult(
        backend=backend,
        status="complete",
        dataset=dataset,
        n_samples=1000,
        n_features=n_features,
        feature_preset=feature_preset,
        n_trials_complete=10,
        n_trials_pruned=2,
        n_trials_failed=0,
        best_trial_number=5,
        best_value=best_value,
        best_int_params=best_int_params,
        best_float_params=best_float_params,
        best_string_params=best_string_params,
        duration_seconds=10.0,
    )


def make_fake_xgboost_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_value: float = 0.85,
) -> UnifiedOptimizationResult:
    """Create a fake XGBoost optimization result for testing.

    Args:
        dataset: Dataset name.
        feature_preset: Feature preset.
        best_value: Best validation AUC.

    Returns:
        UnifiedOptimizationResult for xgboost backend.
    """
    return make_fake_result(
        backend="xgboost",
        dataset=dataset,
        feature_preset=feature_preset,
        best_value=best_value,
        best_int_params=SampledIntParams(max_depth=6, n_estimators=100),
        best_float_params=SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.01,
            reg_lambda=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
    )


def make_fake_mlp_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_value: float = 0.85,
) -> UnifiedOptimizationResult:
    """Create a fake MLP optimization result for testing.

    Args:
        dataset: Dataset name.
        feature_preset: Feature preset.
        best_value: Best validation AUC.

    Returns:
        UnifiedOptimizationResult for mlp backend.
    """
    return make_fake_result(
        backend="mlp",
        dataset=dataset,
        feature_preset=feature_preset,
        best_value=best_value,
        best_int_params=SampledIntParams(n_layers=3, hidden_size=128, batch_size=64),
        best_float_params=SampledFloatParams(learning_rate=0.001, dropout=0.2),
    )


def make_fake_lightgbm_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_value: float = 0.85,
) -> UnifiedOptimizationResult:
    """Create a fake LightGBM optimization result for testing.

    Args:
        dataset: Dataset name.
        feature_preset: Feature preset.
        best_value: Best validation AUC.

    Returns:
        UnifiedOptimizationResult for lightgbm backend.
    """
    return make_fake_result(
        backend="lightgbm",
        dataset=dataset,
        feature_preset=feature_preset,
        best_value=best_value,
        best_int_params=SampledIntParams(
            max_depth=-1, n_estimators=100, num_leaves=31, min_child_samples=20
        ),
        best_float_params=SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.01,
            reg_lambda=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
    )


def make_fake_lstm_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_value: float = 0.85,
) -> UnifiedOptimizationResult:
    """Create a fake LSTM optimization result for testing.

    Args:
        dataset: Dataset name.
        feature_preset: Feature preset.
        best_value: Best validation AUC.

    Returns:
        UnifiedOptimizationResult for lstm backend.
    """
    return make_fake_result(
        backend="lstm",
        dataset=dataset,
        feature_preset=feature_preset,
        best_value=best_value,
        best_int_params=SampledIntParams(hidden_size=64, num_layers=2, batch_size=32),
        best_float_params=SampledFloatParams(learning_rate=0.001, dropout=0.2),
    )


def make_fake_cleargbm_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_value: float = 0.85,
) -> UnifiedOptimizationResult:
    """Create a fake ClearGBM optimization result for testing.

    Args:
        dataset: Dataset name.
        feature_preset: Feature preset.
        best_value: Best validation AUC.

    Returns:
        UnifiedOptimizationResult for cleargbm backend.
    """
    return make_fake_result(
        backend="cleargbm",
        dataset=dataset,
        feature_preset=feature_preset,
        best_value=best_value,
        best_int_params=SampledIntParams(
            max_depth=5, n_estimators=100, min_samples_split=10, min_samples_leaf=5, max_bins=64
        ),
        best_float_params=SampledFloatParams(learning_rate=0.1, subsample=1.0),
    )


# =============================================================================
# Fake Backend Classes for save_model=True Tests
# =============================================================================


class FakePreparedClassifier:
    """Fake classifier for save_model tests."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return constant predictions."""
        n_samples: int = int(x.shape[0])
        return np.column_stack(
            [
                np.full(n_samples, 0.3, dtype=np.float64),
                np.full(n_samples, 0.7, dtype=np.float64),
            ]
        )


class FakeSaveModelBackend:
    """Fake backend for save_model tests."""

    def backend_name(self) -> BackendName:
        """Return xgboost as backend name."""
        return "xgboost"

    def capabilities(self) -> BackendCapabilities:
        """Return fake capabilities."""
        return {
            "supports_train": True,
            "supports_gpu": False,
            "supports_early_stopping": True,
            "supports_feature_importance": True,
            "model_format": "ubj",
        }

    def prepare(
        self, *, n_features: int, n_classes: int, feature_names: list[str] | None
    ) -> PreparedClassifier:
        """Return fake prepared classifier."""
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
        """Return fake train outcome."""
        model_path = output_dir / "model_1.ubj"
        model_path.write_bytes(b"fake model")
        n_samples: int = int(x_features.shape[0])
        n_features_count: int = int(x_features.shape[1])
        fake_metrics: EvalMetrics = {
            "loss": 0.35,
            "ppl": 1.42,
            "auc": 0.88,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
        }
        fake_importances: list[FeatureImportance] = [
            {"name": f"f_{i}", "importance": 1.0 / n_features_count, "rank": i + 1}
            for i in range(n_features_count)
        ]
        return {
            "model_path": str(model_path),
            "model_id": "fake-model-1",
            "samples_total": n_samples,
            "samples_train": int(n_samples * 0.7),
            "samples_val": int(n_samples * 0.15),
            "samples_test": n_samples - int(n_samples * 0.7) - int(n_samples * 0.15),
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

    def evaluate(
        self, *, model: PreparedClassifier, x: NDArray[np.float64], y: NDArray[np.int64]
    ) -> EvalMetrics:
        """Return fake evaluation metrics."""
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
        """Save fake model."""
        Path(path).write_bytes(b"fake model")

    def load(self, *, path: str) -> PreparedClassifier:
        """Load fake model."""
        return FakePreparedClassifier()

    def get_feature_importances(
        self, *, model: PreparedClassifier, feature_names: list[str] | None
    ) -> list[FeatureImportance] | None:
        """Return fake feature importances."""
        if feature_names is None:
            return None
        return [
            {"name": n, "importance": 1.0 / len(feature_names), "rank": i + 1}
            for i, n in enumerate(feature_names)
        ]

    def get_default_search_space(self) -> SearchSpace:
        """Return fake default search space."""
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
        """Return fake focused search space."""
        return self.get_default_search_space()


# =============================================================================
# Fake Dataset Factories
# =============================================================================


def make_fake_dataset_config(name: str) -> DatasetConfig:
    """Create fake dataset config for save_model tests."""
    return {
        "name": name,
        "display_name": f"Fake {name}",
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


def make_fake_loaded_dataset() -> LoadedDataset:
    """Create fake loaded dataset for save_model tests."""
    rng = np.random.default_rng(42)
    x = rng.random((100, 10))
    y = rng.integers(0, 2, size=100).astype(np.int64)
    n_positive = int(np.sum(y))
    meta: DatasetMeta = {
        "name": "fake",
        "n_samples": 100,
        "n_features": 10,
        "n_positive": n_positive,
        "n_negative": 100 - n_positive,
        "positive_ratio": n_positive / 100,
        "feature_names": tuple(f"f_{i}" for i in range(10)),
        "categorical_encodings": (),
    }
    return {"meta": meta, "x": x, "y": y, "groups": None}


__all__ = [
    "FakePreparedClassifier",
    "FakeSaveModelBackend",
    "FeaturePresetLiteral",
    "make_fake_cleargbm_result",
    "make_fake_dataset_config",
    "make_fake_lightgbm_result",
    "make_fake_loaded_dataset",
    "make_fake_lstm_result",
    "make_fake_mlp_result",
    "make_fake_result",
    "make_fake_xgboost_result",
]
