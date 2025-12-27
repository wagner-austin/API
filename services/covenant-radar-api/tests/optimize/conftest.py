"""Shared fixtures for optimize script tests.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    PreparedClassifier,
    ProgressCallback,
)
from covenant_ml.datasets import DatasetConfig, DatasetMeta, LoadedDataset
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    ClearGBMConfig,
    EvalMetrics,
    FeatureImportance,
    LightGBMConfig,
    LSTMConfig,
    MLPConfig,
    TrainConfig,
    TrainOutcome,
)
from numpy.typing import NDArray
from platform_core.logging import setup_rich_logging
from scripts._test_hooks import (
    ClearGBMOptimizationResult,
    LightGBMOptimizationResult,
    LSTMOptimizationResult,
    MLPOptimizationResult,
    XGBoostOptimizationResult,
)

FeaturePresetLiteral = Literal["none", "log_only", "ratios_only", "full"]


# =============================================================================
# Autouse Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _setup_rich_logging_for_tests() -> None:
    """Set up rich logging before each test that needs it."""
    setup_rich_logging(level="WARNING", show_time=False)


# =============================================================================
# Fake Result Factories
# =============================================================================


def make_fake_train_config() -> TrainConfig:
    """Create a fake TrainConfig for testing."""
    return TrainConfig(
        device="cpu",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        early_stopping_rounds=10,
        reg_alpha=0.01,
        reg_lambda=0.01,
    )


def make_fake_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_val_auc: float = 0.85,
    n_features: int = 100,
) -> XGBoostOptimizationResult:
    """Create a fake optimization result for testing."""
    return {
        "backend": "xgboost",
        "status": "complete",
        "dataset": dataset,
        "n_samples": 1000,
        "n_features": n_features,
        "feature_preset": feature_preset,
        "n_trials_complete": 10,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "best_trial_number": 5,
        "best_val_auc": best_val_auc,
        "best_max_depth": 6,
        "best_n_estimators": 100,
        "best_learning_rate": 0.1,
        "best_reg_alpha": 0.01,
        "best_reg_lambda": 0.01,
        "best_subsample": 0.8,
        "best_colsample_bytree": 0.8,
        "duration_seconds": 10.0,
        "recommended_config": make_fake_train_config(),
    }


def make_fake_mlp_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_val_auc: float = 0.85,
) -> MLPOptimizationResult:
    """Create a fake MLP optimization result for testing."""
    return {
        "backend": "mlp",
        "status": "complete",
        "dataset": dataset,
        "n_samples": 1000,
        "n_features": 100,
        "feature_preset": feature_preset,
        "n_trials_complete": 10,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "best_trial_number": 5,
        "best_val_auc": best_val_auc,
        "best_n_layers": 3,
        "best_hidden_size": 128,
        "best_learning_rate": 0.001,
        "best_dropout": 0.2,
        "best_batch_size": 64,
        "duration_seconds": 10.0,
        "recommended_config": MLPConfig(
            device="cpu",
            precision="fp32",
            optimizer="adamw",
            hidden_sizes=(128, 64),
            learning_rate=0.001,
            dropout=0.2,
            batch_size=64,
            n_epochs=100,
            early_stopping_patience=10,
            random_state=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        ),
    }


def make_fake_lightgbm_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_val_auc: float = 0.85,
) -> LightGBMOptimizationResult:
    """Create a fake LightGBM optimization result for testing.

    Note: best_max_depth is always -1 (unlimited) because LightGBM uses
    num_leaves as the primary complexity control, not max_depth.
    """
    return {
        "backend": "lightgbm",
        "status": "complete",
        "dataset": dataset,
        "n_samples": 1000,
        "n_features": 100,
        "feature_preset": feature_preset,
        "n_trials_complete": 10,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "best_trial_number": 5,
        "best_val_auc": best_val_auc,
        "best_max_depth": -1,  # Fixed: unlimited depth, num_leaves controls complexity
        "best_n_estimators": 100,
        "best_num_leaves": 31,
        "best_learning_rate": 0.1,
        "best_reg_alpha": 0.01,
        "best_reg_lambda": 0.01,
        "best_subsample": 0.8,
        "best_colsample_bytree": 0.8,
        "duration_seconds": 10.0,
        "recommended_config": LightGBMConfig(
            device="cpu",
            max_depth=-1,  # Fixed: unlimited depth, num_leaves controls complexity
            n_estimators=100,
            num_leaves=31,
            min_child_samples=20,
            learning_rate=0.1,
            reg_alpha=0.01,
            reg_lambda=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            early_stopping_rounds=10,
        ),
    }


def make_fake_lstm_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_val_auc: float = 0.85,
) -> LSTMOptimizationResult:
    """Create a fake LSTM optimization result for testing."""
    return {
        "backend": "lstm",
        "status": "complete",
        "dataset": dataset,
        "n_samples": 1000,
        "n_features": 100,
        "feature_preset": feature_preset,
        "n_trials_complete": 10,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "best_trial_number": 5,
        "best_val_auc": best_val_auc,
        "best_hidden_size": 64,
        "best_num_layers": 2,
        "best_learning_rate": 0.001,
        "best_dropout": 0.2,
        "best_batch_size": 32,
        "duration_seconds": 10.0,
        "recommended_config": LSTMConfig(
            device="cpu",
            precision="fp32",
            hidden_size=64,
            num_layers=2,
            learning_rate=0.001,
            dropout=0.2,
            batch_size=32,
            n_epochs=100,
            early_stopping_patience=10,
            sequence_length=10,
            bidirectional=False,
            random_state=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        ),
    }


def make_fake_cleargbm_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_val_auc: float = 0.85,
) -> ClearGBMOptimizationResult:
    """Create a fake ClearGBM optimization result for testing."""
    return {
        "backend": "cleargbm",
        "status": "complete",
        "dataset": dataset,
        "n_samples": 1000,
        "n_features": 100,
        "feature_preset": feature_preset,
        "n_trials_complete": 10,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "best_trial_number": 5,
        "best_val_auc": best_val_auc,
        "best_max_depth": 5,
        "best_n_estimators": 100,
        "best_learning_rate": 0.1,
        "best_min_samples_split": 10,
        "best_min_samples_leaf": 5,
        "best_max_bins": 64,
        "best_subsample": 1.0,
        "duration_seconds": 10.0,
        "recommended_config": ClearGBMConfig(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=None,
            max_bins=64,
            subsample=1.0,
            random_state=42,
            track_contributions=True,
            monotonic_constraints=None,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            early_stopping_rounds=10,
        ),
    }


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
        progress: ProgressCallback | None,
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
    return {"meta": meta, "x": x, "y": y}


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
    "make_fake_train_config",
]
