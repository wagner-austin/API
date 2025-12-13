"""Tests for BaseTabularTrainer unified training interface.

Tests the orchestration layer using XGBoost backend (no torch dependency).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.backends.registry import default_registry
from covenant_ml.base_trainer import BaseTabularTrainer
from covenant_ml.types import TrainConfig, TrainOutcome, TrainProgress


def _make_binary_dataset(
    n_samples: int = 200,
    n_features: int = 4,
    pos_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Create synthetic binary classification dataset."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    n_pos = int(n_samples * pos_ratio)
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_pos] = 1
    rng.shuffle(y)
    feature_names = [f"f{i}" for i in range(n_features)]
    return x, y, feature_names


def test_base_trainer_with_xgboost(tmp_path: Path) -> None:
    """BaseTabularTrainer delegates to XGBoost backend and returns outcome."""
    registry = default_registry()
    trainer = BaseTabularTrainer(registry)

    x, y, names = _make_binary_dataset(n_samples=100)

    config: TrainConfig = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_rounds": 2,
        "device": "cpu",
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "scale_pos_weight": 2.0,
    }

    outcome: TrainOutcome = trainer.train(
        backend="xgboost",
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=tmp_path,
        progress=None,
    )

    assert outcome["model_path"].endswith(".ubj")
    assert outcome["samples_total"] == 100
    assert outcome["samples_train"] == 60
    assert outcome["samples_val"] == 20
    assert outcome["samples_test"] == 20
    # Verify model learned (loss decreased, AUC increased)
    loss_final = outcome["val_metrics"]["loss"]
    loss_initial = 0.693  # log(2) - random binary classifier baseline
    assert loss_final < loss_initial, (
        f"Validation loss {loss_final} should be below baseline {loss_initial}"
    )
    best_val_auc = outcome["best_val_auc"]
    assert best_val_auc > 0.5, f"AUC {best_val_auc} should exceed random baseline"
    assert outcome["total_rounds"] >= 1


def test_base_trainer_with_progress_callback(tmp_path: Path) -> None:
    """BaseTabularTrainer passes progress callback to backend."""
    registry = default_registry()
    trainer = BaseTabularTrainer(registry)

    x, y, names = _make_binary_dataset(n_samples=80)

    progress_calls: list[TrainProgress] = []

    def on_progress(p: TrainProgress) -> None:
        progress_calls.append(p)

    config: TrainConfig = {
        "learning_rate": 0.1,
        "max_depth": 2,
        "n_estimators": 3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_rounds": 2,
        "device": "cpu",
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "scale_pos_weight": 2.0,
    }

    outcome: TrainOutcome = trainer.train(
        backend="xgboost",
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    # Progress callback invoked during training
    assert progress_calls, "Progress callback must be invoked"
    # Each progress has expected structure (total_rounds is n_estimators from config)
    n_estimators = config["n_estimators"]
    auc_values: list[float] = []
    for p in progress_calls:
        assert p["round"] >= 1
        assert p["total_rounds"] == n_estimators
        val_auc = p["val_auc"]
        if val_auc is None:
            raise AssertionError("val_auc must not be None during XGBoost training")
        assert 0.0 <= val_auc <= 1.0
        auc_values.append(val_auc)

    # Verify model learned (loss decreased over training)
    loss_final = outcome["val_metrics"]["loss"]
    loss_initial = 0.693  # log(2) - random binary classifier baseline
    assert loss_final < loss_initial, (
        f"Final loss {loss_final} should be below baseline {loss_initial}"
    )
    # Verify AUC improved over training
    best_auc = max(auc_values)
    assert best_auc > 0.5, f"Best AUC {best_auc} should exceed random baseline"
    # outcome["best_val_auc"] is the best AUC seen during training
    assert outcome["best_val_auc"] == best_auc
