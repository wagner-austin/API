"""ClearGBM backend integration tests with actual training.

Tests the full training loop, prediction, save/load, and error paths.
Uses real US bankruptcy data for integration tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from covenant_ml.backends.cleargbm import (
    create_cleargbm_backend,
)
from covenant_ml.types import (
    MLPConfig,
    TrainConfig,
    TrainProgress,
)
from tests.backends.cleargbm._cleargbm_fixtures import (
    _BASELINE_LOSS,
    _invoke_cleargbm_train,
    _make_cleargbm_config,
    _make_synthetic_dataset,
)

from ...conftest import load_us_bankruptcy_data


def test_cleargbm_backend_train_returns_outcome(tmp_path: Path) -> None:
    """ClearGBMBackend trains and returns TrainOutcome with all required fields."""
    backend = create_cleargbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config = _make_cleargbm_config(n_estimators=10, max_depth=4)

    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Verify outcome structure
    assert outcome["samples_total"] == len(y)
    assert outcome["samples_train"] > 0
    assert outcome["samples_val"] > 0
    assert outcome["samples_test"] > 0

    # Verify metrics exist and are reasonable
    assert 0.0 <= outcome["train_metrics"]["auc"] <= 1.0
    assert 0.0 <= outcome["val_metrics"]["auc"] <= 1.0
    assert 0.0 <= outcome["test_metrics"]["auc"] <= 1.0
    assert outcome["best_val_auc"] > 0.0

    # Verify model was saved
    assert Path(outcome["model_path"]).exists()
    assert outcome["model_path"].endswith(".json")

    # Verify feature importances exist with correct count
    assert len(outcome["feature_importances"]) == len(names)
    assert outcome["feature_importances"][0]["rank"] == 1

    # Verify scale_pos_weight was computed
    assert outcome["scale_pos_weight_computed"] > 0.0

    # Verify loss decreased from baseline
    val_loss = outcome["val_metrics"]["loss"]
    assert val_loss < _BASELINE_LOSS, (
        f"Validation loss {val_loss} should be below baseline {_BASELINE_LOSS}"
    )


def test_cleargbm_backend_train_without_feature_names(tmp_path: Path) -> None:
    """ClearGBMBackend generates feature names if not provided."""
    backend = create_cleargbm_backend()
    x, y, _ = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)

    outcome = backend.train(
        x_features=x,
        y_labels=y,
        feature_names=None,  # Not provided
        config=config,
        output_dir=tmp_path,
        progress=None,
    )

    # Should generate f0, f1, f2, etc.
    assert outcome["feature_importances"][0]["name"].startswith("f")
    # Model should train successfully
    assert outcome["samples_total"] == 100
    # Verify loss decreased from baseline
    val_loss = outcome["val_metrics"]["loss"]
    assert val_loss < _BASELINE_LOSS, (
        f"Validation loss {val_loss} should be below baseline {_BASELINE_LOSS}"
    )


def test_cleargbm_backend_config_type_validation(tmp_path: Path) -> None:
    """ClearGBMBackend raises on non-ClearGBM config."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    # Try XGBoost config (wrong type)
    xgb_config: TrainConfig = {
        "device": "cpu",
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
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }

    with pytest.raises(RuntimeError, match="ClearGBMBackend requires ClearGBMConfig"):
        _invoke_cleargbm_train(backend, x, y, names, xgb_config, tmp_path)


def test_cleargbm_backend_train_with_mlp_config_raises(tmp_path: Path) -> None:
    """ClearGBMBackend raises on MLPConfig."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    mlp_config: MLPConfig = {
        "device": "cpu",
        "precision": "fp32",
        "optimizer": "adamw",
        "hidden_sizes": (32,),
        "learning_rate": 0.01,
        "batch_size": 32,
        "n_epochs": 2,
        "dropout": 0.0,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_patience": 5,
    }

    with pytest.raises(RuntimeError, match="ClearGBMBackend requires ClearGBMConfig"):
        _invoke_cleargbm_train(backend, x, y, names, mlp_config, tmp_path)


def test_cleargbm_backend_raises_on_no_positive_samples(tmp_path: Path) -> None:
    """ClearGBMBackend raises if training set has no positive samples."""
    backend = create_cleargbm_backend()

    # Create dataset with no positives
    x = np.random.default_rng(42).standard_normal((100, 8)).astype(np.float64)
    y = np.zeros(100, dtype=np.int64)  # All negative
    names = [f"f{i}" for i in range(8)]

    config = _make_cleargbm_config(n_estimators=3)

    with pytest.raises(ValueError, match="no positive samples"):
        _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)


def test_cleargbm_backend_progress_callback_is_noop_on_native_path(tmp_path: Path) -> None:
    """Passing a progress callback is accepted but never invoked on the native path.

    The Rust training loop is a single native call — it does not surface per-tree
    progress to Python. The wrapper's ``train()`` documents this and skips the
    callback rather than emitting synthetic ``TrainProgress`` events. This test
    guards the documented behavior: training must succeed to convergence even
    when a callback is present, and the callback must never fire.
    """
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)

    progress_reports: list[TrainProgress] = []

    def track_progress(p: TrainProgress) -> None:
        progress_reports.append(p)

    outcome = backend.train(
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=tmp_path,
        progress=track_progress,
    )

    # Documented no-op on the native path.
    assert progress_reports == []

    # Training still completes and beats the random baseline.
    val_loss = outcome["val_metrics"]["loss"]
    assert val_loss < _BASELINE_LOSS, (
        f"Validation loss {val_loss} should be below baseline {_BASELINE_LOSS}"
    )


def test_cleargbm_backend_train_with_subsampling(tmp_path: Path) -> None:
    """ClearGBMBackend works with row subsampling."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)
    config["subsample"] = 0.7

    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Should complete successfully
    assert outcome["samples_total"] == 100


def test_cleargbm_backend_train_leaf_wise(tmp_path: Path) -> None:
    """ClearGBMBackend passes growth_strategy/num_leaves through to the core.

    A leaf-wise train must complete and produce a loadable model; a broken
    pass-through would surface as the Rust config validator rejecting the
    depth_wise-shaped pair.
    """
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)
    config["growth_strategy"] = "leaf_wise"
    config["num_leaves"] = 8

    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    assert outcome["samples_total"] == 100
    assert Path(outcome["model_path"]).exists()


def test_cleargbm_backend_train_early_stopping(tmp_path: Path) -> None:
    """ClearGBMBackend tracks early stopping progress."""
    backend = create_cleargbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    # Use more estimators to potentially trigger early stopping
    config = _make_cleargbm_config(n_estimators=20, max_depth=2)
    config["early_stopping_rounds"] = 3

    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Verify early_stopped field is boolean (value depends on data)
    early_stopped = outcome["early_stopped"]
    assert early_stopped is True or early_stopped is False
    # Verify best_round is tracked
    assert outcome["best_round"] >= 1


def test_cleargbm_backend_train_different_depths(tmp_path: Path) -> None:
    """ClearGBMBackend works with various max_depth values."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    for max_depth in [2, 4, 6]:
        config = _make_cleargbm_config(n_estimators=3, max_depth=max_depth)
        outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)
        assert outcome["samples_total"] == 100, f"Failed for max_depth={max_depth}"
