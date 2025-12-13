"""Tests for XGBoost backend implementation.

Tests the backend interface including train, evaluate, save, load, and error paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.backends.protocol import ClassifierBackend
from covenant_ml.backends.xgboost_backend import create_xgboost_backend
from covenant_ml.types import ClassifierTrainConfig, MLPConfig, TrainConfig, TrainOutcome


def _invoke_backend_train(
    backend: ClassifierBackend,
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    names: list[str] | None,
    config: ClassifierTrainConfig,
    output_dir: Path,
) -> TrainOutcome:
    """Helper to invoke backend train (isolates .train() call for guard)."""
    return backend.train(
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=output_dir,
        progress=None,
    )


def _make_binary_dataset(
    n_samples: int = 100,
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


def test_xgboost_backend_train(tmp_path: Path) -> None:
    """XGBoost backend trains and returns outcome with metrics."""
    backend = create_xgboost_backend()
    x, y, names = _make_binary_dataset()

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

    outcome: TrainOutcome = backend.train(
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=tmp_path,
        progress=None,
    )

    assert outcome["model_path"].endswith(".ubj")
    assert Path(outcome["model_path"]).exists()
    assert outcome["samples_total"] == 100
    assert outcome["samples_train"] == 60
    assert outcome["samples_val"] == 20
    assert outcome["samples_test"] == 20
    # Verify model learned (loss decreased)
    loss_final = outcome["val_metrics"]["loss"]
    loss_initial = 0.693  # log(2) - random binary classifier baseline
    assert loss_final < loss_initial, (
        f"Validation loss {loss_final} should be below baseline {loss_initial}"
    )
    best_auc = outcome["best_val_auc"]
    assert best_auc > 0.5, f"AUC {best_auc} should exceed random baseline"
    assert outcome["total_rounds"] >= 1


def test_xgboost_backend_train_generates_feature_names(tmp_path: Path) -> None:
    """XGBoost backend generates feature names when not provided."""
    backend = create_xgboost_backend()
    x, y, _ = _make_binary_dataset(n_features=3)

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

    outcome: TrainOutcome = backend.train(
        x_features=x,
        y_labels=y,
        feature_names=None,  # Backend should generate names
        config=config,
        output_dir=tmp_path,
        progress=None,
    )

    # Should complete without error and produce valid model
    assert outcome["samples_total"] == 100
    # Verify model learned (loss decreased)
    loss_final = outcome["val_metrics"]["loss"]
    loss_initial = 0.693  # log(2) - random binary classifier baseline
    assert loss_final < loss_initial, (
        f"Validation loss {loss_final} should be below baseline {loss_initial}"
    )
    best_auc = outcome["best_val_auc"]
    assert best_auc > 0.5, f"AUC {best_auc} should exceed random baseline"
    assert Path(outcome["model_path"]).exists()


def test_xgboost_backend_config_type_validation(tmp_path: Path) -> None:
    """XGBoost backend raises RuntimeError when given MLPConfig instead of TrainConfig."""
    backend = create_xgboost_backend()
    x, y, names = _make_binary_dataset()

    mlp_config: MLPConfig = {
        "device": "cpu",
        "precision": "fp32",
        "optimizer": "adamw",
        "hidden_sizes": (64, 32),
        "learning_rate": 0.001,
        "batch_size": 32,
        "n_epochs": 10,
        "dropout": 0.1,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_patience": 5,
    }

    with pytest.raises(RuntimeError, match="XGBoostBackend requires TrainConfig"):
        _invoke_backend_train(backend, x, y, names, mlp_config, tmp_path)


def test_xgboost_backend_prepare_returns_placeholder() -> None:
    """Prepare returns a placeholder that raises on predict_proba."""
    backend = create_xgboost_backend()
    prepared = backend.prepare(n_features=4, n_classes=2, feature_names=None)

    x = np.zeros((1, 4), dtype=np.float64)
    with pytest.raises(RuntimeError, match="not available in this context"):
        prepared.predict_proba(x)


def test_xgboost_backend_save_creates_empty_file(tmp_path: Path) -> None:
    """Save creates an empty file (actual saving done in train)."""
    backend = create_xgboost_backend()
    prepared = backend.prepare(n_features=4, n_classes=2, feature_names=None)

    save_path = tmp_path / "model.ubj"
    backend.save(model=prepared, path=str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size == 0


def test_xgboost_backend_load_returns_placeholder() -> None:
    """Load returns a placeholder (actual loading uses different pipeline)."""
    backend = create_xgboost_backend()
    loaded = backend.load(path="nonexistent.ubj")

    x = np.zeros((1, 4), dtype=np.float64)
    with pytest.raises(RuntimeError, match="not available in this context"):
        loaded.predict_proba(x)


def test_xgboost_backend_get_feature_importances_returns_none() -> None:
    """Feature importances returns None (provided by TrainOutcome)."""
    backend = create_xgboost_backend()
    prepared = backend.prepare(n_features=4, n_classes=2, feature_names=None)

    result = backend.get_feature_importances(model=prepared, feature_names=["a", "b", "c", "d"])
    assert result is None


def test_xgboost_backend_evaluate_uses_predict_proba() -> None:
    """Evaluate calls predict_proba on the model."""
    backend = create_xgboost_backend()
    # Using prepared model that raises shows evaluate tries to use predict_proba
    prepared = backend.prepare(n_features=4, n_classes=2, feature_names=None)

    x = np.zeros((10, 4), dtype=np.float64)
    y = np.zeros(10, dtype=np.int64)

    with pytest.raises(RuntimeError, match="not available in this context"):
        backend.evaluate(model=prepared, x=x, y=y)


class _FakePreparedClassifier:
    """Fake classifier for testing evaluate path."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return fake probabilities (50/50 for all samples)."""
        n_samples = int(x.shape[0])
        return np.full((n_samples, 2), 0.5, dtype=np.float64)


def test_xgboost_backend_evaluate_computes_metrics() -> None:
    """Evaluate computes metrics from model predictions."""
    backend = create_xgboost_backend()
    fake_model = _FakePreparedClassifier()

    x = np.zeros((20, 4), dtype=np.float64)
    # Binary labels with some of each class
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    metrics = backend.evaluate(model=fake_model, x=x, y=y)

    # With 50/50 predictions, AUC should be around 0.5
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] > 0.0  # Log loss is always positive
