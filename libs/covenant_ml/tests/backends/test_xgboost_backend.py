"""Tests for XGBoost backend implementation.

Tests the backend interface including train, evaluate, save, load, and error paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.backends.protocol import ClassifierBackend
from covenant_ml.backends.xgboost import create_xgboost_backend
from covenant_ml.types import (
    ClassifierTrainConfig,
    MLPConfig,
    TrainConfig,
    TrainOutcome,
)


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


def _train_for_reload(backend: ClassifierBackend, tmp_path: Path) -> TrainOutcome:
    """Train a small model so its persisted file can be reloaded.

    Args:
        backend: Backend under test.
        tmp_path: Pytest temporary directory.

    Returns:
        The training outcome, whose model_path points at the saved booster.
    """
    x, y, names = _make_binary_dataset()
    config: TrainConfig = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_rounds": 5,
        "device": "cpu",
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    return _invoke_backend_train(backend, x, y, names, config, tmp_path)


def test_xgboost_backend_prepare_raises() -> None:
    """Prepare is not supported, matching every other tree backend.

    It previously returned a placeholder whose predict_proba raised, which
    deferred the failure to the first prediction instead of the call that
    was actually wrong.
    """
    backend = create_xgboost_backend()

    with pytest.raises(RuntimeError, match="prepare not supported"):
        backend.prepare(n_features=4, n_classes=2, feature_names=None)


def test_xgboost_backend_save_raises(tmp_path: Path) -> None:
    """Save is not supported; the trainer persists the model.

    It previously opened the path in "wb" and wrote zero bytes, so calling it
    with an existing model path destroyed that model. A test asserted the
    resulting file was empty, which made the data loss look intended.
    """
    backend = create_xgboost_backend()
    outcome = _train_for_reload(backend, tmp_path)
    loaded = backend.load(path=outcome["model_path"])

    with pytest.raises(RuntimeError, match="save not supported"):
        backend.save(model=loaded, path=outcome["model_path"])


def test_xgboost_backend_save_does_not_touch_the_target(tmp_path: Path) -> None:
    """The refused save leaves the model file byte-for-byte intact."""
    backend = create_xgboost_backend()
    outcome = _train_for_reload(backend, tmp_path)
    model_file = Path(outcome["model_path"])
    before = model_file.read_bytes()
    loaded = backend.load(path=outcome["model_path"])

    with pytest.raises(RuntimeError, match="save not supported"):
        backend.save(model=loaded, path=str(model_file))

    assert model_file.read_bytes() == before


def test_xgboost_backend_load_and_predict(tmp_path: Path) -> None:
    """Load restores a trained model that predicts, rather than a placeholder."""
    backend = create_xgboost_backend()
    outcome = _train_for_reload(backend, tmp_path)
    x, y, _ = _make_binary_dataset()

    loaded = backend.load(path=outcome["model_path"])
    proba = loaded.predict_proba(x)

    assert proba.shape == (x.shape[0], 2)
    finite: NDArray[np.bool_] = np.isfinite(proba)
    assert int(np.count_nonzero(finite)) == int(proba.size)
    # Weights that failed to arrive would leave predictions at chance.
    metrics = backend.evaluate(model=loaded, x=x, y=y)
    assert metrics["auc"] > 0.7


def test_xgboost_backend_get_feature_importances_returns_none(tmp_path: Path) -> None:
    """Feature importances returns None (provided by TrainOutcome)."""
    backend = create_xgboost_backend()
    outcome = _train_for_reload(backend, tmp_path)
    loaded = backend.load(path=outcome["model_path"])

    result = backend.get_feature_importances(model=loaded, feature_names=["a", "b", "c", "d"])
    assert result is None


class _FailingClassifier:
    """Classifier whose prediction fails, to show evaluate does not swallow it."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Raise, standing in for any model that cannot predict.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("prediction unavailable")


def test_xgboost_backend_evaluate_propagates_prediction_failure() -> None:
    """Evaluate routes through predict_proba and lets its failure surface."""
    backend = create_xgboost_backend()

    x = np.zeros((10, 4), dtype=np.float64)
    y = np.zeros(10, dtype=np.int64)

    with pytest.raises(RuntimeError, match="prediction unavailable"):
        backend.evaluate(model=_FailingClassifier(), x=x, y=y)


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
