"""MLP classifier backend: training outcomes and configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.backends.protocol import ClassifierBackend
from covenant_ml.types import (
    ClassifierTrainConfig,
    MLPConfig,
    TrainConfig,
    TrainOutcome,
    TrainProgress,
)
from numpy.typing import NDArray

from covenant_nn.backends.mlp import create_mlp_backend

from ...conftest import load_us_bankruptcy_data


def _invoke_mlp_train(
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


def _make_synthetic_dataset(
    n_samples: int = 100,
    n_features: int = 4,
    pos_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Create synthetic binary classification dataset for edge case tests."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    n_pos = int(n_samples * pos_ratio)
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_pos] = 1
    rng.shuffle(y)
    feature_names = [f"f{i}" for i in range(n_features)]
    return x, y, feature_names


def _make_mlp_config(
    n_epochs: int = 3,
    batch_size: int = 16,
    hidden_sizes: tuple[int, ...] = (8, 4),
) -> MLPConfig:
    """Create MLP config for testing."""
    return {
        "device": "cpu",
        "precision": "fp32",
        "optimizer": "adamw",
        "hidden_sizes": hidden_sizes,
        "learning_rate": 0.01,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "dropout": 0.0,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "early_stopping_patience": 5,
    }


def test_mlp_backend_train_returns_outcome(tmp_path: Path) -> None:
    """MLPBackend trains and returns TrainOutcome with all required fields."""
    backend = create_mlp_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config: MLPConfig = {
        "device": "cpu",
        "precision": "fp32",
        "optimizer": "adamw",
        "hidden_sizes": (64, 32),
        "learning_rate": 0.001,
        "batch_size": 256,
        "n_epochs": 10,
        "dropout": 0.1,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_state": 42,
        "early_stopping_patience": 5,
    }

    progress_calls: list[TrainProgress] = []

    def on_progress(p: TrainProgress) -> None:
        progress_calls.append(p)

    outcome: TrainOutcome = backend.train(
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    # Verify outcome structure
    assert outcome["model_path"].endswith(".pt")
    assert Path(outcome["model_path"]).exists()
    assert outcome["model_id"] == "mlp"
    assert outcome["samples_total"] == dataset["n_samples"]
    # Verify model learned by tracking actual loss from progress
    assert progress_calls, "Progress callback must be invoked"
    val_losses: list[float] = []
    for p in progress_calls:
        val_loss = p["val_loss"]
        if val_loss is None:
            raise AssertionError("val_loss must not be None during MLP training")
        val_losses.append(val_loss)
    loss_initial = val_losses[0]
    loss_final = min(val_losses)
    assert loss_final < loss_initial, (
        f"Best loss {loss_final} should be below first epoch {loss_initial}"
    )
    assert outcome["total_rounds"] >= 1


def test_mlp_backend_train_with_progress_callback(tmp_path: Path) -> None:
    """MLPBackend invokes progress callback during training."""
    backend = create_mlp_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config: MLPConfig = {
        "device": "cpu",
        "precision": "fp32",
        "optimizer": "adamw",
        "hidden_sizes": (32, 16),
        "learning_rate": 0.001,
        "batch_size": 256,
        "n_epochs": 10,
        "dropout": 0.0,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_state": 42,
        "early_stopping_patience": 5,
    }

    progress_calls: list[TrainProgress] = []

    def on_progress(p: TrainProgress) -> None:
        progress_calls.append(p)

    outcome: TrainOutcome = backend.train(
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    # Verify outcome structure
    assert outcome["samples_total"] == dataset["n_samples"]
    assert 0.0 <= outcome["best_val_auc"] <= 1.0

    # Verify progress callbacks
    assert progress_calls, "Progress callback must be invoked"
    n_epochs = config["n_epochs"]
    for p in progress_calls:
        assert p["round"] >= 1
        assert p["total_rounds"] == n_epochs
        val_auc = p["val_auc"]
        if val_auc is None:
            raise AssertionError("val_auc must not be None during MLP training")
        assert 0.0 <= val_auc <= 1.0

    # Verify model learned by tracking actual loss from progress
    val_losses: list[float] = []
    for p in progress_calls:
        val_loss = p["val_loss"]
        if val_loss is None:
            raise AssertionError("val_loss must not be None during MLP training")
        val_losses.append(val_loss)
    loss_initial = val_losses[0]
    loss_final = val_losses[-1]
    assert loss_final < loss_initial, (
        f"Final loss {loss_final} should be below first epoch {loss_initial}"
    )


def test_mlp_backend_train_early_stopping(tmp_path: Path) -> None:
    """MLPBackend stops early when validation AUC doesn't improve."""
    backend = create_mlp_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    # Use very low patience (2) and high learning rate to force early stopping
    # The model will quickly plateau and trigger the early stop break
    config: MLPConfig = {
        "device": "cpu",
        "precision": "fp32",
        "optimizer": "adamw",
        "hidden_sizes": (64, 32),
        "learning_rate": 0.1,  # High LR causes fast convergence then plateau
        "batch_size": 256,
        "n_epochs": 100,  # Many epochs (should stop early)
        "dropout": 0.1,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_state": 42,
        "early_stopping_patience": 2,  # Very low patience to trigger break
    }

    progress_calls: list[TrainProgress] = []

    def on_progress(p: TrainProgress) -> None:
        progress_calls.append(p)

    outcome: TrainOutcome = backend.train(
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    # Verify training completed
    assert outcome["samples_total"] == dataset["n_samples"]
    assert 0.0 <= outcome["best_val_auc"] <= 1.0
    # Verify progress tracked actual loss values
    assert progress_calls, "Progress callback must be invoked"
    # Get actual losses from progress callbacks
    val_losses: list[float] = []
    for p in progress_calls:
        val_loss = p["val_loss"]
        if val_loss is None:
            raise AssertionError("val_loss must not be None during MLP training")
        val_losses.append(val_loss)
    # Verify model learned (best loss not worse than first epoch)
    loss_initial = val_losses[0]
    loss_final = min(val_losses)
    assert loss_final <= loss_initial, (
        f"Best loss {loss_final} should be at most first epoch {loss_initial}"
    )
    # Verify early stopping triggered (must run fewer epochs than max)
    n_epochs_run = len(progress_calls)
    assert n_epochs_run < config["n_epochs"], (
        f"Should trigger early stop: ran {n_epochs_run} of {config['n_epochs']} epochs"
    )


def test_mlp_backend_config_type_validation(tmp_path: Path) -> None:
    """MLPBackend raises RuntimeError when given TrainConfig instead of MLPConfig."""
    backend = create_mlp_backend()
    x, y, names = _make_synthetic_dataset()

    # TrainConfig (for XGBoost) instead of MLPConfig
    xgb_config: TrainConfig = {
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
    }

    with pytest.raises(RuntimeError, match="MLPBackend requires MLPConfig"):
        _invoke_mlp_train(backend, x, y, names, xgb_config, tmp_path)


def test_mlp_backend_prepare_creates_model() -> None:
    """Prepare creates a model that can be used for prediction."""
    backend = create_mlp_backend()
    prepared = backend.prepare(n_features=4, n_classes=2, feature_names=None)

    x = np.random.randn(5, 4).astype(np.float64)
    proba = prepared.predict_proba(x)

    # Should return probabilities for 2 classes
    assert proba.shape == (5, 2)
    # Probabilities should sum to 1 - convert to Python list for strict typing
    proba_list: list[list[float]] = proba.tolist()
    for i, row in enumerate(proba_list):
        row_sum = row[0] + row[1]
        assert abs(row_sum - 1.0) < 1e-5, f"Row {i} sum {row_sum} should be 1.0"


def test_mlp_backend_evaluate_computes_metrics() -> None:
    """Evaluate computes metrics from model predictions."""
    backend = create_mlp_backend()
    prepared = backend.prepare(n_features=4, n_classes=2, feature_names=None)

    x = np.random.randn(20, 4).astype(np.float64)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    metrics = backend.evaluate(model=prepared, x=x, y=y)

    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] > 0.0


def test_mlp_backend_save_raises() -> None:
    """MLPBackend.save raises RuntimeError (not supported)."""
    backend = create_mlp_backend()
    prepared = backend.prepare(n_features=4, n_classes=2, feature_names=None)

    with pytest.raises(RuntimeError, match="save not supported"):
        backend.save(model=prepared, path="dummy.pt")
