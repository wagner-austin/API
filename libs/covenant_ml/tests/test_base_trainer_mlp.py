"""Tests for BaseTabularTrainer with MLP backend.

Tests the orchestration layer delegates correctly to MLP using real US bankruptcy data.
"""

from __future__ import annotations

from pathlib import Path

from covenant_ml.backends.registry import default_registry
from covenant_ml.base_trainer import BaseTabularTrainer
from covenant_ml.types import MLPConfig, TrainOutcome, TrainProgress

from .conftest import load_us_bankruptcy_data


def test_base_trainer_with_mlp(tmp_path: Path) -> None:
    """BaseTabularTrainer delegates to MLP backend and returns outcome."""
    registry = default_registry()
    trainer = BaseTabularTrainer(registry)

    dataset = load_us_bankruptcy_data()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    progress_calls: list[TrainProgress] = []

    def on_progress(p: TrainProgress) -> None:
        progress_calls.append(p)

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

    outcome: TrainOutcome = trainer.train(
        backend="mlp",
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    assert outcome["model_path"].endswith(".pt")
    assert outcome["samples_total"] == dataset["n_samples"]

    # Collect val_loss from progress
    val_losses: list[float] = []
    for p in progress_calls:
        val_loss = p["val_loss"]
        if val_loss is not None:
            val_losses.append(val_loss)

    # Verify model learned (loss decreased from first epoch)
    loss_initial = val_losses[0]
    loss_final = min(val_losses)
    assert loss_final < loss_initial, (
        f"Best loss {loss_final} should be below first epoch {loss_initial}"
    )
    assert outcome["total_rounds"] >= 1


def test_base_trainer_mlp_with_progress_callback(tmp_path: Path) -> None:
    """BaseTabularTrainer passes progress callback to MLP backend."""
    registry = default_registry()
    trainer = BaseTabularTrainer(registry)

    dataset = load_us_bankruptcy_data()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    progress_calls: list[TrainProgress] = []

    def on_progress(p: TrainProgress) -> None:
        progress_calls.append(p)

    config: MLPConfig = {
        "device": "cpu",
        "precision": "fp32",
        "optimizer": "adamw",
        "hidden_sizes": (32,),
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

    outcome: TrainOutcome = trainer.train(
        backend="mlp",
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    # Progress callback invoked during training
    assert progress_calls, "Progress callback must be invoked"
    # Each progress has expected structure
    n_epochs = config["n_epochs"]
    val_losses: list[float] = []
    for p in progress_calls:
        assert p["round"] >= 1
        assert p["total_rounds"] == n_epochs
        val_loss = p["val_loss"]
        if val_loss is not None:
            val_losses.append(val_loss)

    # Verify model learned (loss decreased from first epoch)
    loss_initial = val_losses[0]
    loss_final = min(val_losses)
    assert loss_final < loss_initial, (
        f"Best loss {loss_final} should be below first epoch {loss_initial}"
    )
    assert outcome["total_rounds"] >= 1
