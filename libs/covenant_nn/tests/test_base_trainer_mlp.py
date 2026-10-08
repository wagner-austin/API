"""Tests for BaseTabularTrainer with MLP backend.

Tests the orchestration layer delegates correctly to MLP using real US bankruptcy data.
"""

from __future__ import annotations

from pathlib import Path

from covenant_ml.backends.registry import BackendRegistration, ClassifierRegistry
from covenant_ml.base_trainer import BaseTabularTrainer
from covenant_ml.types import BackendName, MLPConfig, TrainOutcome, TrainProgress
from platform_ml import OptimizerName, RequestedDevice, RequestedPrecision

from covenant_nn.backends.mlp import create_mlp_backend

from .conftest import load_us_bankruptcy_sample


def _make_mlp_registry() -> ClassifierRegistry:
    """Create a classifier registry with only the MLP backend registered."""
    registry = ClassifierRegistry()
    registry.register(BackendName.MLP, BackendRegistration(create_mlp_backend))
    return registry


def test_base_trainer_with_mlp(tmp_path: Path) -> None:
    """BaseTabularTrainer delegates to MLP, passes its progress callback, returns outcome."""
    registry = _make_mlp_registry()
    trainer = BaseTabularTrainer(registry)

    dataset = load_us_bankruptcy_sample()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    progress_calls: list[TrainProgress] = []

    def on_progress(p: TrainProgress) -> None:
        progress_calls.append(p)

    config: MLPConfig = {
        "device": RequestedDevice.CPU,
        "precision": RequestedPrecision.FP32,
        "optimizer": OptimizerName.ADAMW,
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
        backend=BackendName.MLP,
        x_features=x,
        y_labels=y,
        feature_names=names,
        config=config,
        output_dir=tmp_path,
        progress=on_progress,
    )

    assert outcome["model_path"].endswith(".pt")
    assert outcome["samples_total"] == dataset["n_samples"]
    assert outcome["total_rounds"] >= 1

    # The backend reported every epoch through the trainer's callback
    assert progress_calls, "Progress callback must be invoked"
    val_losses: list[float] = []
    for p in progress_calls:
        assert p["round"] >= 1
        assert p["total_rounds"] == config["n_epochs"]
        val_loss = p["val_loss"]
        if val_loss is None:
            raise AssertionError("val_loss must not be None during MLP training")
        val_losses.append(val_loss)

    # Verify model learned (loss decreased from first epoch)
    loss_initial = val_losses[0]
    loss_final = min(val_losses)
    assert loss_final < loss_initial, (
        f"Best loss {loss_final} should be below first epoch {loss_initial}"
    )
