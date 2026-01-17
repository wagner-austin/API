"""Tests for progress store."""

from __future__ import annotations

from platform_workers.testing import FakeRedis

from model_trainer.core.contracts.progress import TrainingProgress
from model_trainer.worker.progress_store import ProgressStore


def test_progress_store_save_and_load() -> None:
    """Test saving and loading training progress."""
    fake = FakeRedis()
    store = ProgressStore(fake)

    progress: TrainingProgress = {
        "run_id": "run-store-test",
        "phase": "training",
        "epoch": 3,
        "total_epochs": 10,
        "step": 150,
        "total_steps": 500,
        "train_loss": 1.25,
        "train_ppl": 3.5,
        "grad_norm": 0.15,
        "samples_per_sec": 48.0,
        "val_loss": 1.1,
        "val_ppl": 3.0,
        "updated_at": "2024-01-15T14:30:00",
    }

    store.save(progress)
    loaded = store.load("run-store-test")

    assert loaded is not None and loaded["phase"] == "training"
    assert loaded["run_id"] == "run-store-test"
    assert loaded["phase"] == "training"
    assert loaded["epoch"] == 3
    assert loaded["total_epochs"] == 10
    assert loaded["step"] == 150
    assert loaded["total_steps"] == 500
    assert loaded["train_loss"] == 1.25
    assert loaded["train_ppl"] == 3.5
    assert loaded["grad_norm"] == 0.15
    assert loaded["samples_per_sec"] == 48.0
    assert loaded["val_loss"] == 1.1
    assert loaded["val_ppl"] == 3.0
    assert loaded["updated_at"] == "2024-01-15T14:30:00"
    fake.assert_only_called({"set", "expire", "get"})


def test_progress_store_load_missing() -> None:
    """Test loading non-existent progress returns None."""
    fake = FakeRedis()
    store = ProgressStore(fake)

    loaded = store.load("run-nonexistent")
    assert loaded is None
    fake.assert_only_called({"get"})


def test_progress_store_load_non_dict() -> None:
    """Test loading non-dict JSON returns None."""
    fake = FakeRedis()
    store = ProgressStore(fake)

    # Set array JSON in Redis
    fake.set("runs:progress:run-array", "[1, 2, 3]")
    loaded = store.load("run-array")
    assert loaded is None
    fake.assert_only_called({"set", "get"})
