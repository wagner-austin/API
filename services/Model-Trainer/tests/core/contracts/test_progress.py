"""Tests for training progress contracts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from model_trainer.core.contracts.progress import (
    TrainingProgress,
    decode_training_progress,
    encode_training_progress,
    initial_progress,
)


def test_encode_training_progress() -> None:
    """Test encoding TrainingProgress to JSONObject."""
    progress: TrainingProgress = {
        "run_id": "run-123",
        "phase": "training",
        "epoch": 2,
        "total_epochs": 10,
        "step": 50,
        "total_steps": 100,
        "train_loss": 1.5,
        "train_ppl": 4.5,
        "grad_norm": 0.25,
        "samples_per_sec": 32.5,
        "val_loss": 1.2,
        "val_ppl": 3.3,
        "updated_at": "2024-01-15T10:30:00",
    }
    encoded = encode_training_progress(progress)
    assert encoded["run_id"] == "run-123"
    assert encoded["phase"] == "training"
    assert encoded["epoch"] == 2
    assert encoded["total_epochs"] == 10
    assert encoded["step"] == 50
    assert encoded["total_steps"] == 100
    assert encoded["train_loss"] == 1.5
    assert encoded["train_ppl"] == 4.5
    assert encoded["grad_norm"] == 0.25
    assert encoded["samples_per_sec"] == 32.5
    assert encoded["val_loss"] == 1.2
    assert encoded["val_ppl"] == 3.3
    assert encoded["updated_at"] == "2024-01-15T10:30:00"


def test_decode_training_progress() -> None:
    """Test decoding JSONObject to TrainingProgress."""
    obj: JSONObject = {
        "run_id": "run-456",
        "phase": "validation",
        "epoch": 5,
        "total_epochs": 20,
        "step": 250,
        "total_steps": 500,
        "train_loss": 0.8,
        "train_ppl": 2.2,
        "grad_norm": 0.1,
        "samples_per_sec": 64.0,
        "val_loss": 0.9,
        "val_ppl": 2.5,
        "updated_at": "2024-01-15T12:00:00",
    }
    progress = decode_training_progress(obj)
    assert progress["run_id"] == "run-456"
    assert progress["phase"] == "validation"
    assert progress["epoch"] == 5
    assert progress["total_epochs"] == 20
    assert progress["step"] == 250
    assert progress["total_steps"] == 500
    assert progress["train_loss"] == 0.8
    assert progress["train_ppl"] == 2.2
    assert progress["grad_norm"] == 0.1
    assert progress["samples_per_sec"] == 64.0
    assert progress["val_loss"] == 0.9
    assert progress["val_ppl"] == 2.5
    assert progress["updated_at"] == "2024-01-15T12:00:00"


def test_decode_training_progress_with_null_val_metrics() -> None:
    """Test decoding TrainingProgress with null validation metrics."""
    obj: JSONObject = {
        "run_id": "run-789",
        "phase": "queued",
        "epoch": 0,
        "total_epochs": 5,
        "step": 0,
        "total_steps": 0,
        "train_loss": 0.0,
        "train_ppl": 0.0,
        "grad_norm": 0.0,
        "samples_per_sec": 0.0,
        "val_loss": None,
        "val_ppl": None,
        "updated_at": "2024-01-15T08:00:00",
    }
    progress = decode_training_progress(obj)
    assert progress["val_loss"] is None
    assert progress["val_ppl"] is None


def test_decode_training_progress_invalid_phase() -> None:
    """Test decoding TrainingProgress with invalid phase raises error."""
    obj: JSONObject = {
        "run_id": "run-bad",
        "phase": "invalid_phase",
        "epoch": 0,
        "total_epochs": 5,
        "step": 0,
        "total_steps": 0,
        "train_loss": 0.0,
        "train_ppl": 0.0,
        "grad_norm": 0.0,
        "samples_per_sec": 0.0,
        "val_loss": None,
        "val_ppl": None,
        "updated_at": "2024-01-15T08:00:00",
    }
    with pytest.raises(JSONTypeError, match="must be one of"):
        decode_training_progress(obj)


def test_initial_progress() -> None:
    """Test creating initial progress in queued state."""
    progress = initial_progress(
        run_id="run-init",
        total_epochs=10,
        updated_at="2024-01-15T09:00:00",
    )
    assert progress["run_id"] == "run-init"
    assert progress["phase"] == "queued"
    assert progress["epoch"] == 0
    assert progress["total_epochs"] == 10
    assert progress["step"] == 0
    assert progress["total_steps"] == 0
    assert progress["train_loss"] == 0.0
    assert progress["train_ppl"] == 0.0
    assert progress["grad_norm"] == 0.0
    assert progress["samples_per_sec"] == 0.0
    assert progress["val_loss"] is None
    assert progress["val_ppl"] is None
    assert progress["updated_at"] == "2024-01-15T09:00:00"


def test_decode_all_valid_phases() -> None:
    """Test decoding all valid training phases."""
    phases = [
        "queued",
        "tokenization",
        "training",
        "validation",
        "test",
        "saving",
        "uploading",
        "completed",
        "cancelled",
        "failed",
    ]
    for phase in phases:
        obj: JSONObject = {
            "run_id": f"run-{phase}",
            "phase": phase,
            "epoch": 0,
            "total_epochs": 1,
            "step": 0,
            "total_steps": 0,
            "train_loss": 0.0,
            "train_ppl": 0.0,
            "grad_norm": 0.0,
            "samples_per_sec": 0.0,
            "val_loss": None,
            "val_ppl": None,
            "updated_at": "2024-01-15T10:00:00",
        }
        progress = decode_training_progress(obj)
        assert progress["phase"] == phase
