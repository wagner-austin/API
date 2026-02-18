"""Tests for training progress contracts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from art_trainer.core.contracts.progress import (
    ArtTrainingProgress,
    decode_art_training_progress,
    encode_art_training_progress,
)


def test_encode_art_training_progress() -> None:
    """Test encoding ArtTrainingProgress to JSON."""
    progress: ArtTrainingProgress = {
        "job_id": "test-123",
        "phase": "training",
        "step": 50,
        "total_steps": 100,
        "loss": 0.05,
        "learning_rate": 0.0001,
        "updated_at": "2024-01-15T10:30:00",
    }
    encoded = encode_art_training_progress(progress)
    assert encoded["job_id"] == "test-123"
    assert encoded["phase"] == "training"
    assert encoded["step"] == 50
    assert encoded["total_steps"] == 100
    assert encoded["loss"] == 0.05
    assert encoded["learning_rate"] == 0.0001
    assert encoded["updated_at"] == "2024-01-15T10:30:00"


def test_decode_art_training_progress() -> None:
    """Test decoding JSON to ArtTrainingProgress."""
    obj: JSONObject = {
        "job_id": "test-456",
        "phase": "completed",
        "step": 100,
        "total_steps": 100,
        "loss": 0.03,
        "learning_rate": 0.0001,
        "updated_at": "2024-01-15T11:00:00",
    }
    decoded = decode_art_training_progress(obj)
    assert decoded["job_id"] == "test-456"
    assert decoded["phase"] == "completed"
    assert decoded["step"] == 100
    assert decoded["total_steps"] == 100
    assert decoded["loss"] == 0.03
    assert decoded["learning_rate"] == 0.0001
    assert decoded["updated_at"] == "2024-01-15T11:00:00"


def test_decode_art_training_progress_with_null_loss() -> None:
    """Test decoding progress with null loss value."""
    obj: JSONObject = {
        "job_id": "test-789",
        "phase": "preparing",
        "step": 0,
        "total_steps": 100,
        "loss": None,
        "learning_rate": 0.0001,
        "updated_at": "2024-01-15T10:00:00",
    }
    decoded = decode_art_training_progress(obj)
    assert decoded["loss"] is None


def test_decode_art_training_progress_invalid_phase() -> None:
    """Test that invalid phase raises JSONTypeError."""
    obj: JSONObject = {
        "job_id": "test-000",
        "phase": "invalid_phase",
        "step": 0,
        "total_steps": 100,
        "loss": None,
        "learning_rate": 0.0001,
        "updated_at": "2024-01-15T10:00:00",
    }
    with pytest.raises(JSONTypeError, match="phase"):
        decode_art_training_progress(obj)


def test_roundtrip_encode_decode() -> None:
    """Test that encoding then decoding preserves data."""
    original: ArtTrainingProgress = {
        "job_id": "roundtrip-test",
        "phase": "training",
        "step": 75,
        "total_steps": 150,
        "loss": 0.042,
        "learning_rate": 0.00005,
        "updated_at": "2024-01-15T12:00:00",
    }
    encoded = encode_art_training_progress(original)
    decoded = decode_art_training_progress(encoded)
    assert decoded == original


def test_decode_all_valid_phases() -> None:
    """Test decoding with all valid phase values."""
    phases = [
        "queued",
        "preparing",
        "training",
        "saving",
        "uploading",
        "completed",
        "failed",
        "cancelled",
    ]
    for phase in phases:
        obj: JSONObject = {
            "job_id": f"test-{phase}",
            "phase": phase,
            "step": 0,
            "total_steps": 100,
            "loss": None,
            "learning_rate": 0.0001,
            "updated_at": "2024-01-15T10:00:00",
        }
        decoded = decode_art_training_progress(obj)
        assert decoded["phase"] == phase
