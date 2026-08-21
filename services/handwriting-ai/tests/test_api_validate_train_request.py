"""Train-request payload validation."""

from __future__ import annotations

from typing import Protocol

import pytest
from platform_core.errors import AppError

from handwriting_ai.api.routes.training import (
    _validate_train_request,
)
from handwriting_ai.api.types import JsonDict


class _RedisConnectionProto(Protocol):
    """Protocol for Redis connection used by RQ."""

    pass


# --- Tests for dependencies.py ---


def test_validate_train_request_valid_payload() -> None:
    """Test _validate_train_request with valid payload."""
    payload: JsonDict = {
        "user_id": 123,
        "model_id": "test-model",
        "epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "seed": 42,
        "augment": True,
        "notes": "Test notes",
    }
    result = _validate_train_request(payload)
    assert result["user_id"] == 123
    assert result["model_id"] == "test-model"
    assert result["epochs"] == 10
    assert result["batch_size"] == 32
    assert result["lr"] == 0.001
    assert result["seed"] == 42
    assert result["augment"] is True
    assert result["notes"] == "Test notes"


def test_validate_train_request_notes_null() -> None:
    """Test _validate_train_request with null notes."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
        "augment": False,
        "notes": None,
    }
    result = _validate_train_request(payload)
    assert result["notes"] is None


def test_validate_train_request_notes_missing() -> None:
    """Test _validate_train_request with missing notes (defaults to None)."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    result = _validate_train_request(payload)
    assert result["notes"] is None


def test_validate_train_request_augment_default() -> None:
    """Test _validate_train_request with missing augment (defaults to False)."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    result = _validate_train_request(payload)
    assert result["augment"] is False


def test_validate_train_request_lr_as_int() -> None:
    """Test _validate_train_request accepts lr as int."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 1,
        "seed": 1,
    }
    result = _validate_train_request(payload)
    assert result["lr"] == 1.0
    assert type(result["lr"]) is float


def test_validate_train_request_invalid_user_id_type() -> None:
    """Test _validate_train_request rejects non-int user_id."""
    payload: JsonDict = {
        "user_id": "not-an-int",
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="user_id must be an integer"):
        _validate_train_request(payload)


def test_validate_train_request_user_id_bool_rejected() -> None:
    """Test _validate_train_request rejects bool user_id."""
    payload: JsonDict = {
        "user_id": True,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="user_id must be an integer"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_model_id() -> None:
    """Test _validate_train_request rejects empty model_id."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="model_id must be a non-empty string"):
        _validate_train_request(payload)


def test_validate_train_request_model_id_whitespace() -> None:
    """Test _validate_train_request rejects whitespace-only model_id."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "   ",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="model_id must be a non-empty string"):
        _validate_train_request(payload)


def test_validate_train_request_model_id_not_string() -> None:
    """Test _validate_train_request rejects non-string model_id."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": 123,
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="model_id must be a non-empty string"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_epochs() -> None:
    """Test _validate_train_request rejects non-positive epochs."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 0,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="epochs must be a positive integer"):
        _validate_train_request(payload)


def test_validate_train_request_epochs_bool_rejected() -> None:
    """Test _validate_train_request rejects bool epochs."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": True,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="epochs must be a positive integer"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_batch_size() -> None:
    """Test _validate_train_request rejects non-positive batch_size."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 0,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="batch_size must be a positive integer"):
        _validate_train_request(payload)


def test_validate_train_request_batch_size_bool_rejected() -> None:
    """Test _validate_train_request rejects bool batch_size."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": True,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="batch_size must be a positive integer"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_lr_zero() -> None:
    """Test _validate_train_request rejects zero lr."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0,
        "seed": 1,
    }
    with pytest.raises(AppError, match="lr must be a positive number"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_lr_negative() -> None:
    """Test _validate_train_request rejects negative lr."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": -0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="lr must be a positive number"):
        _validate_train_request(payload)


def test_validate_train_request_lr_bool_rejected() -> None:
    """Test _validate_train_request rejects bool lr."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": True,
        "seed": 1,
    }
    with pytest.raises(AppError, match="lr must be a positive number"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_seed() -> None:
    """Test _validate_train_request rejects non-int seed."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": "not-an-int",
    }
    with pytest.raises(AppError, match="seed must be an integer"):
        _validate_train_request(payload)


def test_validate_train_request_seed_bool_rejected() -> None:
    """Test _validate_train_request rejects bool seed."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": False,
    }
    with pytest.raises(AppError, match="seed must be an integer"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_augment() -> None:
    """Test _validate_train_request rejects non-bool augment."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
        "augment": "yes",
    }
    with pytest.raises(AppError, match="augment must be a boolean"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_notes_type() -> None:
    """Test _validate_train_request rejects non-string notes."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
        "notes": 123,
    }
    with pytest.raises(AppError, match="notes must be a string or null"):
        _validate_train_request(payload)


# --- Tests for training.py build_router ---
