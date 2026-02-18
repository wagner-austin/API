"""Tests for LoRA training validators."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from art_trainer.api.validators.lora import decode_lora_train_request


def _make_valid_request() -> JSONObject:
    """Create a valid request object.

    Returns:
        Valid LoraTrainRequest as JSONObject.
    """
    return {
        "user_id": 123,
        "base_model": "sd15",
        "training_type": "style",
        "dataset_file_id": "file-abc-123",
        "steps": 1000,
        "learning_rate": 0.0001,
        "network_rank": 16,
        "network_alpha": 16,
        "resolution": 512,
        "batch_size": 1,
        "seed": 42,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
    }


def test_decode_lora_train_request_valid() -> None:
    """Test decoding valid request."""
    obj = _make_valid_request()
    result = decode_lora_train_request(obj)

    assert result["user_id"] == 123
    assert result["base_model"] == "sd15"
    assert result["training_type"] == "style"
    assert result["dataset_file_id"] == "file-abc-123"


def test_decode_lora_train_request_invalid_base_model() -> None:
    """Test decoding request with invalid base_model raises JSONTypeError."""
    obj = _make_valid_request()
    obj["base_model"] = "invalid_model"

    with pytest.raises(JSONTypeError, match="base_model"):
        decode_lora_train_request(obj)


def test_decode_lora_train_request_invalid_training_type() -> None:
    """Test decoding request with invalid training_type raises JSONTypeError."""
    obj = _make_valid_request()
    obj["training_type"] = "invalid_type"

    with pytest.raises(JSONTypeError, match="training_type"):
        decode_lora_train_request(obj)


def test_decode_lora_train_request_all_base_models() -> None:
    """Test decoding with all valid base_model values."""
    for model in ["sd15", "sdxl", "flux"]:
        obj = _make_valid_request()
        obj["base_model"] = model
        result = decode_lora_train_request(obj)
        assert result["base_model"] == model


def test_decode_lora_train_request_all_training_types() -> None:
    """Test decoding with all valid training_type values."""
    for training_type in ["style", "character", "concept"]:
        obj = _make_valid_request()
        obj["training_type"] = training_type
        result = decode_lora_train_request(obj)
        assert result["training_type"] == training_type


def test_decode_lora_train_request_missing_field() -> None:
    """Test decoding request with missing field raises JSONTypeError."""
    obj = _make_valid_request()
    del obj["user_id"]

    with pytest.raises(JSONTypeError, match="user_id"):
        decode_lora_train_request(obj)
