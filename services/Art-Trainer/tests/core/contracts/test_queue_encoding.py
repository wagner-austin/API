"""Tests for queue payload encoding/decoding."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from art_trainer.core.contracts.queue import LoraTrainPayload
from art_trainer.core.contracts.queue_encoding import (
    decode_lora_train_payload,
    encode_lora_train_payload,
)


def test_encode_lora_train_payload() -> None:
    """Test encoding LoraTrainPayload to JSON."""
    payload: LoraTrainPayload = {
        "job_id": "test-job-001",
        "user_id": 123,
        "base_model": "sdxl",
        "training_type": "character",
        "dataset_file_id": "file-abc-123",
        "steps": 2000,
        "learning_rate": 0.0001,
        "network_rank": 32,
        "network_alpha": 16,
        "resolution": 1024,
        "batch_size": 2,
        "seed": 12345,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 2,
    }
    encoded = encode_lora_train_payload(payload)
    assert encoded["job_id"] == "test-job-001"
    assert encoded["user_id"] == 123
    assert encoded["base_model"] == "sdxl"
    assert encoded["training_type"] == "character"
    assert encoded["dataset_file_id"] == "file-abc-123"
    assert encoded["steps"] == 2000


def test_decode_lora_train_payload() -> None:
    """Test decoding JSON to LoraTrainPayload."""
    obj: JSONObject = {
        "job_id": "test-job-002",
        "user_id": 456,
        "base_model": "flux",
        "training_type": "concept",
        "dataset_file_id": "file-xyz-789",
        "steps": 3000,
        "learning_rate": 0.00005,
        "network_rank": 64,
        "network_alpha": 32,
        "resolution": 1024,
        "batch_size": 1,
        "seed": 99999,
        "caption_extension": ".caption",
        "shuffle_caption": False,
        "keep_tokens": 0,
    }
    decoded = decode_lora_train_payload(obj)
    assert decoded["job_id"] == "test-job-002"
    assert decoded["user_id"] == 456
    assert decoded["base_model"] == "flux"
    assert decoded["training_type"] == "concept"
    assert decoded["dataset_file_id"] == "file-xyz-789"
    assert decoded["shuffle_caption"] is False


def test_decode_lora_train_payload_sdxl_character() -> None:
    """Test decoding JSON with sdxl and character values."""
    obj: JSONObject = {
        "job_id": "test-job-sdxl",
        "user_id": 789,
        "base_model": "sdxl",
        "training_type": "character",
        "dataset_file_id": "file-sdxl-001",
        "steps": 2000,
        "learning_rate": 0.0001,
        "network_rank": 32,
        "network_alpha": 16,
        "resolution": 1024,
        "batch_size": 2,
        "seed": 67890,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 2,
    }
    decoded = decode_lora_train_payload(obj)
    assert decoded["job_id"] == "test-job-sdxl"
    assert decoded["user_id"] == 789
    assert decoded["base_model"] == "sdxl"
    assert decoded["training_type"] == "character"
    assert decoded["dataset_file_id"] == "file-sdxl-001"


def test_decode_lora_train_payload_invalid_base_model() -> None:
    """Test that invalid base_model raises JSONTypeError."""
    obj: JSONObject = {
        "job_id": "test-job-003",
        "user_id": 1,
        "base_model": "invalid_model",
        "training_type": "style",
        "dataset_file_id": "file-123",
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
    with pytest.raises(JSONTypeError, match="base_model"):
        decode_lora_train_payload(obj)


def test_decode_lora_train_payload_invalid_training_type() -> None:
    """Test that invalid training_type raises JSONTypeError."""
    obj: JSONObject = {
        "job_id": "test-job-004",
        "user_id": 1,
        "base_model": "sd15",
        "training_type": "invalid_type",
        "dataset_file_id": "file-123",
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
    with pytest.raises(JSONTypeError, match="training_type"):
        decode_lora_train_payload(obj)


def test_roundtrip_encode_decode() -> None:
    """Test that encoding then decoding preserves data."""
    original: LoraTrainPayload = {
        "job_id": "roundtrip-payload",
        "user_id": 789,
        "base_model": "sd15",
        "training_type": "style",
        "dataset_file_id": "file-roundtrip",
        "steps": 1500,
        "learning_rate": 0.0002,
        "network_rank": 8,
        "network_alpha": 8,
        "resolution": 768,
        "batch_size": 4,
        "seed": 54321,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
    }
    encoded = encode_lora_train_payload(original)
    decoded = decode_lora_train_payload(encoded)
    assert decoded == original
