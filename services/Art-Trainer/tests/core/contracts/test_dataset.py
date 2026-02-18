"""Tests for dataset contracts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from art_trainer.core.contracts.dataset import (
    CaptionResult,
    DatasetInfo,
    DatasetUploadResult,
    _narrow_training_type,
    decode_caption_result,
    decode_dataset_info,
    decode_dataset_upload_result,
    encode_caption_result,
    encode_dataset_info,
    encode_dataset_upload_result,
)


def test_encode_dataset_info() -> None:
    """Test encode_dataset_info encodes correctly."""
    info: DatasetInfo = {
        "dataset_id": "ds-123",
        "image_count": 10,
        "dataset_path": "/data/datasets/ds-123",
    }

    result = encode_dataset_info(info)

    assert result["dataset_id"] == "ds-123"
    assert result["image_count"] == 10
    assert result["dataset_path"] == "/data/datasets/ds-123"


def test_decode_dataset_info() -> None:
    """Test decode_dataset_info decodes correctly."""
    obj: JSONObject = {
        "dataset_id": "ds-456",
        "image_count": 20,
        "dataset_path": "/data/datasets/ds-456",
    }

    result = decode_dataset_info(obj)

    assert result["dataset_id"] == "ds-456"
    assert result["image_count"] == 20
    assert result["dataset_path"] == "/data/datasets/ds-456"


def test_encode_caption_result() -> None:
    """Test encode_caption_result encodes correctly."""
    result: CaptionResult = {
        "image_name": "photo.jpg",
        "caption": "sks person in a park",
        "caption_path": "/data/datasets/ds-123/photo.txt",
    }

    encoded = encode_caption_result(result)

    assert encoded["image_name"] == "photo.jpg"
    assert encoded["caption"] == "sks person in a park"
    assert encoded["caption_path"] == "/data/datasets/ds-123/photo.txt"


def test_decode_caption_result() -> None:
    """Test decode_caption_result decodes correctly."""
    obj: JSONObject = {
        "image_name": "image.png",
        "caption": "sks person smiling",
        "caption_path": "/data/datasets/ds-123/image.txt",
    }

    result = decode_caption_result(obj)

    assert result["image_name"] == "image.png"
    assert result["caption"] == "sks person smiling"
    assert result["caption_path"] == "/data/datasets/ds-123/image.txt"


def test_encode_dataset_upload_result() -> None:
    """Test encode_dataset_upload_result encodes correctly."""
    result: DatasetUploadResult = {
        "dataset_id": "ds-789",
        "image_count": 15,
        "caption_count": 15,
        "dataset_path": "/data/datasets/ds-789",
    }

    encoded = encode_dataset_upload_result(result)

    assert encoded["dataset_id"] == "ds-789"
    assert encoded["image_count"] == 15
    assert encoded["caption_count"] == 15
    assert encoded["dataset_path"] == "/data/datasets/ds-789"


def test_decode_dataset_upload_result() -> None:
    """Test decode_dataset_upload_result decodes correctly."""
    obj: JSONObject = {
        "dataset_id": "ds-abc",
        "image_count": 25,
        "caption_count": 20,
        "dataset_path": "/data/datasets/ds-abc",
    }

    result = decode_dataset_upload_result(obj)

    assert result["dataset_id"] == "ds-abc"
    assert result["image_count"] == 25
    assert result["caption_count"] == 20
    assert result["dataset_path"] == "/data/datasets/ds-abc"


def test_narrow_training_type_style() -> None:
    """Test _narrow_training_type accepts style."""
    result = _narrow_training_type("style")
    assert result == "style"


def test_narrow_training_type_character() -> None:
    """Test _narrow_training_type accepts character."""
    result = _narrow_training_type("character")
    assert result == "character"


def test_narrow_training_type_concept() -> None:
    """Test _narrow_training_type accepts concept."""
    result = _narrow_training_type("concept")
    assert result == "concept"


def test_narrow_training_type_invalid() -> None:
    """Test _narrow_training_type rejects invalid values."""
    with pytest.raises(JSONTypeError) as exc_info:
        _narrow_training_type("invalid")

    assert "must be 'style', 'character', or 'concept'" in str(exc_info.value)
