"""Tests for dataset validators."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from art_trainer.api.validators.dataset import (
    decode_dataset_caption_request,
    decode_dataset_upload_request,
)


def test_decode_dataset_upload_request_style() -> None:
    """Test decode_dataset_upload_request with style training type."""
    obj: JSONObject = {
        "trigger_word": "sks person",
        "training_type": "style",
        "auto_caption": True,
    }

    result = decode_dataset_upload_request(obj)

    assert result["trigger_word"] == "sks person"
    assert result["training_type"] == "style"
    assert result["auto_caption"] is True


def test_decode_dataset_upload_request_character() -> None:
    """Test decode_dataset_upload_request with character training type."""
    obj: JSONObject = {
        "trigger_word": "john doe",
        "training_type": "character",
        "auto_caption": False,
    }

    result = decode_dataset_upload_request(obj)

    assert result["trigger_word"] == "john doe"
    assert result["training_type"] == "character"
    assert result["auto_caption"] is False


def test_decode_dataset_upload_request_concept() -> None:
    """Test decode_dataset_upload_request with concept training type."""
    obj: JSONObject = {
        "trigger_word": "anime style",
        "training_type": "concept",
        "auto_caption": True,
    }

    result = decode_dataset_upload_request(obj)

    assert result["trigger_word"] == "anime style"
    assert result["training_type"] == "concept"
    assert result["auto_caption"] is True


def test_decode_dataset_upload_request_invalid_training_type() -> None:
    """Test decode_dataset_upload_request with invalid training type."""
    obj: JSONObject = {
        "trigger_word": "sks person",
        "training_type": "invalid",
        "auto_caption": True,
    }

    with pytest.raises(JSONTypeError) as exc_info:
        decode_dataset_upload_request(obj)

    assert "must be 'style', 'character', or 'concept'" in str(exc_info.value)


def test_decode_dataset_upload_request_missing_trigger_word() -> None:
    """Test decode_dataset_upload_request with missing trigger_word."""
    obj: JSONObject = {
        "training_type": "style",
        "auto_caption": True,
    }

    with pytest.raises(JSONTypeError):
        decode_dataset_upload_request(obj)


def test_decode_dataset_upload_request_missing_training_type() -> None:
    """Test decode_dataset_upload_request with missing training_type."""
    obj: JSONObject = {
        "trigger_word": "sks person",
        "auto_caption": True,
    }

    with pytest.raises(JSONTypeError):
        decode_dataset_upload_request(obj)


def test_decode_dataset_upload_request_missing_auto_caption() -> None:
    """Test decode_dataset_upload_request with missing auto_caption."""
    obj: JSONObject = {
        "trigger_word": "sks person",
        "training_type": "style",
    }

    with pytest.raises(JSONTypeError):
        decode_dataset_upload_request(obj)


def test_decode_dataset_caption_request_blip() -> None:
    """Test decode_dataset_caption_request with blip backend."""
    obj: JSONObject = {
        "trigger_word": "sks person",
        "backend": "blip",
        "model_name": "Salesforce/blip-image-captioning-base",
    }

    result = decode_dataset_caption_request(obj)

    assert result["trigger_word"] == "sks person"
    assert result["backend"] == "blip"
    assert result["model_name"] == "Salesforce/blip-image-captioning-base"


def test_decode_dataset_caption_request_gemini() -> None:
    """Test decode_dataset_caption_request with gemini backend."""
    obj: JSONObject = {
        "trigger_word": "anime style",
        "backend": "gemini",
        "model_name": "gemini-2.0-flash",
    }

    result = decode_dataset_caption_request(obj)

    assert result["trigger_word"] == "anime style"
    assert result["backend"] == "gemini"
    assert result["model_name"] == "gemini-2.0-flash"


def test_decode_dataset_caption_request_openai() -> None:
    """Test decode_dataset_caption_request with openai backend."""
    obj: JSONObject = {
        "trigger_word": "portrait",
        "backend": "openai",
        "model_name": "gpt-4o",
    }

    result = decode_dataset_caption_request(obj)

    assert result["trigger_word"] == "portrait"
    assert result["backend"] == "openai"
    assert result["model_name"] == "gpt-4o"


def test_decode_dataset_caption_request_invalid_backend() -> None:
    """Test decode_dataset_caption_request with invalid backend."""
    obj: JSONObject = {
        "trigger_word": "sks person",
        "backend": "invalid",
        "model_name": "some-model",
    }

    with pytest.raises(JSONTypeError) as exc_info:
        decode_dataset_caption_request(obj)

    assert "must be 'blip', 'gemini', or 'openai'" in str(exc_info.value)


def test_decode_dataset_caption_request_missing_trigger_word() -> None:
    """Test decode_dataset_caption_request with missing trigger_word."""
    obj: JSONObject = {
        "backend": "gemini",
        "model_name": "gemini-2.0-flash",
    }

    with pytest.raises(JSONTypeError):
        decode_dataset_caption_request(obj)


def test_decode_dataset_caption_request_missing_backend() -> None:
    """Test decode_dataset_caption_request with missing backend."""
    obj: JSONObject = {
        "trigger_word": "sks person",
        "model_name": "gemini-2.0-flash",
    }

    with pytest.raises(JSONTypeError):
        decode_dataset_caption_request(obj)


def test_decode_dataset_caption_request_missing_model_name() -> None:
    """Test decode_dataset_caption_request with missing model_name."""
    obj: JSONObject = {
        "trigger_word": "sks person",
        "backend": "gemini",
    }

    with pytest.raises(JSONTypeError):
        decode_dataset_caption_request(obj)
