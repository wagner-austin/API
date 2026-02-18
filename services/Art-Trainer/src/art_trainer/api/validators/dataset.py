"""Validators for dataset API requests.

This module provides decode functions for dataset request validation.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_bool,
    require_str,
)

from art_trainer.api.schemas.dataset import DatasetCaptionRequest, DatasetUploadRequest


def _narrow_training_type(raw: str) -> Literal["style", "character", "concept"]:
    """Narrow training_type string to Literal type with validation.

    Args:
        raw: Raw training_type string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not a valid training type.
    """
    if raw == "style":
        return "style"
    if raw == "character":
        return "character"
    if raw == "concept":
        return "concept"
    raise JSONTypeError(
        f"Field 'training_type' must be 'style', 'character', or 'concept', got '{raw}'"
    )


def decode_dataset_upload_request(obj: JSONObject) -> DatasetUploadRequest:
    """Decode and validate a dataset upload request.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated DatasetUploadRequest.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    trigger_word = require_str(obj, "trigger_word")
    training_type = _narrow_training_type(require_str(obj, "training_type"))
    auto_caption = require_bool(obj, "auto_caption")

    return {
        "trigger_word": trigger_word,
        "training_type": training_type,
        "auto_caption": auto_caption,
    }


def _narrow_caption_backend(raw: str) -> Literal["blip", "gemini", "openai"]:
    """Narrow backend string to Literal type with validation.

    Args:
        raw: Raw backend string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not a valid backend.
    """
    if raw == "blip":
        return "blip"
    if raw == "gemini":
        return "gemini"
    if raw == "openai":
        return "openai"
    raise JSONTypeError(f"Field 'backend' must be 'blip', 'gemini', or 'openai', got '{raw}'")


def decode_dataset_caption_request(obj: JSONObject) -> DatasetCaptionRequest:
    """Decode and validate a dataset caption request.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated DatasetCaptionRequest.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    trigger_word = require_str(obj, "trigger_word")
    backend = _narrow_caption_backend(require_str(obj, "backend"))
    model_name = require_str(obj, "model_name")

    return {
        "trigger_word": trigger_word,
        "backend": backend,
        "model_name": model_name,
    }


__all__ = [
    "decode_dataset_caption_request",
    "decode_dataset_upload_request",
]
