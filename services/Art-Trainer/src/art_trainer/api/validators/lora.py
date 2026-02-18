"""Validators for LoRA training API requests.

This module provides validation and decoding for LoRA API requests.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_bool,
    require_float,
    require_int,
    require_str,
)

from art_trainer.api.schemas.lora import LoraTrainRequest


def _narrow_base_model(raw: str) -> Literal["sd15", "sdxl", "flux"]:
    """Narrow base_model string to Literal type.

    Args:
        raw: Raw base_model string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not valid.
    """
    if raw == "sd15":
        return "sd15"
    if raw == "sdxl":
        return "sdxl"
    if raw == "flux":
        return "flux"
    raise JSONTypeError(f"Field 'base_model' must be 'sd15', 'sdxl', or 'flux', got '{raw}'")


def _narrow_training_type(raw: str) -> Literal["style", "character", "concept"]:
    """Narrow training_type string to Literal type.

    Args:
        raw: Raw training_type string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not valid.
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


def decode_lora_train_request(obj: JSONObject) -> LoraTrainRequest:
    """Decode and validate a LoRA training request.

    Args:
        obj: JSON object from request body.

    Returns:
        Validated LoraTrainRequest.

    Raises:
        JSONTypeError: If validation fails.
    """
    user_id = require_int(obj, "user_id")
    base_model = _narrow_base_model(require_str(obj, "base_model"))
    training_type = _narrow_training_type(require_str(obj, "training_type"))
    dataset_file_id = require_str(obj, "dataset_file_id")
    steps = require_int(obj, "steps")
    learning_rate = require_float(obj, "learning_rate")
    network_rank = require_int(obj, "network_rank")
    network_alpha = require_int(obj, "network_alpha")
    resolution = require_int(obj, "resolution")
    batch_size = require_int(obj, "batch_size")
    seed = require_int(obj, "seed")
    caption_extension = require_str(obj, "caption_extension")
    shuffle_caption = require_bool(obj, "shuffle_caption")
    keep_tokens = require_int(obj, "keep_tokens")

    return {
        "user_id": user_id,
        "base_model": base_model,
        "training_type": training_type,
        "dataset_file_id": dataset_file_id,
        "steps": steps,
        "learning_rate": learning_rate,
        "network_rank": network_rank,
        "network_alpha": network_alpha,
        "resolution": resolution,
        "batch_size": batch_size,
        "seed": seed,
        "caption_extension": caption_extension,
        "shuffle_caption": shuffle_caption,
        "keep_tokens": keep_tokens,
    }


__all__ = [
    "decode_lora_train_request",
]
