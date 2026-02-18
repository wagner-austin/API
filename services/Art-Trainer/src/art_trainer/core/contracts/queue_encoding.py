"""Encoding and decoding functions for queue payloads.

This module provides type-safe serialization and deserialization of
LoraTrainPayload for RQ job queue communication.

All encode functions convert TypedDicts to JSONObject (dict[str, JSONValue]).
All decode functions validate and convert JSONObject back to TypedDicts using
require_* helpers from platform_core.json_utils.
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

from .queue import LoraTrainPayload


def _narrow_base_model(raw: str) -> Literal["sd15", "sdxl", "flux"]:
    """Narrow base_model string to Literal type with validation.

    Args:
        raw: Raw base_model string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not a valid base model.
    """
    if raw == "sd15":
        return "sd15"
    if raw == "sdxl":
        return "sdxl"
    if raw == "flux":
        return "flux"
    raise JSONTypeError(f"Field 'base_model' must be 'sd15', 'sdxl', or 'flux', got '{raw}'")


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


def encode_lora_train_payload(payload: LoraTrainPayload) -> JSONObject:
    """Encode LoraTrainPayload TypedDict to JSONObject for RQ serialization.

    Args:
        payload: LoRA training payload to encode.

    Returns:
        JSON-serializable dictionary suitable for RQ job queue.
    """
    return {
        "job_id": payload["job_id"],
        "user_id": payload["user_id"],
        "base_model": payload["base_model"],
        "training_type": payload["training_type"],
        "dataset_file_id": payload["dataset_file_id"],
        "steps": payload["steps"],
        "learning_rate": payload["learning_rate"],
        "network_rank": payload["network_rank"],
        "network_alpha": payload["network_alpha"],
        "resolution": payload["resolution"],
        "batch_size": payload["batch_size"],
        "seed": payload["seed"],
        "caption_extension": payload["caption_extension"],
        "shuffle_caption": payload["shuffle_caption"],
        "keep_tokens": payload["keep_tokens"],
    }


def decode_lora_train_payload(obj: JSONObject) -> LoraTrainPayload:
    """Decode JSONObject to LoraTrainPayload with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated LoraTrainPayload TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    job_id = require_str(obj, "job_id")
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
        "job_id": job_id,
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
    "decode_lora_train_payload",
    "encode_lora_train_payload",
]
