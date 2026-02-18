"""Dataset contracts for Art-Trainer.

This module defines TypedDicts for dataset upload and management.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_int,
    require_str,
)


class DatasetInfo(TypedDict, total=True):
    """Information about an uploaded dataset.

    Attributes:
        dataset_id: Unique identifier for the dataset.
        image_count: Number of images in the dataset.
        dataset_path: Path to the dataset directory.
    """

    dataset_id: str
    image_count: int
    dataset_path: str


class CaptionResult(TypedDict, total=True):
    """Result of captioning an image.

    Attributes:
        image_name: Name of the image file.
        caption: Generated caption text.
        caption_path: Path to the caption file.
    """

    image_name: str
    caption: str
    caption_path: str


class DatasetUploadResult(TypedDict, total=True):
    """Result of dataset upload and captioning.

    Attributes:
        dataset_id: Unique identifier for the dataset.
        image_count: Number of images uploaded.
        caption_count: Number of captions generated.
        dataset_path: Path to the dataset directory.
    """

    dataset_id: str
    image_count: int
    caption_count: int
    dataset_path: str


def encode_dataset_info(info: DatasetInfo) -> JSONObject:
    """Encode DatasetInfo to JSON.

    Args:
        info: Dataset info to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "dataset_id": info["dataset_id"],
        "image_count": info["image_count"],
        "dataset_path": info["dataset_path"],
    }


def decode_dataset_info(obj: JSONObject) -> DatasetInfo:
    """Decode JSON to DatasetInfo.

    Args:
        obj: JSON object to decode.

    Returns:
        Decoded DatasetInfo.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    dataset_id = require_str(obj, "dataset_id")
    image_count = require_int(obj, "image_count")
    dataset_path = require_str(obj, "dataset_path")

    return {
        "dataset_id": dataset_id,
        "image_count": image_count,
        "dataset_path": dataset_path,
    }


def encode_caption_result(result: CaptionResult) -> JSONObject:
    """Encode CaptionResult to JSON.

    Args:
        result: Caption result to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "image_name": result["image_name"],
        "caption": result["caption"],
        "caption_path": result["caption_path"],
    }


def decode_caption_result(obj: JSONObject) -> CaptionResult:
    """Decode JSON to CaptionResult.

    Args:
        obj: JSON object to decode.

    Returns:
        Decoded CaptionResult.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    image_name = require_str(obj, "image_name")
    caption = require_str(obj, "caption")
    caption_path = require_str(obj, "caption_path")

    return {
        "image_name": image_name,
        "caption": caption,
        "caption_path": caption_path,
    }


def encode_dataset_upload_result(result: DatasetUploadResult) -> JSONObject:
    """Encode DatasetUploadResult to JSON.

    Args:
        result: Upload result to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "dataset_id": result["dataset_id"],
        "image_count": result["image_count"],
        "caption_count": result["caption_count"],
        "dataset_path": result["dataset_path"],
    }


def decode_dataset_upload_result(obj: JSONObject) -> DatasetUploadResult:
    """Decode JSON to DatasetUploadResult.

    Args:
        obj: JSON object to decode.

    Returns:
        Decoded DatasetUploadResult.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    dataset_id = require_str(obj, "dataset_id")
    image_count = require_int(obj, "image_count")
    caption_count = require_int(obj, "caption_count")
    dataset_path = require_str(obj, "dataset_path")

    return {
        "dataset_id": dataset_id,
        "image_count": image_count,
        "caption_count": caption_count,
        "dataset_path": dataset_path,
    }


def _narrow_training_type(raw: str) -> str:
    """Validate training type for captioning context.

    Args:
        raw: Raw training type string.

    Returns:
        Validated training type.

    Raises:
        JSONTypeError: If value is not a valid training type.
    """
    if raw in ("style", "character", "concept"):
        return raw
    raise JSONTypeError(
        f"Field 'training_type' must be 'style', 'character', or 'concept', got '{raw}'"
    )


__all__ = [
    "CaptionResult",
    "DatasetInfo",
    "DatasetUploadResult",
    "decode_caption_result",
    "decode_dataset_info",
    "decode_dataset_upload_result",
    "encode_caption_result",
    "encode_dataset_info",
    "encode_dataset_upload_result",
]
