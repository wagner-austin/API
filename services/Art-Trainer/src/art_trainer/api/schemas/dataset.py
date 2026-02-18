"""API schemas for dataset endpoints.

This module defines TypedDicts for dataset API request/response types.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import JSONObject


class DatasetUploadRequest(TypedDict, total=True):
    """Request to upload a dataset.

    Attributes:
        trigger_word: Trigger word for captions (e.g., "sks person").
        training_type: Type of training (style, character, concept).
        auto_caption: Whether to auto-generate captions with BLIP.
    """

    trigger_word: str
    training_type: Literal["style", "character", "concept"]
    auto_caption: bool


class DatasetUploadResponse(TypedDict, total=True):
    """Response from dataset upload.

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


def encode_dataset_upload_response(response: DatasetUploadResponse) -> JSONObject:
    """Encode DatasetUploadResponse to JSON.

    Args:
        response: Response to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "dataset_id": response["dataset_id"],
        "image_count": response["image_count"],
        "caption_count": response["caption_count"],
        "dataset_path": response["dataset_path"],
    }


class DatasetCaptionRequest(TypedDict, total=True):
    """Request to caption a dataset.

    Attributes:
        trigger_word: Trigger word to prepend to captions (e.g., "sks person").
        backend: Caption backend to use (blip, gemini, or openai).
        model_name: Model name for the backend. Defaults vary by backend.
    """

    trigger_word: str
    backend: Literal["blip", "gemini", "openai"]
    model_name: str


class DatasetCaptionResponse(TypedDict, total=True):
    """Response from dataset captioning.

    Attributes:
        dataset_id: Dataset identifier.
        captioned_count: Number of images captioned.
        skipped_count: Number of images skipped (already had captions).
        backend: Caption backend used.
    """

    dataset_id: str
    captioned_count: int
    skipped_count: int
    backend: Literal["blip", "gemini", "openai"]


def encode_dataset_caption_response(response: DatasetCaptionResponse) -> JSONObject:
    """Encode DatasetCaptionResponse to JSON.

    Args:
        response: Response to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "dataset_id": response["dataset_id"],
        "captioned_count": response["captioned_count"],
        "skipped_count": response["skipped_count"],
        "backend": response["backend"],
    }


__all__ = [
    "DatasetCaptionRequest",
    "DatasetCaptionResponse",
    "DatasetUploadRequest",
    "DatasetUploadResponse",
    "encode_dataset_caption_response",
    "encode_dataset_upload_response",
]
