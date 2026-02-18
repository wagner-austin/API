"""API schemas for LoRA training endpoints.

This module defines the TypedDicts for LoRA training API requests and responses.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class LoraTrainRequest(TypedDict, total=True):
    """Request to start LoRA training.

    Attributes:
        user_id: User initiating the training.
        base_model: Base model type (sd15, sdxl, or flux).
        training_type: Type of training (style, character, or concept).
        dataset_file_id: File ID for the dataset in data-bank.
        steps: Number of training steps.
        learning_rate: Learning rate for training.
        network_rank: LoRA network rank.
        network_alpha: LoRA network alpha.
        resolution: Training image resolution.
        batch_size: Training batch size.
        seed: Random seed for reproducibility.
        caption_extension: File extension for captions.
        shuffle_caption: Whether to shuffle caption tokens.
        keep_tokens: Number of tokens to keep unshuffled.
    """

    user_id: int
    base_model: Literal["sd15", "sdxl", "flux"]
    training_type: Literal["style", "character", "concept"]
    dataset_file_id: str
    steps: int
    learning_rate: float
    network_rank: int
    network_alpha: int
    resolution: int
    batch_size: int
    seed: int
    caption_extension: str
    shuffle_caption: bool
    keep_tokens: int


class LoraTrainResponse(TypedDict, total=True):
    """Response from LoRA training request.

    Attributes:
        job_id: Unique identifier for the training job.
    """

    job_id: str


class LoraStatusResponse(TypedDict, total=True):
    """Response for LoRA job status.

    Attributes:
        job_id: Unique job identifier.
        status: Current job status.
        message: Status message or None.
        lora_file_id: File ID of the trained LoRA in data-bank.
            Only present when status is 'completed'.
        lora_name: Name of the deployed LoRA in ComfyUI.
            Only present when status is 'completed'.
    """

    job_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    message: str | None
    lora_file_id: str | None
    lora_name: str | None


class LoraProgressResponse(TypedDict, total=True):
    """Response for LoRA job progress.

    Attributes:
        job_id: Unique job identifier.
        phase: Current training phase.
        step: Current training step.
        total_steps: Total training steps.
        loss: Current loss value.
        learning_rate: Current learning rate.
        updated_at: ISO 8601 timestamp of last update.
        lora_file_id: File ID of the trained LoRA in data-bank.
            Only present when phase is 'completed'.
        lora_name: Name of the deployed LoRA in ComfyUI.
            Only present when phase is 'completed'.
    """

    job_id: str
    phase: Literal[
        "queued",
        "preparing",
        "training",
        "saving",
        "uploading",
        "completed",
        "failed",
        "cancelled",
    ]
    step: int
    total_steps: int
    loss: float | None
    learning_rate: float
    updated_at: str
    lora_file_id: str | None
    lora_name: str | None


class LoraCancelResponse(TypedDict, total=True):
    """Response for LoRA job cancellation.

    Attributes:
        status: Cancellation status.
    """

    status: Literal["cancellation-requested"]


__all__ = [
    "LoraCancelResponse",
    "LoraProgressResponse",
    "LoraStatusResponse",
    "LoraTrainRequest",
    "LoraTrainResponse",
]
