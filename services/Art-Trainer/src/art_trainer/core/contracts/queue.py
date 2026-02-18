"""Queue payload contracts for RQ job communication.

This module defines TypedDicts for job queue payloads.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class LoraTrainPayload(TypedDict, total=True):
    """Payload for LoRA training job queue.

    Attributes:
        job_id: Unique job identifier.
        user_id: User who initiated the training.
        base_model: Base model type (sd15, sdxl, or flux).
        training_type: Type of training (style, character, or concept).
        dataset_file_id: File ID for the dataset tarball in data-bank.
        steps: Number of training steps.
        learning_rate: Learning rate for training.
        network_rank: LoRA network rank (dimension).
        network_alpha: LoRA network alpha scaling factor.
        resolution: Training image resolution.
        batch_size: Training batch size.
        seed: Random seed for reproducibility.
        caption_extension: File extension for caption files.
        shuffle_caption: Whether to shuffle caption tokens.
        keep_tokens: Number of tokens to keep unshuffled.
    """

    job_id: str
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


__all__ = [
    "LoraTrainPayload",
]
