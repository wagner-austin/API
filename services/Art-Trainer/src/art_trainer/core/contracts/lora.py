"""LoRA training configuration and outcome contracts.

This module defines the core TypedDicts for LoRA training configuration
and training outcome results.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class LoraTrainConfig(TypedDict, total=True):
    """Configuration for LoRA training.

    Attributes:
        job_id: Unique identifier for the training job.
        base_model: Base model type (sd15, sdxl, or flux).
        training_type: Type of training (style, character, or concept).
        dataset_dir: Path to the training dataset directory.
        output_dir: Path for output model files.
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
    base_model: Literal["sd15", "sdxl", "flux"]
    training_type: Literal["style", "character", "concept"]
    dataset_dir: str
    output_dir: str
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


class LoraTrainOutcome(TypedDict, total=True):
    """Outcome of a LoRA training run.

    Attributes:
        success: Whether training completed successfully.
        lora_path: Path to the trained LoRA file (None on failure).
        final_loss: Final training loss value (None on failure).
        error_message: Error message if training failed (None on success).
    """

    success: bool
    lora_path: str | None
    final_loss: float | None
    error_message: str | None


__all__ = [
    "LoraTrainConfig",
    "LoraTrainOutcome",
]
