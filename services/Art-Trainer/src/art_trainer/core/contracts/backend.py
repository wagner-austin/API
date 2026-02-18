"""Training backend protocol and types.

This module defines the Protocol for training backends and related
callback types used during training.
"""

from __future__ import annotations

from typing import Protocol

from .lora import LoraTrainConfig, LoraTrainOutcome
from .progress import ArtTrainingProgress


class ProgressCallback(Protocol):
    """Protocol for progress reporting during training.

    Args:
        progress: Current training progress state.
    """

    def __call__(self, progress: ArtTrainingProgress) -> None:
        """Report training progress.

        Args:
            progress: Current training progress state.
        """
        ...


class CancelledCheck(Protocol):
    """Protocol for checking if training has been cancelled.

    Returns:
        True if training should be cancelled, False otherwise.
    """

    def __call__(self) -> bool:
        """Check if training has been cancelled.

        Returns:
            True if training should be cancelled, False otherwise.
        """
        ...


class TrainingBackend(Protocol):
    """Protocol for training backend implementations.

    Training backends handle the actual training process using
    specific tools (e.g., Kohya_ss, SimpleTuner, etc.).
    """

    def name(self) -> str:
        """Get the backend name.

        Returns:
            Human-readable backend name.
        """
        ...

    def is_available(self) -> bool:
        """Check if the backend is available for use.

        Returns:
            True if the backend is available, False otherwise.
        """
        ...

    def train(
        self,
        config: LoraTrainConfig,
        *,
        progress_callback: ProgressCallback | None = None,
        cancelled: CancelledCheck | None = None,
    ) -> LoraTrainOutcome:
        """Execute LoRA training.

        Args:
            config: Training configuration.
            progress_callback: Optional callback for progress reporting.
            cancelled: Optional callback to check for cancellation.

        Returns:
            Training outcome with success status and results.
        """
        ...


__all__ = [
    "CancelledCheck",
    "ProgressCallback",
    "TrainingBackend",
]
