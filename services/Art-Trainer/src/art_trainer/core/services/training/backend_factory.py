"""Backend factory for creating training backends.

This module provides factory functions for creating training backend instances.
"""

from __future__ import annotations

from art_trainer.core.config.settings import Settings
from art_trainer.core.contracts.backend import TrainingBackend

from .backends.kohya.adapter import KohyaBackend


def create_kohya_backend(settings: Settings) -> TrainingBackend:
    """Create a Kohya_ss training backend.

    Args:
        settings: Application settings.

    Returns:
        KohyaBackend instance.
    """
    return KohyaBackend(settings)


__all__ = [
    "create_kohya_backend",
]
