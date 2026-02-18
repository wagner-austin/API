"""Registry for training backends.

This module provides a registry pattern for managing training backends.
"""

from __future__ import annotations

from typing import Protocol

from art_trainer.core.config.settings import Settings
from art_trainer.core.contracts.backend import TrainingBackend


class BackendFactory(Protocol):
    """Protocol for backend factory functions."""

    def __call__(self, settings: Settings) -> TrainingBackend:
        """Create a training backend.

        Args:
            settings: Application settings.

        Returns:
            TrainingBackend instance.
        """
        ...


class BackendRegistry:
    """Registry for training backends.

    Manages registration and lookup of training backends.
    """

    _backends: dict[str, BackendFactory]
    _settings: Settings

    def __init__(self, backends: dict[str, BackendFactory], settings: Settings) -> None:
        """Initialize backend registry.

        Args:
            backends: Dictionary mapping backend names to factories.
            settings: Application settings.
        """
        self._backends = backends
        self._settings = settings

    def get(self, name: str) -> TrainingBackend | None:
        """Get a backend by name.

        Args:
            name: Backend name.

        Returns:
            TrainingBackend instance or None if not found.
        """
        factory = self._backends.get(name)
        if factory is None:
            return None
        return factory(self._settings)

    def available_backends(self) -> list[str]:
        """Get list of available backend names.

        Returns:
            List of registered backend names.
        """
        return list(self._backends.keys())


__all__ = [
    "BackendFactory",
    "BackendRegistry",
]
