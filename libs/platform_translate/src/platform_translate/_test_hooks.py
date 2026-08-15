"""Internal test hooks for platform_translate - allows injecting test dependencies.

This module provides dependency injection hooks following the pattern:
- Production code sets hooks to real implementations at startup
- Tests set hooks to fakes before running

Usage in production:
    # At startup, hooks are already set to defaults (production implementations)

Usage in tests:
    from platform_translate import _test_hooks
    _test_hooks.backend_factory = fake_backend_factory
    # ... run test ...
    # Reset after test if needed
"""

from __future__ import annotations

from typing import Protocol

from .backends.protocol import TranslationBackendProtocol
from .types import TranslatorConfig

# =============================================================================
# Backend Factory Protocol
# =============================================================================


class BackendFactoryProtocol(Protocol):
    """Protocol for backend factory function."""

    def __call__(self, config: TranslatorConfig) -> TranslationBackendProtocol:
        """Create translation backend from configuration.

        Args:
            config: Translator configuration.

        Returns:
            Configured translation backend.

        Raises:
            ValueError: If backend type is not supported.
        """
        ...


# =============================================================================
# Default Implementations
# =============================================================================


def _default_backend_factory(config: TranslatorConfig) -> TranslationBackendProtocol:
    """Production implementation - creates real backend based on config.

    Args:
        config: Translator configuration.

    Returns:
        Configured translation backend.

    Raises:
        ValueError: If backend type is not supported.
    """
    backend_type = config["backend"]

    if backend_type == "anthropic":
        from .backends.anthropic import create_anthropic_backend

        backend: TranslationBackendProtocol = create_anthropic_backend(
            api_key=config["api_key"],
            model=config["model"],
        )
        return backend

    if backend_type == "openai":
        from .backends.openai import create_openai_backend

        return create_openai_backend(
            api_key=config["api_key"],
            model=config["model"],
        )

    raise ValueError(f"Unsupported backend: {backend_type}")


# =============================================================================
# Module-level Hooks
# =============================================================================


# Hook for backend creation
backend_factory: BackendFactoryProtocol = _default_backend_factory


__all__ = [
    "BackendFactoryProtocol",
    "_default_backend_factory",
    "backend_factory",
]
