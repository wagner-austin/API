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

from pathlib import Path
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


# =============================================================================
# Guard Script Hooks
# =============================================================================


class FindMonorepoRootProto(Protocol):
    """Protocol for _find_monorepo_root hook."""

    def __call__(self, start: Path) -> Path:
        """Find monorepo root starting from given path.

        Args:
            start: Starting path to search from.

        Returns:
            Path to monorepo root.
        """
        ...


class RunForProjectProto(Protocol):
    """Protocol for run_for_project hook."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guard checks for a project.

        Args:
            monorepo_root: Path to monorepo root.
            project_root: Path to project root.

        Returns:
            Exit code from guard checks.
        """
        ...


class LoadOrchestratorProto(Protocol):
    """Protocol for _load_orchestrator hook."""

    def __call__(self, monorepo_root: Path) -> RunForProjectProto:
        """Load the guard orchestrator.

        Args:
            monorepo_root: Path to monorepo root.

        Returns:
            run_for_project function.
        """
        ...


# Guard hooks - None means use default behavior (production implementation)
guard_find_monorepo_root: FindMonorepoRootProto | None = None
guard_load_orchestrator: LoadOrchestratorProto | None = None


__all__ = [
    "BackendFactoryProtocol",
    "FindMonorepoRootProto",
    "LoadOrchestratorProto",
    "RunForProjectProto",
    "_default_backend_factory",
    "backend_factory",
    "guard_find_monorepo_root",
    "guard_load_orchestrator",
]
