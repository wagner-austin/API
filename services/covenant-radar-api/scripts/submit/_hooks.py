"""Test hooks for submit pipeline components.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_ml.backends.registry import ClassifierRegistry

# =============================================================================
# Console Hook
# =============================================================================


class ConsoleProtocol(Protocol):
    """Protocol for console output."""

    def write(self, message: str) -> None:
        """Write a message to the console.

        Args:
            message: Message to write.
        """
        ...


class _RichConsoleAdapter:
    """Adapter that wraps Rich console to implement ConsoleProtocol."""

    def write(self, message: str) -> None:
        """Write a message using Rich console.

        Args:
            message: Message to write.
        """
        from platform_core.rich_logging import get_rich_console

        console = get_rich_console()
        console_print = console.print
        console_print(message)


class ConsoleHookCallable(Protocol):
    """Protocol for console hook factory function."""

    def __call__(self) -> ConsoleProtocol:
        """Create a console for output.

        Returns:
            ConsoleProtocol implementation.
        """
        ...


def _default_console_factory() -> ConsoleProtocol:
    """Default console factory returning Rich console adapter.

    Returns:
        RichConsoleAdapter instance.
    """
    return _RichConsoleAdapter()


console_hook: ConsoleHookCallable = _default_console_factory


def get_console() -> ConsoleProtocol:
    """Get the current console via hook.

    Returns:
        ConsoleProtocol implementation from current hook.
    """
    return console_hook()


# =============================================================================
# Project Root Hook
# =============================================================================


class ProjectRootCallable(Protocol):
    """Protocol for project root path factory function."""

    def __call__(self) -> Path:
        """Get the project root path.

        Returns:
            Path to project root directory.
        """
        ...


def _default_project_root() -> Path:
    """Default project root factory.

    Returns:
        Path to covenant-radar-api root.
    """
    return Path(__file__).parent.parent.parent


project_root_hook: ProjectRootCallable = _default_project_root


def get_project_root() -> Path:
    """Get project root path via hook.

    Returns:
        Path to project root directory.
    """
    return project_root_hook()


# =============================================================================
# Registry Hook
# =============================================================================


class RegistryHookCallable(Protocol):
    """Protocol for registry hook factory function."""

    def __call__(self) -> ClassifierRegistry:
        """Create a classifier registry.

        Returns:
            ClassifierRegistry implementation.
        """
        ...


def _default_registry_factory() -> ClassifierRegistry:
    """Default registry factory returning real ClassifierRegistry.

    Returns:
        ClassifierRegistry from covenant_ml.
    """
    from covenant_ml.backends.registry import default_registry

    return default_registry()


registry_hook: RegistryHookCallable = _default_registry_factory


def get_registry() -> ClassifierRegistry:
    """Get the current registry via hook.

    Returns:
        ClassifierRegistry from current hook.
    """
    return registry_hook()


__all__ = [
    "ClassifierRegistry",
    "ConsoleHookCallable",
    "ConsoleProtocol",
    "ProjectRootCallable",
    "RegistryHookCallable",
    "console_hook",
    "get_console",
    "get_project_root",
    "get_registry",
    "project_root_hook",
    "registry_hook",
]
