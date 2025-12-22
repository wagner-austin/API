"""Test hooks for discover_datasets module.

Provides dependency injection points for testing without mocks.
"""

from __future__ import annotations

from typing import Protocol

from platform_core.logging import RichConsoleProtocol, get_rich_console


class ConsoleFactory(Protocol):
    """Protocol for console factory function."""

    def __call__(self) -> RichConsoleProtocol:
        """Create and return a console instance."""
        ...


def _default_console_factory() -> RichConsoleProtocol:
    """Default console factory using platform_core."""
    return get_rich_console()


console_factory: ConsoleFactory = _default_console_factory
