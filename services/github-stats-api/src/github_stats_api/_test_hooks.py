"""Internal hooks for dependency injection in tests.

This module provides hook points that allow tests to inject fake implementations
without modifying core logic. Production code sets these to real implementations.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.http_client import HttpxAsyncClient, build_async_client


def _default_build_client(timeout_seconds: float) -> HttpxAsyncClient:
    """Default client builder using real httpx.

    Args:
        timeout_seconds: Request timeout.

    Returns:
        Async HTTP client.
    """
    return build_async_client(timeout_seconds=timeout_seconds)


# Hook for building HTTP clients - tests can replace with fake
_build_client_hook: Callable[[float], HttpxAsyncClient] = _default_build_client


def get_client_hook() -> Callable[[float], HttpxAsyncClient]:
    """Get current client builder hook.

    Returns:
        Current client builder function.
    """
    return _build_client_hook


def set_client_hook(hook: Callable[[float], HttpxAsyncClient]) -> None:
    """Set client builder hook for testing.

    Args:
        hook: Client builder function to use.
    """
    global _build_client_hook
    _build_client_hook = hook


def reset_client_hook() -> None:
    """Reset client builder hook to default."""
    global _build_client_hook
    _build_client_hook = _default_build_client


__all__ = ["get_client_hook", "reset_client_hook", "set_client_hook"]
