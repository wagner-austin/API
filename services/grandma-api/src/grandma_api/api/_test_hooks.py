"""Internal test hooks for grandma-api API - allows injecting test dependencies.

This module re-exports test hooks from the core module for backwards compatibility.
New code should use grandma_api.core.container directly with ServiceContainer.

Usage in tests:
    from grandma_api.api import _test_hooks
    _test_hooks.stt_client_factory = lambda api_key: FakeSTTClient()
"""

from __future__ import annotations

from grandma_api.core.container import (
    STTClientFactoryProtocol,
    _default_stt_client_factory,
)

# Hook for STT client factory. Tests can override to provide fake client.
# This is a module-level variable that tests can replace.
stt_client_factory: STTClientFactoryProtocol = _default_stt_client_factory


def reset_hooks() -> None:
    """Reset all hooks to their production defaults.

    Call this in test teardown to ensure clean state.
    """
    global stt_client_factory
    stt_client_factory = _default_stt_client_factory


__all__ = [
    "STTClientFactoryProtocol",
    "_default_stt_client_factory",
    "reset_hooks",
    "stt_client_factory",
]
