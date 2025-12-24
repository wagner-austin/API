"""Internal test hooks for grandma-api core - allows injecting test dependencies.

This module provides dependency injection hooks for the core layer.
Production code sets hooks to real implementations at startup;
tests set them to fakes.

Usage in tests:
    from grandma_api.core import _test_hooks
    _test_hooks.stt_client_factory = lambda api_key: FakeSTTClient()
"""

from __future__ import annotations

from grandma_api.core.container import (
    STTClientFactoryProtocol,
    _default_stt_client_factory,
)

# Hook for STT client factory. Tests can override to provide fake client.
stt_client_factory: STTClientFactoryProtocol = _default_stt_client_factory


def reset_hooks() -> None:
    """Reset all hooks to their production defaults.

    Call this in test teardown to ensure clean state.
    """
    global stt_client_factory
    stt_client_factory = _default_stt_client_factory


__all__ = [
    "reset_hooks",
    "stt_client_factory",
]
