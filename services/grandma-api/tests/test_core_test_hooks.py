"""Tests for grandma_api.core._test_hooks module."""

from __future__ import annotations

from platform_stt._test_hooks import STTClientProtocol

from grandma_api.core import _test_hooks
from grandma_api.core.container import _default_stt_client_factory

from .conftest import FakeSTTClient


def test_stt_client_factory_default() -> None:
    """Test stt_client_factory defaults to production implementation."""
    assert _test_hooks.stt_client_factory is _default_stt_client_factory


def test_stt_client_factory_can_be_overridden() -> None:
    """Test stt_client_factory can be replaced for testing."""
    original = _test_hooks.stt_client_factory

    def fake_factory(api_key: str) -> STTClientProtocol:
        del api_key
        return FakeSTTClient()

    _test_hooks.stt_client_factory = fake_factory
    assert _test_hooks.stt_client_factory is fake_factory

    # Restore
    _test_hooks.stt_client_factory = original


def test_reset_hooks_restores_defaults() -> None:
    """Test reset_hooks restores production implementations."""
    original = _test_hooks.stt_client_factory

    def fake_factory(api_key: str) -> STTClientProtocol:
        del api_key
        return FakeSTTClient()

    _test_hooks.stt_client_factory = fake_factory
    assert _test_hooks.stt_client_factory is fake_factory

    _test_hooks.reset_hooks()
    assert _test_hooks.stt_client_factory is _default_stt_client_factory

    # Ensure original is restored
    _test_hooks.stt_client_factory = original
