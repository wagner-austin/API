"""Tests for grandma_api.api._test_hooks module."""

from __future__ import annotations

from platform_stt import _test_hooks as stt_hooks
from platform_stt._test_hooks import OpenAIClientProtocol, STTClientProtocol
from platform_stt.types import BinaryFileProtocol, RawVerboseDict

from grandma_api.api import _test_hooks

from .conftest import FakeSTTClient


class _FakeOpenAIAudioTranscriptions:
    """Fake transcriptions interface."""

    def create(
        self,
        *,
        model: str,
        file: BinaryFileProtocol,
        response_format: str,
        language: str | None = None,
        timeout: float | None = None,
    ) -> RawVerboseDict:
        return {"text": "test", "segments": [], "language": "en"}


class _FakeOpenAIAudioTranslations:
    """Fake translations interface."""

    def create(
        self,
        *,
        model: str,
        file: BinaryFileProtocol,
        response_format: str,
        timeout: float | None = None,
    ) -> RawVerboseDict:
        return {"text": "test", "segments": [], "language": "en"}


class _FakeOpenAIAudio:
    """Fake audio namespace."""

    @property
    def transcriptions(self) -> _FakeOpenAIAudioTranscriptions:
        return _FakeOpenAIAudioTranscriptions()

    @property
    def translations(self) -> _FakeOpenAIAudioTranslations:
        return _FakeOpenAIAudioTranslations()


class _FakeOpenAIClient:
    """Fake OpenAI client."""

    @property
    def audio(self) -> _FakeOpenAIAudio:
        return _FakeOpenAIAudio()


def test_stt_client_factory_is_default() -> None:
    """Test stt_client_factory hook is set to default implementation."""
    assert _test_hooks.stt_client_factory is _test_hooks._default_stt_client_factory


def test_default_stt_client_factory_creates_client() -> None:
    """Test _default_stt_client_factory creates an STT client."""
    # Mock the OpenAI client factory to avoid needing real openai module
    original = stt_hooks.openai_client_factory

    def _fake_factory(*, api_key: str, timeout: float, max_retries: int) -> OpenAIClientProtocol:
        return _FakeOpenAIClient()

    stt_hooks.openai_client_factory = _fake_factory

    try:
        client = _test_hooks._default_stt_client_factory("sk-test-key")
        # Verify it has the expected protocol methods
        assert callable(client.transcribe)
        assert callable(client.translate)
    finally:
        stt_hooks.openai_client_factory = original


def test_stt_client_factory_can_be_overridden() -> None:
    """Test stt_client_factory hook can be overridden for testing."""
    original = _test_hooks.stt_client_factory
    call_count = 0

    def fake_factory(api_key: str) -> STTClientProtocol:
        nonlocal call_count
        call_count += 1
        return FakeSTTClient()

    _test_hooks.stt_client_factory = fake_factory

    # Verify factory can be called
    _ = _test_hooks.stt_client_factory("test-key")
    assert call_count == 1

    _test_hooks.stt_client_factory = original


def test_reset_hooks_restores_defaults() -> None:
    """Test reset_hooks restores production implementations."""
    original = _test_hooks.stt_client_factory

    def fake_factory(api_key: str) -> STTClientProtocol:
        return FakeSTTClient()

    _test_hooks.stt_client_factory = fake_factory
    assert _test_hooks.stt_client_factory is fake_factory

    _test_hooks.reset_hooks()
    assert _test_hooks.stt_client_factory is _test_hooks._default_stt_client_factory

    # Restore for other tests
    _test_hooks.stt_client_factory = original
