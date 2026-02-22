"""Tests for grandma_api.core.container module."""

from __future__ import annotations

from platform_langid._test_hooks import SpokenLanguageDetectorProtocol
from platform_langid.types import DetectorConfig, SpokenLanguageResult
from platform_translate.types import TranslatorConfig

from grandma_api.config import GrandmaApiSettings
from grandma_api.core.container import (
    ServiceContainer,
    _default_langid_detector_factory,
    _default_translator_factory,
)

from .conftest import (
    make_fake_langid_detector,
    make_fake_stt_client,
    make_fake_translator,
)


def test_default_langid_detector_factory_creates_detector() -> None:
    """Test _default_langid_detector_factory creates a detector."""
    config = DetectorConfig(
        model_id="facebook/mms-lid-4017",
        device="cpu",
        confidence_threshold=0.0,
    )
    detector = _default_langid_detector_factory(config)
    assert callable(detector.detect)


def test_default_translator_factory_creates_translator() -> None:
    """Test _default_translator_factory creates a translator."""
    config = TranslatorConfig(
        backend="anthropic",
        api_key="test-key",
        model="claude-3-haiku-20240307",
    )
    translator = _default_translator_factory(config)
    assert translator.backend_id == "anthropic"


def test_get_langid_detector_returns_detector() -> None:
    """Test get_langid_detector returns a configured detector.

    Verifies that the container creates a detector using the factory
    with the default detector configuration.
    """
    settings = GrandmaApiSettings(
        openai_api_key="sk-test",
        api_token="test-token",
        port=8080,
        log_level="INFO",
        log_format="json",
    )
    expected_result = SpokenLanguageResult(
        language="es",
        confidence=0.98,
        model_id="facebook/mms-lid-4017",
    )
    fake_detector, langid_factory = make_fake_langid_detector(expected_result)
    _, stt_factory = make_fake_stt_client()
    _, translator_factory = make_fake_translator()

    container = ServiceContainer(
        settings=settings,
        stt_client_factory=stt_factory,
        langid_detector_factory=langid_factory,
        translator_factory=translator_factory,
    )

    detector: SpokenLanguageDetectorProtocol = container.get_langid_detector()

    # Verify detector is returned and callable
    assert callable(detector.detect)
    # Verify it's the fake detector
    result = detector.detect(b"audio", 16000)
    assert result["language"] == "es"
    assert result["confidence"] == 0.98
    assert fake_detector.call_count == 1
