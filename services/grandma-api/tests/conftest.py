"""Test fixtures for grandma-api."""

from __future__ import annotations

from collections.abc import Generator
from typing import BinaryIO

import pytest
from platform_core.config import config_test_hooks
from platform_langid._test_hooks import SpokenLanguageDetectorProtocol
from platform_langid.types import DetectorConfig, SpokenLanguageResult
from platform_stt import VerboseResponse, VerboseSegment
from platform_stt._test_hooks import STTClientProtocol
from platform_translate.backends.protocol import TranslationBackendProtocol
from platform_translate.types import TranslationResult, TranslatorConfig
from scripts import _test_hooks as scripts_test_hooks

from grandma_api.config import GrandmaApiSettings
from grandma_api.core.container import (
    LangIdDetectorFactoryProtocol,
    ServiceContainer,
    STTClientFactoryProtocol,
    TranslatorFactoryProtocol,
)


class FakeSTTClient:
    """Fake STT client for testing.

    Returns configurable responses without making real API calls.
    """

    __slots__ = ("_response", "call_count", "last_file")

    def __init__(self, response: VerboseResponse | None = None) -> None:
        """Initialize fake client.

        Args:
            response: Response to return from translate().
        """
        default = VerboseResponse(
            text="Hello from grandmother",
            language="vi",
            segments=[VerboseSegment(text="Hello", start=0.0, end=1.0)],
        )
        self._response = response if response is not None else default
        self.call_count = 0
        self.last_file: bytes | None = None

    def transcribe(
        self,
        *,
        file: BinaryIO,
        language: str | None = None,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Fake transcription.

        Args:
            file: Binary file-like object.
            language: Optional language hint.
            timeout: Optional timeout.

        Returns:
            Configured response.
        """
        _ = (language, timeout)
        self.last_file = file.read()
        file.seek(0)
        self.call_count += 1
        return self._response

    def translate(
        self,
        *,
        file: BinaryIO,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Fake translation.

        Args:
            file: Binary file-like object.
            timeout: Optional timeout.

        Returns:
            Configured response.
        """
        _ = timeout
        self.last_file = file.read()
        file.seek(0)
        self.call_count += 1
        return self._response


class FakeLangIdDetector:
    """Fake language ID detector for testing.

    Returns configurable detection results without loading ML models.
    """

    __slots__ = ("_result", "call_count")

    def __init__(self, result: SpokenLanguageResult | None = None) -> None:
        """Initialize fake detector.

        Args:
            result: Result to return from detect().
        """
        default = SpokenLanguageResult(
            language="vi",
            confidence=0.95,
            model_id="facebook/mms-lid-4017",
        )
        self._result = result if result is not None else default
        self.call_count = 0

    def detect(self, audio_bytes: bytes, sample_rate: int) -> SpokenLanguageResult:
        """Fake detection.

        Args:
            audio_bytes: Audio data bytes.
            sample_rate: Audio sample rate.

        Returns:
            Configured result.
        """
        _ = (audio_bytes, sample_rate)
        self.call_count += 1
        return self._result


class FakeTranslator:
    """Fake translator for testing.

    Returns configurable translation results without making API calls.
    """

    __slots__ = ("_translated_text", "call_count")

    def __init__(self, translated_text: str = "Hello from grandmother") -> None:
        """Initialize fake translator.

        Args:
            translated_text: Text to return from translate().
        """
        self._translated_text = translated_text
        self.call_count = 0

    @property
    def backend_id(self) -> str:
        """Get backend identifier."""
        return "fake"

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Fake translation.

        Args:
            text: Source text.
            source_language: Source language code.
            target_language: Target language code.

        Returns:
            Translation result with configured text.
        """
        self.call_count += 1
        return TranslationResult(
            text=self._translated_text,
            source_language=source_language,
            target_language=target_language,
            backend="fake",
        )


def make_fake_stt_client(
    response: VerboseResponse | None = None,
) -> tuple[FakeSTTClient, STTClientFactoryProtocol]:
    """Create fake STT client and factory.

    Args:
        response: Response to return from client.

    Returns:
        Tuple of (fake_client, factory_function).
    """
    client = FakeSTTClient(response)

    def factory(api_key: str) -> STTClientProtocol:
        del api_key  # unused
        return client

    return client, factory


def make_fake_langid_detector(
    result: SpokenLanguageResult | None = None,
) -> tuple[FakeLangIdDetector, LangIdDetectorFactoryProtocol]:
    """Create fake language ID detector and factory.

    Args:
        result: Result to return from detector.

    Returns:
        Tuple of (fake_detector, factory_function).
    """
    detector = FakeLangIdDetector(result)

    def factory(config: DetectorConfig) -> SpokenLanguageDetectorProtocol:
        del config  # unused
        return detector

    return detector, factory


def make_fake_translator(
    translated_text: str = "Hello from grandmother",
) -> tuple[FakeTranslator, TranslatorFactoryProtocol]:
    """Create fake translator and factory.

    Args:
        translated_text: Text to return from translations.

    Returns:
        Tuple of (fake_translator, factory_function).
    """
    translator = FakeTranslator(translated_text)

    def factory(config: TranslatorConfig) -> TranslationBackendProtocol:
        del config  # unused
        return translator

    return translator, factory


def make_test_container(
    settings: GrandmaApiSettings,
    response: VerboseResponse | None = None,
    detection_result: SpokenLanguageResult | None = None,
    translated_text: str = "Hello from grandmother",
) -> tuple[ServiceContainer, FakeSTTClient, FakeLangIdDetector, FakeTranslator]:
    """Create ServiceContainer with fake services for testing.

    Args:
        settings: Test settings.
        response: Response to return from fake STT client.
        detection_result: Result to return from fake detector.
        translated_text: Text to return from fake translator.

    Returns:
        Tuple of (ServiceContainer, FakeSTTClient, FakeLangIdDetector, FakeTranslator).
    """
    stt_client, stt_factory = make_fake_stt_client(response)
    langid_detector, langid_factory = make_fake_langid_detector(detection_result)
    translator, translator_factory = make_fake_translator(translated_text)
    container = ServiceContainer(
        settings=settings,
        stt_client_factory=stt_factory,
        langid_detector_factory=langid_factory,
        translator_factory=translator_factory,
    )
    return container, stt_client, langid_detector, translator


def set_fake_env(env: dict[str, str]) -> None:
    """Set fake environment variables for testing.

    Args:
        env: Dictionary of environment variable values.
    """

    def _fake_env(key: str) -> str | None:
        return env.get(key)

    config_test_hooks.get_env = _fake_env


@pytest.fixture(autouse=True)
def _restore_config_hooks() -> Generator[None, None, None]:
    """Restore config hooks after each test."""
    original_get_env = config_test_hooks.get_env
    yield
    config_test_hooks.get_env = original_get_env


@pytest.fixture(autouse=True)
def _restore_scripts_hooks() -> Generator[None, None, None]:
    """Restore scripts hooks after each test.

    The webserver tests inject a server factory and a serve function; without
    this the fakes outlive the test that set them.
    """
    original_serve_forever = scripts_test_hooks.serve_forever
    original_server_factory = scripts_test_hooks.server_factory
    yield
    scripts_test_hooks.serve_forever = original_serve_forever
    scripts_test_hooks.server_factory = original_server_factory


def generate_test_wav() -> bytes:
    """Generate a real WAV file with 1 second of silence at 16kHz mono.

    Returns:
        Valid WAV file bytes.
    """
    sample_rate = 16000
    duration_seconds = 1
    num_samples = sample_rate * duration_seconds
    bits_per_sample = 16
    num_channels = 1
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    # WAV header
    header = bytearray()
    header.extend(b"RIFF")
    header.extend((36 + data_size).to_bytes(4, "little"))
    header.extend(b"WAVE")
    header.extend(b"fmt ")
    header.extend((16).to_bytes(4, "little"))  # Subchunk1Size
    header.extend((1).to_bytes(2, "little"))  # AudioFormat (PCM)
    header.extend(num_channels.to_bytes(2, "little"))
    header.extend(sample_rate.to_bytes(4, "little"))
    header.extend(byte_rate.to_bytes(4, "little"))
    header.extend(block_align.to_bytes(2, "little"))
    header.extend(bits_per_sample.to_bytes(2, "little"))
    header.extend(b"data")
    header.extend(data_size.to_bytes(4, "little"))

    # Silence data
    audio_data = b"\x00" * data_size

    return bytes(header) + audio_data
