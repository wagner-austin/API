"""Tests for platform_stt.whisper_client module."""

from __future__ import annotations

import io
from typing import BinaryIO

from platform_stt import _test_hooks
from platform_stt.testing import reset_hooks
from platform_stt.types import BinaryFileProtocol, RawVerboseDict
from platform_stt.whisper_client import OpenAISttClient


class FakeTranscriptions:
    """Fake transcriptions interface."""

    __slots__ = ("_response", "calls")

    def __init__(self, response: RawVerboseDict | None = None) -> None:
        default: RawVerboseDict = {
            "text": "Test transcription",
            "segments": [{"text": "Test", "start": 0.0, "end": 1.0}],
        }
        self._response = response or default
        self.calls: list[dict[str, str | float | None]] = []

    def create(
        self,
        *,
        model: str,
        file: BinaryFileProtocol,
        response_format: str,
        language: str | None = None,
        timeout: float | None = None,
    ) -> RawVerboseDict:
        """Record call and return response."""
        self.calls.append(
            {
                "model": model,
                "response_format": response_format,
                "language": language,
                "timeout": timeout,
            }
        )
        return self._response


class FakeTranslations:
    """Fake translations interface."""

    __slots__ = ("_response", "calls")

    def __init__(self, response: RawVerboseDict | None = None) -> None:
        default: RawVerboseDict = {
            "text": "Test translation",
            "segments": [{"text": "Test", "start": 0.0, "end": 1.0}],
        }
        self._response = response or default
        self.calls: list[dict[str, str | float | None]] = []

    def create(
        self,
        *,
        model: str,
        file: BinaryFileProtocol,
        response_format: str,
        timeout: float | None = None,
    ) -> RawVerboseDict:
        """Record call and return response."""
        self.calls.append(
            {
                "model": model,
                "response_format": response_format,
                "timeout": timeout,
            }
        )
        return self._response


class FakeAudioNamespace:
    """Fake audio namespace."""

    __slots__ = ("_transcriptions", "_translations")

    def __init__(
        self,
        transcriptions: FakeTranscriptions | None = None,
        translations: FakeTranslations | None = None,
    ) -> None:
        self._transcriptions = transcriptions or FakeTranscriptions()
        self._translations = translations or FakeTranslations()

    @property
    def transcriptions(self) -> FakeTranscriptions:
        return self._transcriptions

    @property
    def translations(self) -> FakeTranslations:
        return self._translations


class FakeOpenAIClient:
    """Fake OpenAI client for testing."""

    __slots__ = ("_audio",)

    def __init__(self, audio: FakeAudioNamespace | None = None) -> None:
        self._audio = audio or FakeAudioNamespace()

    @property
    def audio(self) -> FakeAudioNamespace:
        return self._audio


class FakeOpenAIClientFactory:
    """Factory that creates FakeOpenAIClient instances."""

    __slots__ = ("_client", "calls")

    def __init__(self, client: FakeOpenAIClient) -> None:
        self._client = client
        self.calls: list[dict[str, str | float | int]] = []

    def __call__(
        self, *, api_key: str, timeout: float, max_retries: int
    ) -> _test_hooks.OpenAIClientProtocol:
        """Create client and record call."""
        self.calls.append(
            {
                "api_key": api_key,
                "timeout": timeout,
                "max_retries": max_retries,
            }
        )
        return self._client


class TestOpenAISttClient:
    """Tests for OpenAISttClient class."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_init(self) -> None:
        """Initialize client with parameters."""
        fake_client = FakeOpenAIClient()
        factory = FakeOpenAIClientFactory(fake_client)
        _test_hooks.openai_client_factory = factory

        client = OpenAISttClient(
            api_key="test-key",
            timeout_seconds=600.0,
            max_retries=3,
        )

        assert client.api_key == "test-key"
        assert client.timeout_seconds == 600.0
        assert client.max_retries == 3
        assert factory.calls[0]["api_key"] == "test-key"
        assert factory.calls[0]["timeout"] == 600.0
        assert factory.calls[0]["max_retries"] == 3

    def test_transcribe_basic(self) -> None:
        """Transcribe audio file."""
        transcriptions = FakeTranscriptions(
            {
                "text": "Hello world",
                "segments": [
                    {"text": "Hello", "start": 0.0, "end": 0.5},
                    {"text": "world", "start": 0.5, "end": 1.0},
                ],
            }
        )
        audio = FakeAudioNamespace(transcriptions=transcriptions)
        fake_client = FakeOpenAIClient(audio=audio)
        factory = FakeOpenAIClientFactory(fake_client)
        _test_hooks.openai_client_factory = factory

        client = OpenAISttClient(api_key="test-key")
        file_obj: BinaryIO = io.BytesIO(b"fake audio")
        result = client.transcribe(file=file_obj)

        assert result["text"] == "Hello world"
        assert len(result["segments"]) == 2
        assert transcriptions.calls[0]["model"] == "whisper-1"
        assert transcriptions.calls[0]["response_format"] == "verbose_json"

    def test_transcribe_with_language(self) -> None:
        """Transcribe with language hint."""
        transcriptions = FakeTranscriptions()
        audio = FakeAudioNamespace(transcriptions=transcriptions)
        fake_client = FakeOpenAIClient(audio=audio)
        factory = FakeOpenAIClientFactory(fake_client)
        _test_hooks.openai_client_factory = factory

        client = OpenAISttClient(api_key="test-key")
        file_obj: BinaryIO = io.BytesIO(b"fake audio")
        client.transcribe(file=file_obj, language="vi")

        assert transcriptions.calls[0]["language"] == "vi"

    def test_transcribe_with_timeout(self) -> None:
        """Transcribe with custom timeout."""
        transcriptions = FakeTranscriptions()
        audio = FakeAudioNamespace(transcriptions=transcriptions)
        fake_client = FakeOpenAIClient(audio=audio)
        factory = FakeOpenAIClientFactory(fake_client)
        _test_hooks.openai_client_factory = factory

        client = OpenAISttClient(api_key="test-key")
        file_obj: BinaryIO = io.BytesIO(b"fake audio")
        client.transcribe(file=file_obj, timeout=300.0)

        assert transcriptions.calls[0]["timeout"] == 300.0

    def test_translate_basic(self) -> None:
        """Translate audio to English."""
        translations = FakeTranslations(
            {
                "text": "Hello in English",
                "segments": [{"text": "Hello in English", "start": 0.0, "end": 1.0}],
            }
        )
        audio = FakeAudioNamespace(translations=translations)
        fake_client = FakeOpenAIClient(audio=audio)
        factory = FakeOpenAIClientFactory(fake_client)
        _test_hooks.openai_client_factory = factory

        client = OpenAISttClient(api_key="test-key")
        file_obj: BinaryIO = io.BytesIO(b"fake audio")
        result = client.translate(file=file_obj)

        assert result["text"] == "Hello in English"
        assert translations.calls[0]["model"] == "whisper-1"
        assert translations.calls[0]["response_format"] == "verbose_json"

    def test_translate_with_timeout(self) -> None:
        """Translate with custom timeout."""
        translations = FakeTranslations()
        audio = FakeAudioNamespace(translations=translations)
        fake_client = FakeOpenAIClient(audio=audio)
        factory = FakeOpenAIClientFactory(fake_client)
        _test_hooks.openai_client_factory = factory

        client = OpenAISttClient(api_key="test-key")
        file_obj: BinaryIO = io.BytesIO(b"fake audio")
        client.translate(file=file_obj, timeout=120.0)

        assert translations.calls[0]["timeout"] == 120.0

    def test_process_transcribe(self) -> None:
        """Process with transcribe task."""
        transcriptions = FakeTranscriptions(
            {
                "text": "Transcribed",
                "segments": [{"text": "Transcribed", "start": 0.0, "end": 1.0}],
            }
        )
        audio = FakeAudioNamespace(transcriptions=transcriptions)
        fake_client = FakeOpenAIClient(audio=audio)
        factory = FakeOpenAIClientFactory(fake_client)
        _test_hooks.openai_client_factory = factory

        client = OpenAISttClient(api_key="test-key")
        file_obj: BinaryIO = io.BytesIO(b"fake audio")
        result = client.process(file=file_obj, task="transcribe", language="en")

        assert result["text"] == "Transcribed"
        assert transcriptions.calls[0]["language"] == "en"

    def test_process_translate(self) -> None:
        """Process with translate task."""
        translations = FakeTranslations(
            {
                "text": "Translated",
                "segments": [{"text": "Translated", "start": 0.0, "end": 1.0}],
            }
        )
        audio = FakeAudioNamespace(translations=translations)
        fake_client = FakeOpenAIClient(audio=audio)
        factory = FakeOpenAIClientFactory(fake_client)
        _test_hooks.openai_client_factory = factory

        client = OpenAISttClient(api_key="test-key")
        file_obj: BinaryIO = io.BytesIO(b"fake audio")
        result = client.process(file=file_obj, task="translate")

        assert result["text"] == "Translated"
        assert len(translations.calls) == 1
