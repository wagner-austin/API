"""OpenAI Whisper STT client with transcription and translation support.

Provides a typed interface to the OpenAI Whisper API with support for:
- Transcription (audio -> text in source language)
- Translation (audio -> English text)
- Configurable language hints for better accuracy
"""

from __future__ import annotations

from typing import BinaryIO

from platform_core.logging import get_logger

from . import _test_hooks
from .types import VerboseResponse, WhisperTask
from .whisper_parse import to_verbose_response


class OpenAISttClient:
    """OpenAI Whisper API client for speech-to-text operations.

    Supports both transcription (same language) and translation (to English).
    Uses the whisper-1 model via OpenAI API.

    Attributes:
        api_key: OpenAI API key.
        timeout_seconds: Request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
    """

    __slots__ = ("_client", "_logger", "api_key", "max_retries", "timeout_seconds")

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 900.0,
        max_retries: int = 2,
    ) -> None:
        """Initialize OpenAI STT client.

        Args:
            api_key: OpenAI API key for authentication.
            timeout_seconds: Request timeout in seconds (default: 900).
            max_retries: Maximum retry attempts (default: 2).
        """
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = self._make_client()
        self._logger = get_logger(__name__)

    def _make_client(self) -> _test_hooks.OpenAIClientProtocol:
        """Create OpenAI client instance."""
        return _test_hooks.openai_client_factory(
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    def transcribe(
        self,
        *,
        file: BinaryIO,
        language: str | None = None,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Transcribe audio file to text in source language.

        Args:
            file: Binary file-like object containing audio data.
            language: Optional ISO 639-1 language code hint (e.g., "vi" for Vietnamese).
                     If None, Whisper auto-detects the language.
            timeout: Optional request timeout override.

        Returns:
            VerboseResponse with transcribed text and segments.
        """
        self._logger.debug("Transcribing audio, language=%s", language or "auto")
        client = self._client
        raw = client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
            response_format="verbose_json",
            language=language,
            timeout=timeout,
        )
        return to_verbose_response(raw)

    def translate(
        self,
        *,
        file: BinaryIO,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Translate audio file to English text.

        Uses Whisper's translation mode to produce English output regardless
        of source language. This is optimal for non-English audio when
        English output is desired.

        Args:
            file: Binary file-like object containing audio data.
            timeout: Optional request timeout override.

        Returns:
            VerboseResponse with translated English text and segments.
        """
        self._logger.debug("Translating audio to English")
        client = self._client
        raw = client.audio.translations.create(
            model="whisper-1",
            file=file,
            response_format="verbose_json",
            timeout=timeout,
        )
        return to_verbose_response(raw)

    def process(
        self,
        *,
        file: BinaryIO,
        task: WhisperTask = "transcribe",
        language: str | None = None,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Process audio file with specified task.

        Unified method that routes to transcribe or translate based on task.

        Args:
            file: Binary file-like object containing audio data.
            task: Either "transcribe" or "translate".
            language: Optional language hint (only used for transcribe task).
            timeout: Optional request timeout override.

        Returns:
            VerboseResponse with processed text and segments.
        """
        if task == "translate":
            return self.translate(file=file, timeout=timeout)
        return self.transcribe(file=file, language=language, timeout=timeout)


__all__ = ["OpenAISttClient"]
