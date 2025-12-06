from __future__ import annotations

from typing import BinaryIO

from ..stt_provider import STTClient
from ..types import OpenAIClientProto, VerboseResponseTD
from ..whisper_parse import to_verbose_dict


def _create_openai_client(*, api_key: str, timeout: float, max_retries: int) -> OpenAIClientProto:
    # Import and immediately assign to Protocol to bypass Any from untyped module
    mod = __import__("openai")
    # Call constructor and assign result to Protocol type
    client: OpenAIClientProto = mod.OpenAI(
        api_key=api_key, timeout=timeout, max_retries=max_retries
    )
    return client


class OpenAISttClient(STTClient):
    api_key: str
    timeout_seconds: float
    max_retries: int
    _client: OpenAIClientProto

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 900.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = self._make_client()

    def _make_client(self) -> OpenAIClientProto:
        return _create_openai_client(
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    def transcribe_verbose(self, *, file: BinaryIO, timeout: float | None) -> VerboseResponseTD:
        client = self._client
        raw = client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
            response_format="verbose_json",
            timeout=timeout,
        )
        return to_verbose_dict(raw)


__all__ = ["OpenAISttClient"]
