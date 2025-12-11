from __future__ import annotations

import io

from transcript_api import _test_hooks
from transcript_api.adapters.openai_client import OpenAISttClient
from transcript_api.types import (
    OpenAIClientProto,
    SupportsToDictRecursive,
    _AudioProto,
    _BinaryFileProto,
    _TranscriptionsProto,
)


class _FakeTranscriptionResult:
    """Fake transcription result implementing SupportsToDictRecursive."""

    def to_dict_recursive(
        self,
    ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
        return {"text": "", "segments": [{"text": "ok", "start": 0.0, "end": 0.1}]}


class _FakeTranscriptions:
    """Fake transcriptions implementing _TranscriptionsProto."""

    def create(
        self,
        *,
        model: str,
        file: _BinaryFileProto,
        response_format: str,
        timeout: float | None,
    ) -> SupportsToDictRecursive:
        assert model == "whisper-1" and response_format == "verbose_json"
        return _FakeTranscriptionResult()


class _FakeAudio:
    """Fake audio implementing _AudioProto."""

    @property
    def transcriptions(self) -> _TranscriptionsProto:
        return _FakeTranscriptions()


class _FakeClient:
    """Fake client implementing OpenAIClientProto."""

    @property
    def audio(self) -> _AudioProto:
        return _FakeAudio()


def test_openai_stt_client_integration() -> None:
    def _factory(*, api_key: str, timeout: float, max_retries: int) -> OpenAIClientProto:
        return _FakeClient()

    _test_hooks.openai_client_factory = _factory

    client = OpenAISttClient(api_key="k", timeout_seconds=10.0, max_retries=1)
    out = client.transcribe_verbose(file=io.BytesIO(b"x"), timeout=5.0)
    assert "segments" in out
