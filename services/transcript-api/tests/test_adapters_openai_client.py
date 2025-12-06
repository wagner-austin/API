from __future__ import annotations

import io
import sys
from types import ModuleType
from typing import Protocol

import pytest

from transcript_api.adapters.openai_client import OpenAISttClient
from transcript_api.types import VerboseResponseTD


class _FakeAudio:
    class _Transcriptions:
        def create(
            self, *, model: str, file: io.BytesIO, response_format: str, timeout: float | None
        ) -> VerboseResponseTD:
            assert model == "whisper-1" and response_format == "verbose_json"
            return {"text": "", "segments": [{"text": "ok", "start": 0.0, "end": 0.1}]}

    def __init__(self) -> None:
        self.transcriptions = self._Transcriptions()


class _FakeClient:
    def __init__(self, *, api_key: str, timeout: float, max_retries: int) -> None:
        self.audio = _FakeAudio()


def test_openai_stt_client_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    def _factory(*, api_key: str, timeout: float, max_retries: int) -> _FakeClient:
        return _FakeClient(api_key=api_key, timeout=timeout, max_retries=max_retries)

    class _OpenAIFactory(Protocol):
        def __call__(self, *, api_key: str, timeout: float, max_retries: int) -> _FakeClient: ...

    class _OpenAIModule(ModuleType):
        OpenAI: _OpenAIFactory

    mod = _OpenAIModule("openai")

    # Expose a factory with the required keyword-only signature
    def _openai_ctor(*, api_key: str, timeout: float, max_retries: int) -> _FakeClient:
        return _factory(api_key=api_key, timeout=timeout, max_retries=max_retries)

    mod.OpenAI = _openai_ctor
    monkeypatch.setitem(sys.modules, "openai", mod)

    client = OpenAISttClient(api_key="k", timeout_seconds=10.0, max_retries=1)
    out = client.transcribe_verbose(file=io.BytesIO(b"x"), timeout=5.0)
    assert "segments" in out
