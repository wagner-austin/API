from __future__ import annotations

import sys
from typing import Protocol

import pytest

from transcript_api.startup import make_app_from_env
from transcript_api.types import JsonValue


class _FileLike(Protocol):
    pass


class _TracebackLike(Protocol):
    pass


class _FakeOpenAIClient:
    def __init__(self, *, api_key: str, timeout: float, max_retries: int) -> None:
        class _Transcriptions:
            def create(
                self,
                *,
                model: str,
                file: _FileLike,
                response_format: str,
                timeout: float | None,
            ) -> dict[str, JsonValue]:
                return {}

        class _Audio:
            def __init__(self) -> None:
                self.transcriptions = _Transcriptions()

        self.audio = _Audio()


class _FakeOpenAIMod:
    @staticmethod
    def _factory(*, api_key: str, timeout: float, max_retries: int) -> _FakeOpenAIClient:
        return _FakeOpenAIClient(api_key=api_key, timeout=timeout, max_retries=max_retries)

    OpenAI = staticmethod(_factory)


class _FakeYTResource:
    def fetch(self) -> list[dict[str, JsonValue]]:
        return []


class _FakeYTListing:
    def find_transcript(self, languages: list[str]) -> _FakeYTResource:
        return _FakeYTResource()

    def translate(self, language: str) -> _FakeYTResource:
        return _FakeYTResource()


class _FakeYTApi:
    @staticmethod
    def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JsonValue]]:
        return []

    @staticmethod
    def list_transcripts(video_id: str) -> _FakeYTListing:
        return _FakeYTListing()


class _FakeYTMod:
    YouTubeTranscriptApi = _FakeYTApi
    NoTranscriptFound = KeyError
    TranscriptsDisabled = RuntimeError
    VideoUnavailable = RuntimeError


class _FakeYDL:
    def __init__(self, opts: dict[str, JsonValue]) -> None:
        self._opts = opts

    def __enter__(self) -> _FakeYDL:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: _TracebackLike | None,
    ) -> None:
        return None

    def extract_info(self, url: str, download: bool) -> dict[str, JsonValue]:
        return {}

    def prepare_filename(self, info: dict[str, JsonValue]) -> str:
        return "audio.m4a"


class _FakeYtDlpMod:
    @staticmethod
    def _factory(opts: dict[str, JsonValue]) -> _FakeYDL:
        return _FakeYDL(opts)

    YoutubeDL = staticmethod(_factory)


def test_make_app_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", _FakeOpenAIMod())
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", _FakeYTMod())
    monkeypatch.setitem(sys.modules, "yt_dlp", _FakeYtDlpMod())
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    app = make_app_from_env()
    # Basic sanity: app instance created with routes attached - access router directly
    _ = app.router
