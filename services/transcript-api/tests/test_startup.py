from __future__ import annotations

from platform_core.config import _test_hooks as platform_hooks
from platform_core.json_utils import JSONValue
from platform_core.testing import make_fake_env

from transcript_api import _test_hooks
from transcript_api._test_hooks import YTApiProto
from transcript_api.startup import make_app_from_env
from transcript_api.types import (
    OpenAIClientProto,
    RawTranscriptItem,
    SupportsToDictRecursive,
    YtDlpProto,
    _AudioProto,
    _BinaryFileProto,
    _TracebackProto,
    _TranscriptionsProto,
)


class _FakeTranscriptionResult:
    """Fake transcription result implementing SupportsToDictRecursive."""

    def to_dict_recursive(
        self,
    ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
        return {"text": "", "segments": []}


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
        return _FakeTranscriptionResult()


class _FakeAudio:
    @property
    def transcriptions(self) -> _TranscriptionsProto:
        return _FakeTranscriptions()


class _FakeOpenAIClient:
    @property
    def audio(self) -> _AudioProto:
        return _FakeAudio()


class _FakeYTResource:
    def fetch(self) -> list[RawTranscriptItem]:
        return []


class _FakeYTListing:
    def find_transcript(self, languages: list[str]) -> _FakeYTResource | None:
        return _FakeYTResource()

    def translate(self, language: str) -> _FakeYTResource:
        return _FakeYTResource()


class _FakeYTApi:
    @staticmethod
    def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]:
        return []

    @staticmethod
    def list_transcripts(video_id: str) -> _FakeYTListing:
        return _FakeYTListing()


class _FakeYDL:
    def __init__(self, opts: dict[str, JSONValue]) -> None:
        self._opts = opts

    def __enter__(self) -> YtDlpProto:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: _TracebackProto | None,
    ) -> None:
        return None

    def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
        return {}

    def prepare_filename(self, info: dict[str, JSONValue]) -> str:
        return "audio.m4a"


def test_make_app_from_env() -> None:
    # Set environment variables
    platform_hooks.get_env = make_fake_env({"OPENAI_API_KEY": "k"})

    # Set up hooks for OpenAI client
    def _openai_factory(*, api_key: str, timeout: float, max_retries: int) -> OpenAIClientProto:
        return _FakeOpenAIClient()

    _test_hooks.openai_client_factory = _openai_factory

    # Set up hooks for YouTube API
    def _yt_api_factory() -> YTApiProto:
        api: YTApiProto = _FakeYTApi
        return api

    def _yt_exc_factory() -> tuple[type[Exception], type[Exception], type[Exception]]:
        return (KeyError, RuntimeError, RuntimeError)

    _test_hooks.yt_api_factory = _yt_api_factory
    _test_hooks.yt_exceptions_factory = _yt_exc_factory

    # Set up hooks for yt-dlp
    def _yt_dlp_factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _FakeYDL(opts)

    _test_hooks.yt_dlp_factory = _yt_dlp_factory

    app = make_app_from_env()
    # Basic sanity: app instance created with routes attached - access router directly
    _ = app.router
