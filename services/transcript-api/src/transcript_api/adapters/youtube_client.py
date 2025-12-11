from __future__ import annotations

from typing import Protocol, runtime_checkable

from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.logging import get_logger

from .. import _test_hooks
from ..provider import (
    TranscriptLanguageUnavailableError,
    TranscriptListing,
    TranscriptResource,
    TranscriptTranslateUnavailableError,
    YouTubeTranscriptClient,
)
from ..types import RawTranscriptItem


@runtime_checkable
class _YTResourceProto(Protocol):
    def fetch(self) -> list[RawTranscriptItem]: ...


@runtime_checkable
class _YTListingProto(Protocol):
    def find_transcript(self, languages: list[str]) -> _YTResourceProto | None: ...
    def translate(self, language: str) -> _YTResourceProto: ...


@runtime_checkable
class _GetTranscriptFn(Protocol):
    def __call__(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]: ...


@runtime_checkable
class _ListTranscriptsFn(Protocol):
    def __call__(self, video_id: str) -> _YTListingProto: ...


class _YTResource(TranscriptResource):
    def __init__(self, inner: _YTResourceProto) -> None:
        self._inner = inner

    def fetch(self) -> list[RawTranscriptItem]:
        return self._inner.fetch()


class _YTListing(TranscriptListing):
    def __init__(self, inner: _YTListingProto) -> None:
        self._inner = inner

    def find_transcript(self, languages: list[str]) -> TranscriptResource | None:
        try:
            res = self._inner.find_transcript(languages)
        except KeyError as exc:
            raise TranscriptLanguageUnavailableError(str(exc)) from None
        return _YTResource(res) if res is not None else None

    def translate(self, language: str) -> TranscriptResource:
        try:
            res = self._inner.translate(language)
        except (RuntimeError, JSONTypeError) as exc:
            get_logger(__name__).info("translate failed: %s", exc)
            raise TranscriptTranslateUnavailableError(str(exc)) from None
        return _YTResource(res)


@runtime_checkable
class _YTApiProto(Protocol):
    @staticmethod
    def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]: ...
    @staticmethod
    def list_transcripts(video_id: str) -> _YTListingProto: ...


def _create_yt_api() -> _YTApiProto:
    # Use hook for dependency injection
    return _test_hooks.yt_api_factory()


def _get_yt_transcript_exceptions() -> tuple[type[Exception], type[Exception], type[Exception]]:
    """Get youtube_transcript_api exception classes via hook."""
    return _test_hooks.yt_exceptions_factory()


def _yt_get_transcript(video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
    yt_api = _create_yt_api()
    exc_classes = _get_yt_transcript_exceptions()

    try:
        # The external library returns list[dict[str, Any]] - we validate and coerce
        raw_transcript_result = yt_api.get_transcript(video_id, languages=languages)
        coerced_transcript: list[RawTranscriptItem] = []
        for item in raw_transcript_result:
            text = item.get("text", "")
            start = item.get("start", 0.0)
            duration = item.get("duration", 0.0)

            if not isinstance(text, str):
                raise JSONTypeError("Expected string for 'text' in transcript item")
            if not isinstance(start, int | float):
                raise JSONTypeError("Expected int or float for 'start' in transcript item")
            if not isinstance(duration, int | float):
                raise JSONTypeError("Expected int or float for 'duration' in transcript item")

            coerced_transcript.append(
                {"text": text, "start": float(start), "duration": float(duration)}
            )
        return coerced_transcript
    except exc_classes as exc:
        from ..provider import DirectTranscriptUnavailableError

        raise DirectTranscriptUnavailableError(str(exc)) from None


def _get_yt_listing_exceptions() -> tuple[type[Exception], type[Exception]]:
    """Get youtube_transcript_api listing exception classes via hook."""
    exc = _test_hooks.yt_exceptions_factory()
    # Return (VideoUnavailable, TranscriptsDisabled)
    return (exc[2], exc[1])


def _yt_list_transcripts(video_id: str) -> _YTListingProto:
    yt_api = _create_yt_api()
    exc_classes = _get_yt_listing_exceptions()

    try:
        return yt_api.list_transcripts(video_id)
    except exc_classes as exc:
        from ..provider import TranscriptListingError

        raise TranscriptListingError(str(exc)) from None


class YouTubeTranscriptApiAdapter(YouTubeTranscriptClient):
    """Adapter over youtube_transcript_api with strict error mapping without vendor typing.

    Imports are runtime-only; typing uses Protocols and TypedDicts to prevent Any.
    """

    def get_transcript(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
        gt: _GetTranscriptFn = _yt_get_transcript
        return gt(video_id, languages=languages)

    def list_transcripts(self, video_id: str) -> TranscriptListing:
        lt: _ListTranscriptsFn = _yt_list_transcripts
        listing_obj = lt(video_id)
        return _YTListing(listing_obj)
