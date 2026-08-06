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
class _YTFetchedProto(Protocol):
    """A fetched transcript (youtube_transcript_api >= 1.x).

    ``to_raw_data`` is the library's own conversion back to the plain dicts
    this adapter validates and coerces to RawTranscriptItem.
    """

    def to_raw_data(self) -> list[dict[str, JSONValue]]: ...


@runtime_checkable
class _YTResourceProto(Protocol):
    def fetch(self) -> _YTFetchedProto: ...


@runtime_checkable
class _YTListingProto(Protocol):
    def find_transcript(self, languages: list[str]) -> _YTResourceProto: ...
    def translate(self, language: str) -> _YTResourceProto: ...


@runtime_checkable
class _GetTranscriptFn(Protocol):
    def __call__(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]: ...


@runtime_checkable
class _ListTranscriptsFn(Protocol):
    def __call__(self, video_id: str) -> _YTListingProto: ...


def _coerce_items(raw_items: list[dict[str, JSONValue]]) -> list[RawTranscriptItem]:
    """Validate and coerce the library's plain dicts to RawTranscriptItem.

    The external library returns untyped dicts; every field is checked
    before use and a wrong shape raises rather than degrades.

    Args:
        raw_items: The library's ``to_raw_data`` output.

    Returns:
        The coerced items.

    Raises:
        JSONTypeError: When an item carries a field of the wrong type.
    """
    coerced: list[RawTranscriptItem] = []
    for item in raw_items:
        text = item.get("text", "")
        start = item.get("start", 0.0)
        duration = item.get("duration", 0.0)

        if not isinstance(text, str):
            raise JSONTypeError("Expected string for 'text' in transcript item")
        if not isinstance(start, int | float):
            raise JSONTypeError("Expected int or float for 'start' in transcript item")
        if not isinstance(duration, int | float):
            raise JSONTypeError("Expected int or float for 'duration' in transcript item")

        coerced.append({"text": text, "start": float(start), "duration": float(duration)})
    return coerced


class _YTResource(TranscriptResource):
    def __init__(self, inner: _YTResourceProto) -> None:
        self._inner = inner

    def fetch(self) -> list[RawTranscriptItem]:
        return _coerce_items(self._inner.fetch().to_raw_data())


class _YTListing(TranscriptListing):
    def __init__(self, inner: _YTListingProto) -> None:
        self._inner = inner

    def find_transcript(self, languages: list[str]) -> TranscriptResource | None:
        no_transcript_found = _test_hooks.yt_exceptions_factory()[0]
        try:
            res = self._inner.find_transcript(languages)
        except no_transcript_found as exc:
            raise TranscriptLanguageUnavailableError(str(exc)) from None
        return _YTResource(res)

    def translate(self, language: str) -> TranscriptResource:
        translate_exceptions = _test_hooks.yt_translate_exceptions_factory()
        try:
            res = self._inner.translate(language)
        except translate_exceptions as exc:
            get_logger(__name__).info("translate failed: %s", exc)
            raise TranscriptTranslateUnavailableError(str(exc)) from None
        return _YTResource(res)


@runtime_checkable
class _YTApiProto(Protocol):
    """The 1.x instance API: ``fetch``/``list`` replace the 0.x statics."""

    def fetch(self, video_id: str, languages: list[str]) -> _YTFetchedProto: ...
    def list(self, video_id: str) -> _YTListingProto: ...


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
        return _coerce_items(yt_api.fetch(video_id, languages=languages).to_raw_data())
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
        return yt_api.list(video_id)
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
