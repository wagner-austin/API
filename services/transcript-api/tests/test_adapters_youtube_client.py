from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from transcript_api import _test_hooks
from transcript_api._test_hooks import YTApiProto, YTFetchedProto, YTListingProto
from transcript_api.adapters.youtube_client import (
    YouTubeTranscriptApiAdapter,
    _YTResourceProto,
)
from transcript_api.provider import (
    DirectTranscriptUnavailableError,
    TranscriptLanguageUnavailableError,
    TranscriptListingError,
    TranscriptTranslateUnavailableError,
)


class _Fetched:
    """A fake FetchedTranscript: carries the raw dicts the 1.x library
    converts back with ``to_raw_data``."""

    def __init__(self, raw: list[dict[str, JSONValue]]) -> None:
        self._raw = raw

    def to_raw_data(self) -> list[dict[str, JSONValue]]:
        return self._raw


class _NoTranscriptFoundError(Exception): ...


class _NotTranslatableError(Exception): ...


class _TranslationUnavailableError(Exception): ...


def _install(api: YTApiProto) -> None:
    """Point every YouTube hook at the fake API and this module's exceptions."""

    def _api_factory() -> YTApiProto:
        return api

    def _exc_factory() -> tuple[type[Exception], type[Exception], type[Exception]]:
        return (_NoTranscriptFoundError, RuntimeError, RuntimeError)

    def _translate_exc_factory() -> tuple[type[Exception], type[Exception]]:
        return (_NotTranslatableError, _TranslationUnavailableError)

    _test_hooks.yt_api_factory = _api_factory
    _test_hooks.yt_exceptions_factory = _exc_factory
    _test_hooks.yt_translate_exceptions_factory = _translate_exc_factory


def test_get_transcript_coerces_the_fetched_raw_data() -> None:
    class _API:
        def fetch(self, video_id: str, languages: list[str]) -> YTFetchedProto:
            assert video_id == "vid" and languages == ["en"]
            return _Fetched([{"text": "hello", "start": 0, "duration": 1.0}])

        def list(self, video_id: str) -> YTListingProto:
            raise RuntimeError

    _install(_API())
    adapter = YouTubeTranscriptApiAdapter()
    out = adapter.get_transcript("vid", ["en"])
    # The int start is coerced to float; the item arrives fully typed.
    assert out == [{"text": "hello", "start": 0.0, "duration": 1.0}]


def test_get_transcript_maps_direct_unavailable() -> None:
    class _API:
        def fetch(self, video_id: str, languages: list[str]) -> YTFetchedProto:
            raise _NoTranscriptFoundError("nope")

        def list(self, video_id: str) -> YTListingProto:
            raise RuntimeError

    _install(_API())
    adapter = YouTubeTranscriptApiAdapter()
    with pytest.raises(DirectTranscriptUnavailableError):
        _ = adapter.get_transcript("vid", ["en"])


class _Res:
    def __init__(self, raw: list[dict[str, JSONValue]]) -> None:
        self._raw = raw

    def fetch(self) -> YTFetchedProto:
        return _Fetched(self._raw)


def test_listing_finds_fetches_and_maps_translate_failures() -> None:
    class _Listing:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto:
            assert languages == ["en"]
            return _Res([{"text": "x", "start": 0.0, "duration": 1.0}])

        def translate(self, language: str) -> _YTResourceProto:
            raise _NotTranslatableError("no translate")

    class _API:
        def fetch(self, video_id: str, languages: list[str]) -> YTFetchedProto:
            return _Fetched([])

        def list(self, video_id: str) -> YTListingProto:
            return _Listing()

    _install(_API())
    adapter = YouTubeTranscriptApiAdapter()
    listing = adapter.list_transcripts("vid")
    res = listing.find_transcript(["en"])
    assert res is not None and res.fetch() == [{"text": "x", "start": 0.0, "duration": 1.0}]
    with pytest.raises(TranscriptTranslateUnavailableError):
        _ = listing.translate("en")


def test_find_transcript_maps_no_transcript_found() -> None:
    """The 1.x library RAISES when no transcript matches, where 0.x adapters
    read a None: the mapped error keeps the provider's fallback semantics."""

    class _Listing:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto:
            raise _NoTranscriptFoundError("nf")

        def translate(self, language: str) -> _YTResourceProto:
            return _Res([])

    class _API:
        def fetch(self, video_id: str, languages: list[str]) -> YTFetchedProto:
            return _Fetched([])

        def list(self, video_id: str) -> YTListingProto:
            return _Listing()

    _install(_API())
    adapter = YouTubeTranscriptApiAdapter()
    listing = adapter.list_transcripts("vid")
    with pytest.raises(TranscriptLanguageUnavailableError):
        _ = listing.find_transcript(["en"])


def test_translate_success_returns_a_fetchable_resource() -> None:
    class _Listing:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto:
            return _Res([])

        def translate(self, language: str) -> _YTResourceProto:
            return _Res([{"text": "ok", "start": 0.0, "duration": 1.0}])

    class _API:
        def fetch(self, video_id: str, languages: list[str]) -> YTFetchedProto:
            return _Fetched([])

        def list(self, video_id: str) -> YTListingProto:
            return _Listing()

    _install(_API())
    adapter = YouTubeTranscriptApiAdapter()
    assert adapter.get_transcript("vid", ["en"]) == []
    listing = adapter.list_transcripts("vid")
    res = listing.translate("en")
    assert res.fetch()[0]["text"] == "ok"


def test_list_transcripts_unavailable_maps_error() -> None:
    class _API:
        def fetch(self, video_id: str, languages: list[str]) -> YTFetchedProto:
            return _Fetched([])

        def list(self, video_id: str) -> YTListingProto:
            raise RuntimeError("disabled")

    _install(_API())
    adapter = YouTubeTranscriptApiAdapter()
    with pytest.raises(TranscriptListingError):
        _ = adapter.list_transcripts("vid")


@pytest.mark.parametrize(
    ("payload", "error_field"),
    [
        ({"text": 123, "start": 0.0, "duration": 1.0}, "text"),
        ({"text": "t", "start": "bad", "duration": 1.0}, "start"),
        ({"text": "t", "start": 0.0, "duration": "bad"}, "duration"),
    ],
)
def test_get_transcript_rejects_invalid_field_types(
    payload: dict[str, str | int | float],
    error_field: str,
) -> None:
    """Wrong-shaped vendor data raises rather than degrades."""
    out_dict: dict[str, JSONValue] = {}
    for k, v in payload.items():
        if isinstance(v, str | int | float):
            out_dict[k] = v

    class _BadAPI:
        def fetch(self, video_id: str, languages: list[str]) -> YTFetchedProto:
            assert video_id == "vid" and languages == ["en"]
            return _Fetched([out_dict])

        def list(self, video_id: str) -> YTListingProto:
            raise RuntimeError

    _install(_BadAPI())
    adapter = YouTubeTranscriptApiAdapter()
    with pytest.raises(JSONTypeError):
        _ = adapter.get_transcript("vid", ["en"])


def test_resource_fetch_validates_the_fetched_raw_data() -> None:
    """The listing path runs the same coercion as the direct path."""

    class _Listing:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto:
            return _Res([{"text": 5, "start": 0.0, "duration": 1.0}])

        def translate(self, language: str) -> _YTResourceProto:
            return _Res([])

    class _API:
        def fetch(self, video_id: str, languages: list[str]) -> YTFetchedProto:
            return _Fetched([])

        def list(self, video_id: str) -> YTListingProto:
            return _Listing()

    _install(_API())
    adapter = YouTubeTranscriptApiAdapter()
    listing = adapter.list_transcripts("vid")
    res = listing.find_transcript(["en"])
    if res is None:
        pytest.fail("expected a transcript resource from the listing")
    with pytest.raises(JSONTypeError):
        _ = res.fetch()
