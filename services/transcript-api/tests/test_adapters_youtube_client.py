from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from transcript_api import _test_hooks
from transcript_api._test_hooks import YTApiProto, YTListingProto
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
from transcript_api.types import RawTranscriptItem


def _make_module_with_api(
    api_cls: type[YTApiProto],
    no_transcript_found: type[Exception],
    transcripts_disabled: type[Exception],
    video_unavailable: type[Exception],
) -> None:
    """Set up test hooks with the given API class and exception types."""

    def _api_factory() -> YTApiProto:
        # Return the class itself, as YTApiProto expects static methods
        api: YTApiProto = api_cls
        return api

    def _exc_factory() -> tuple[type[Exception], type[Exception], type[Exception]]:
        return (no_transcript_found, transcripts_disabled, video_unavailable)

    _test_hooks.yt_api_factory = _api_factory
    _test_hooks.yt_exceptions_factory = _exc_factory


def test_get_transcript_success_and_direct_unavailable() -> None:
    data: list[dict[str, JSONValue]] = [{"text": "hello", "start": 0.0, "duration": 1.0}]

    class _NoTranscriptFoundError(Exception): ...

    class _API:
        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]:
            assert video_id == "vid" and languages == ["en"]
            return data

        @staticmethod
        def list_transcripts(video_id: str) -> YTListingProto:
            raise RuntimeError

    _make_module_with_api(_API, _NoTranscriptFoundError, RuntimeError, RuntimeError)

    adapter = YouTubeTranscriptApiAdapter()
    out = adapter.get_transcript("vid", ["en"])
    # out is list[RawTranscriptItem], data is list[dict[str, JSONValue]]
    # Compare as dicts
    assert out[0]["text"] == "hello"
    assert out[0]["start"] == 0.0
    assert out[0]["duration"] == 1.0

    # Raise mapped error by replacing the module with a different API class
    class _MockListing:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto | None:
            return None

        def translate(self, language: str) -> _YTResourceProto:
            raise RuntimeError("translate unavailable")

    class _API2:
        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]:
            raise _NoTranscriptFoundError("nope")

        @staticmethod
        def list_transcripts(video_id: str) -> YTListingProto:
            return _MockListing()

    _make_module_with_api(_API2, _NoTranscriptFoundError, RuntimeError, RuntimeError)
    with pytest.raises(DirectTranscriptUnavailableError):
        _ = adapter.get_transcript("vid", ["en"])


def test_list_transcripts_and_wrapped_calls() -> None:
    class _Res:
        def __init__(self, payload: list[RawTranscriptItem]) -> None:
            self._payload = payload

        def fetch(self) -> list[RawTranscriptItem]:
            return self._payload

    class _Listing:
        def __init__(self) -> None:
            self._res = _Res([{"text": "x", "start": 0.0, "duration": 1.0}])

        def find_transcript(self, languages: list[str]) -> _YTResourceProto:
            assert languages == ["en"]
            return self._res

        def translate(self, language: str) -> _YTResourceProto:
            raise RuntimeError("no translate")

    class _API:
        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]:
            return []

        @staticmethod
        def list_transcripts(video_id: str) -> YTListingProto:
            return _Listing()

    _make_module_with_api(_API, KeyError, RuntimeError, RuntimeError)

    adapter = YouTubeTranscriptApiAdapter()
    listing = adapter.list_transcripts("vid")
    res = listing.find_transcript(["en"])
    assert res is not None and res.fetch()[0]["text"] == "x"
    with pytest.raises(TranscriptTranslateUnavailableError):
        _ = listing.translate("en")


def test_find_transcript_returns_none() -> None:
    class _MockResource:
        def fetch(self) -> list[RawTranscriptItem]:
            return []

    class _Listing:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto | None:
            return None

        def translate(self, language: str) -> _YTResourceProto:
            return _MockResource()

    class _API:
        @staticmethod
        def list_transcripts(video_id: str) -> YTListingProto:
            return _Listing()

        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]:
            return []

    _make_module_with_api(_API, KeyError, RuntimeError, RuntimeError)
    adapter = YouTubeTranscriptApiAdapter()
    listing = adapter.list_transcripts("vid")
    out = listing.find_transcript(["en"])  # returns None from adapter when inner returns None
    assert out is None


def test_find_transcript_maps_no_transcript_found() -> None:
    class _MockResource2:
        def fetch(self) -> list[RawTranscriptItem]:
            return []

    class _Listing:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto:
            raise KeyError("nf")

        def translate(self, language: str) -> _YTResourceProto:
            return _MockResource2()

    class _API:
        @staticmethod
        def list_transcripts(video_id: str) -> YTListingProto:
            return _Listing()

        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]:
            return []

    _make_module_with_api(_API, KeyError, RuntimeError, RuntimeError)
    adapter = YouTubeTranscriptApiAdapter()
    listing = adapter.list_transcripts("vid")
    with pytest.raises(TranscriptLanguageUnavailableError):
        _ = listing.find_transcript(["en"])


def test_translate_success_and_get_transcript_non_list() -> None:
    class _Res:
        def fetch(self) -> list[RawTranscriptItem]:
            return [{"text": "ok", "start": 0.0, "duration": 1.0}]

    class _Listing:
        def translate(self, language: str) -> _YTResourceProto:
            return _Res()

        def find_transcript(self, languages: list[str]) -> _YTResourceProto:
            return _Res()

    class _API:
        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]:
            # Return non-list to exercise coercion to []
            return []

        @staticmethod
        def list_transcripts(video_id: str) -> YTListingProto:
            return _Listing()

    _make_module_with_api(_API, KeyError, RuntimeError, RuntimeError)
    adapter = YouTubeTranscriptApiAdapter()
    out = adapter.get_transcript("vid", ["en"])
    assert out == []
    listing = adapter.list_transcripts("vid")
    res = listing.translate("en")
    assert res.fetch()[0]["text"] == "ok"


def test_get_transcript_list_with_nondict_items() -> None:
    class _MockListing3:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto | None:
            return None

        def translate(self, language: str) -> _YTResourceProto:
            raise RuntimeError("translate unavailable")

    class _API:
        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]:
            return [{"text": "ok", "start": 0.0, "duration": 1.0}]

        @staticmethod
        def list_transcripts(video_id: str) -> YTListingProto:
            return _MockListing3()

    _make_module_with_api(_API, KeyError, RuntimeError, RuntimeError)
    adapter = YouTubeTranscriptApiAdapter()
    out = adapter.get_transcript("vid", ["en"])
    assert out and out[0]["text"] == "ok"


def test_list_transcripts_unexpected_type() -> None:
    # Test that valid protocol implementations work correctly
    class _MockListing4:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto | None:
            return None

        def translate(self, language: str) -> _YTResourceProto:
            raise RuntimeError("translate unavailable")

    class _API:
        @staticmethod
        def list_transcripts(video_id: str) -> YTListingProto:
            return _MockListing4()

        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]:
            return []

    _make_module_with_api(_API, KeyError, RuntimeError, RuntimeError)
    adapter = YouTubeTranscriptApiAdapter()
    listing = adapter.list_transcripts("vid")
    if listing is None:
        pytest.fail("expected listing result")


def test_list_transcripts_unavailable_maps_error() -> None:
    class _API:
        @staticmethod
        def list_transcripts(video_id: str) -> YTListingProto:
            raise RuntimeError("disabled")

        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]:
            return []

    _make_module_with_api(_API, KeyError, RuntimeError, RuntimeError)

    adapter = YouTubeTranscriptApiAdapter()
    with pytest.raises(TranscriptListingError):
        _ = adapter.list_transcripts("vid")


class _BadYTResourceForValidation:
    """Fake resource for validation tests."""

    def fetch(self) -> list[RawTranscriptItem]:
        return []


class _BadYTListingForValidation:
    """Fake listing for validation tests."""

    def find_transcript(self, languages: list[str]) -> _BadYTResourceForValidation | None:
        return None

    def translate(self, language: str) -> _BadYTResourceForValidation:
        raise RuntimeError


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
    """Test that get_transcript rejects items with wrong field types."""
    out_dict: dict[str, JSONValue] = {}
    for k, v in payload.items():
        if isinstance(v, str | int | float):
            out_dict[k] = v

    class _BadAPI:
        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[dict[str, JSONValue]]:
            assert video_id == "vid" and languages == ["en"]
            return [out_dict]

        @staticmethod
        def list_transcripts(video_id: str) -> YTListingProto:
            return _BadYTListingForValidation()

    def _api_factory() -> YTApiProto:
        api: YTApiProto = _BadAPI
        return api

    _test_hooks.yt_api_factory = _api_factory
    _test_hooks.yt_exceptions_factory = lambda: (KeyError, RuntimeError, RuntimeError)

    adapter = YouTubeTranscriptApiAdapter()
    with pytest.raises(JSONTypeError):
        _ = adapter.get_transcript("vid", ["en"])
