from __future__ import annotations

import logging

import pytest
from platform_core.errors import AppError

from transcript_api.provider import (
    DirectTranscriptUnavailableError,
    TranscriptLanguageUnavailableError,
    TranscriptListing,
    TranscriptListingError,
    TranscriptResource,
    TranscriptTranslateUnavailableError,
    YouTubeTranscriptProvider,
    _as_float,
    _coerce_raw_items,
)
from transcript_api.types import RawTranscriptItem, TranscriptOptions


class _FakeResource:
    def __init__(self, data: list[dict[str, str | int | float]]) -> None:
        self._data = data

    def fetch(self) -> list[RawTranscriptItem]:
        out: list[RawTranscriptItem] = []
        for item in self._data:
            text_v = str(item.get("text", ""))
            start_val = item.get("start", 0.0)
            dur_val = item.get("duration", 0.0)
            start_v = _as_float(start_val if isinstance(start_val, (str, int, float)) else 0.0)
            dur_v = _as_float(dur_val if isinstance(dur_val, (str, int, float)) else 0.0)
            typed: RawTranscriptItem = {"text": text_v, "start": start_v, "duration": dur_v}
            out.append(typed)
        return out


class _ListingAlwaysFound:
    def __init__(self, resource: TranscriptResource) -> None:
        self._resource = resource

    def find_transcript(self, languages: list[str]) -> TranscriptResource:
        assert languages
        return self._resource

    def translate(self, language: str) -> TranscriptResource:
        return self._resource


class _ListingMissingLanguage:
    def __init__(self, resource: TranscriptResource) -> None:
        self._resource = resource

    def find_transcript(self, languages: list[str]) -> TranscriptResource:
        raise TranscriptLanguageUnavailableError("nf")

    def translate(self, language: str) -> TranscriptResource:
        return self._resource


class _ListingTranslateUnavailable:
    def __init__(self) -> None:
        self._resource = _FakeResource(
            [{"text": "unused", "start": 0.0, "duration": 1.0}],
        )

    def find_transcript(self, languages: list[str]) -> TranscriptResource:
        raise TranscriptLanguageUnavailableError("nf")

    def translate(self, language: str) -> TranscriptResource:
        raise TranscriptTranslateUnavailableError("no translate")


class _ClientDirectOnly:
    def __init__(
        self,
        data: list[dict[str, str | int | float] | str | int | float | None],
    ) -> None:
        self._data_raw = data

    def get_transcript(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
        assert video_id == "vid"
        assert languages == ["en"]
        # Return raw data - provider will coerce it
        return _coerce_raw_items(self._data_raw)

    def list_transcripts(self, video_id: str) -> TranscriptListing:
        raise AssertionError("list_transcripts should not be called in direct-only client")


class _ClientWithFallback:
    def __init__(self, listing: TranscriptListing) -> None:
        self._listing = listing

    def get_transcript(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
        raise DirectTranscriptUnavailableError(f"{video_id} no direct")

    def list_transcripts(self, video_id: str) -> TranscriptListing:
        return self._listing


class _ClientListingError:
    def get_transcript(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
        raise DirectTranscriptUnavailableError("no direct")

    def list_transcripts(self, video_id: str) -> TranscriptListing:
        raise TranscriptListingError("listing failed")


class _ClientListingUnexpected:
    def get_transcript(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
        raise DirectTranscriptUnavailableError("no direct")

    def list_transcripts(self, video_id: str) -> TranscriptListing:
        class _BadListing:
            def find_transcript(self, languages: list[str]) -> TranscriptResource:
                raise AssertionError("should not be called")

            def translate(self, language: str) -> TranscriptResource:
                raise AssertionError("should not be called")

        return _BadListing()


def test_as_float_edges_provider() -> None:
    assert _as_float(5) == 5.0
    assert _as_float(3.25) == 3.25
    assert _as_float("7.5") == 7.5
    assert _as_float("bad") == 0.0
    assert _as_float(None) == 0.0
    assert _as_float({"k": 1}) == 0.0


def test_coerce_raw_items_typing_and_non_list() -> None:
    empty_dict: dict[str, str | int | float] = {}
    with pytest.raises(AppError):
        _ = _coerce_raw_items(empty_dict)

    items: list[dict[str, str | int | float] | str | int | float | None] = [
        {"text": " hello ", "start": "1.0", "duration": "2.0"},
        {"text": "", "start": 0, "duration": 0},
        123,
    ]
    out = _coerce_raw_items(items)
    assert out
    first = out[0]
    assert first["text"].strip() == "hello"
    assert first["start"] == 1.0 and first["duration"] == 2.0


def test_fetch_uses_direct_transcript_when_available() -> None:
    client = _ClientDirectOnly(
        [
            {"text": "hello", "start": "0", "duration": 1},
            {"text": " ", "start": 1, "duration": 1},
        ]
    )
    prov = YouTubeTranscriptProvider(client)
    opts = TranscriptOptions(preferred_langs=["en"])
    out = prov.fetch("vid", opts)
    assert len(out) == 1
    assert out[0]["text"] == "hello" and out[0]["start"] == 0.0


def test_fetch_falls_back_to_listing_when_direct_unavailable() -> None:
    resource = _FakeResource(
        [
            {"text": "ok", "start": 0, "duration": 1},
            {"text": "   ", "start": 1, "duration": 1},
        ]
    )
    listing = _ListingAlwaysFound(resource)
    client = _ClientWithFallback(listing)
    prov = YouTubeTranscriptProvider(client)
    opts = TranscriptOptions(preferred_langs=["en"])
    out = prov.fetch("vid", opts)
    assert out and out[0]["text"] == "ok"


def test_fetch_fallback_uses_translate_when_choose_raises() -> None:
    resource = _FakeResource(
        [
            {"text": "t", "start": "0", "duration": "1.5"},
        ]
    )
    listing = _ListingMissingLanguage(resource)
    client = _ClientWithFallback(listing)
    prov = YouTubeTranscriptProvider(client)
    opts = TranscriptOptions(preferred_langs=["en"])
    out = prov.fetch("vid", opts)
    assert len(out) == 1
    assert out[0]["text"] == "t" and out[0]["duration"] == 1.5


def test_fallback_translate_unavailable_raises_user_input() -> None:
    client = _ClientWithFallback(_ListingTranslateUnavailable())
    prov = YouTubeTranscriptProvider(client)
    opts = TranscriptOptions(preferred_langs=["en"])
    with pytest.raises(AppError):
        _ = prov.fetch("vid", opts)


def test_list_transcripts_error_and_unexpected_type() -> None:
    prov_err = YouTubeTranscriptProvider(_ClientListingError())
    with pytest.raises(AppError):
        _ = prov_err._list_transcripts("vid")

    prov_bad = YouTubeTranscriptProvider(_ClientListingUnexpected())
    listing = prov_bad._list_transcripts("vid")
    # Verify listing satisfies TranscriptListing protocol by checking methods exist
    assert callable(listing.find_transcript)
    assert callable(listing.translate)


class _ClientListingNone:
    def __init__(self, listing: TranscriptListing) -> None:
        self._listing = listing

    def get_transcript(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
        raise DirectTranscriptUnavailableError("no direct")

    def list_transcripts(self, video_id: str) -> TranscriptListing:
        return self._listing


def test_fallback_listing_raises_when_no_transcript_anywhere() -> None:
    class _ListingNone(TranscriptListing):
        def find_transcript(self, languages: list[str]) -> TranscriptResource | None:
            return None

        def translate(self, language: str) -> TranscriptResource:
            return _FakeResource(
                [
                    {"text": "", "start": 0.0, "duration": 1.0},
                ]
            )

    listing = _ListingNone()
    client = _ClientListingNone(listing)
    prov = YouTubeTranscriptProvider(client)
    opts = TranscriptOptions(preferred_langs=["en"])
    with pytest.raises(AppError):
        _ = prov.fetch("vid", opts)


def test_choose_transcript_none_and_language_error() -> None:
    class _ListingNone(TranscriptListing):
        def find_transcript(self, languages: list[str]) -> TranscriptResource | None:
            return None

        def translate(self, language: str) -> TranscriptResource:
            return _FakeResource(
                [
                    {"text": "x", "start": 0, "duration": 1},
                ]
            )

    listing_none = _ListingNone()
    prov = YouTubeTranscriptProvider(_ClientWithFallback(listing_none))
    chosen = prov._choose_transcript(listing_none, ["en"])
    assert chosen is None

    class _ListingError:
        def find_transcript(self, languages: list[str]) -> TranscriptResource:
            raise TranscriptLanguageUnavailableError("nf")

        def translate(self, language: str) -> TranscriptResource:
            return _FakeResource(
                [
                    {"text": "y", "start": 0, "duration": 1},
                ]
            )

    listing_err = _ListingError()
    with pytest.raises(AppError):
        _ = prov._choose_transcript(listing_err, ["en"])


logger = logging.getLogger(__name__)
