from __future__ import annotations

import sys
from types import ModuleType
from typing import Protocol

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from transcript_api.adapters.youtube_client import (
    YouTubeTranscriptApiAdapter,
    _YTListingProto,
    _YTResourceProto,
)
from transcript_api.provider import (
    DirectTranscriptUnavailableError,
    TranscriptLanguageUnavailableError,
    TranscriptListingError,
    TranscriptTranslateUnavailableError,
)
from transcript_api.types import RawTranscriptItem


class _ApiProto(Protocol):
    @staticmethod
    def get_transcript(
        video_id: str,
        languages: list[str],
    ) -> (
        list[RawTranscriptItem] | list[dict[str, JSONValue]] | list[int | dict[str, JSONValue]]
    ): ...

    @staticmethod
    def list_transcripts(video_id: str) -> _YTListingProto | str: ...


class _YTModule(ModuleType):
    YouTubeTranscriptApi: _ApiProto
    NoTranscriptFound: type[Exception]
    TranscriptsDisabled: type[Exception]
    VideoUnavailable: type[Exception]


def test_get_transcript_success_and_direct_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    data: list[RawTranscriptItem] = [{"text": "hello", "start": 0.0, "duration": 1.0}]

    class _API:
        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
            assert video_id == "vid" and languages == ["en"]
            return data

        @staticmethod
        def list_transcripts(video_id: str) -> _YTListingProto:
            raise RuntimeError

    class _NoTranscriptFoundError(Exception): ...

    mod = _YTModule("youtube_transcript_api")
    mod.YouTubeTranscriptApi = _API
    mod.NoTranscriptFound = _NoTranscriptFoundError
    mod.TranscriptsDisabled = RuntimeError
    mod.VideoUnavailable = RuntimeError
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", mod)

    adapter = YouTubeTranscriptApiAdapter()
    out = adapter.get_transcript("vid", ["en"])
    assert out == data

    # Raise mapped error by replacing the module with a different API class
    class _MockListing:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto | None:
            return None

        def translate(self, language: str) -> _YTResourceProto:
            raise RuntimeError("translate unavailable")

    class _API2:
        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
            raise _NoTranscriptFoundError("nope")

        @staticmethod
        def list_transcripts(video_id: str) -> _YTListingProto:
            return _MockListing()

    mod2 = _YTModule("youtube_transcript_api")
    mod2.YouTubeTranscriptApi = _API2
    mod2.NoTranscriptFound = _NoTranscriptFoundError
    mod2.TranscriptsDisabled = RuntimeError
    mod2.VideoUnavailable = RuntimeError
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", mod2)
    with pytest.raises(DirectTranscriptUnavailableError):
        _ = adapter.get_transcript("vid", ["en"])


def test_list_transcripts_and_wrapped_calls(monkeypatch: pytest.MonkeyPatch) -> None:
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
        def get_transcript(video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
            return []

        @staticmethod
        def list_transcripts(video_id: str) -> _YTListingProto:
            return _Listing()

    mod3 = _YTModule("youtube_transcript_api")
    mod3.YouTubeTranscriptApi = _API
    mod3.NoTranscriptFound = KeyError  # mapped in adapter
    mod3.TranscriptsDisabled = RuntimeError
    mod3.VideoUnavailable = RuntimeError
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", mod3)

    adapter = YouTubeTranscriptApiAdapter()
    listing = adapter.list_transcripts("vid")
    res = listing.find_transcript(["en"])
    assert res is not None and res.fetch()[0]["text"] == "x"
    with pytest.raises(TranscriptTranslateUnavailableError):
        _ = listing.translate("en")


def test_find_transcript_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
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
        def list_transcripts(video_id: str) -> _YTListingProto:
            return _Listing()

        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
            return []

    _mod = _YTModule("youtube_transcript_api")
    _mod.YouTubeTranscriptApi = _API
    _mod.NoTranscriptFound = KeyError
    _mod.TranscriptsDisabled = RuntimeError
    _mod.VideoUnavailable = RuntimeError
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", _mod)
    adapter = YouTubeTranscriptApiAdapter()
    listing = adapter.list_transcripts("vid")
    out = listing.find_transcript(["en"])  # returns None from adapter when inner returns None
    assert out is None


def test_find_transcript_maps_no_transcript_found(monkeypatch: pytest.MonkeyPatch) -> None:
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
        def list_transcripts(video_id: str) -> _YTListingProto:
            return _Listing()

        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
            return []

    _mod = _YTModule("youtube_transcript_api")
    _mod.YouTubeTranscriptApi = _API
    _mod.NoTranscriptFound = KeyError
    _mod.TranscriptsDisabled = RuntimeError
    _mod.VideoUnavailable = RuntimeError
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", _mod)
    adapter = YouTubeTranscriptApiAdapter()
    listing = adapter.list_transcripts("vid")
    with pytest.raises(TranscriptLanguageUnavailableError):
        _ = listing.find_transcript(["en"])


def test_translate_success_and_get_transcript_non_list(monkeypatch: pytest.MonkeyPatch) -> None:
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
        def get_transcript(video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
            # Return non-list to exercise coercion to []
            return []

        @staticmethod
        def list_transcripts(video_id: str) -> _YTListingProto:
            return _Listing()

    _mod = _YTModule("youtube_transcript_api")
    _mod.YouTubeTranscriptApi = _API
    _mod.NoTranscriptFound = KeyError
    _mod.TranscriptsDisabled = RuntimeError
    _mod.VideoUnavailable = RuntimeError
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", _mod)
    adapter = YouTubeTranscriptApiAdapter()
    out = adapter.get_transcript("vid", ["en"])
    assert out == []
    listing = adapter.list_transcripts("vid")
    res = listing.translate("en")
    assert res.fetch()[0]["text"] == "ok"


def test_get_transcript_list_with_nondict_items(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MockListing3:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto | None:
            return None

        def translate(self, language: str) -> _YTResourceProto:
            raise RuntimeError("translate unavailable")

    class _API:
        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
            return [{"text": "ok", "start": 0.0, "duration": 1.0}]

        @staticmethod
        def list_transcripts(video_id: str) -> _YTListingProto:
            return _MockListing3()

    _mod = _YTModule("youtube_transcript_api")
    _mod.YouTubeTranscriptApi = _API
    _mod.NoTranscriptFound = KeyError
    _mod.TranscriptsDisabled = RuntimeError
    _mod.VideoUnavailable = RuntimeError
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", _mod)
    adapter = YouTubeTranscriptApiAdapter()
    out = adapter.get_transcript("vid", ["en"])
    assert out and out[0]["text"] == "ok"


def test_list_transcripts_unexpected_type(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test that valid protocol implementations work correctly
    class _MockListing4:
        def find_transcript(self, languages: list[str]) -> _YTResourceProto | None:
            return None

        def translate(self, language: str) -> _YTResourceProto:
            raise RuntimeError("translate unavailable")

    class _API:
        @staticmethod
        def list_transcripts(video_id: str) -> _YTListingProto:
            return _MockListing4()

        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
            return []

    _mod = _YTModule("youtube_transcript_api")
    _mod.YouTubeTranscriptApi = _API
    _mod.NoTranscriptFound = KeyError
    _mod.TranscriptsDisabled = RuntimeError
    _mod.VideoUnavailable = RuntimeError
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", _mod)
    adapter = YouTubeTranscriptApiAdapter()
    listing = adapter.list_transcripts("vid")
    if listing is None:
        pytest.fail("expected listing result")


def test_list_transcripts_unavailable_maps_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _API:
        @staticmethod
        def list_transcripts(video_id: str) -> _YTListingProto:
            raise RuntimeError("disabled")

        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
            return []

    _mod = _YTModule("youtube_transcript_api")
    _mod.YouTubeTranscriptApi = _API
    _mod.TranscriptsDisabled = RuntimeError
    _mod.VideoUnavailable = RuntimeError
    _mod.NoTranscriptFound = KeyError
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", _mod)

    adapter = YouTubeTranscriptApiAdapter()
    with pytest.raises(TranscriptListingError):
        _ = adapter.list_transcripts("vid")


@pytest.mark.parametrize(
    "payload",
    [
        [123],
        [{"text": 123, "start": 0.0, "duration": 1.0}],
        [{"text": "t", "start": "bad", "duration": 1.0}],
        [{"text": "t", "start": 0.0, "duration": "bad"}],
    ],
)
def test_get_transcript_rejects_invalid_items(
    monkeypatch: pytest.MonkeyPatch, payload: list[int | dict[str, str | int | float]]
) -> None:
    class _API:
        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[int | dict[str, JSONValue]]:
            assert video_id == "vid" and languages == ["en"]
            out: list[int | dict[str, JSONValue]] = []
            for it in payload:
                if isinstance(it, dict):
                    d: dict[str, JSONValue] = {}
                    for k, v in it.items():
                        if isinstance(v, (str, int, float)):
                            d[k] = v
                    out.append(d)
                else:
                    out.append(int(it))
            return out

        @staticmethod
        def list_transcripts(video_id: str) -> _YTListingProto:
            raise RuntimeError

    _mod = _YTModule("youtube_transcript_api")
    _mod.YouTubeTranscriptApi = _API
    _mod.NoTranscriptFound = KeyError
    _mod.TranscriptsDisabled = RuntimeError
    _mod.VideoUnavailable = RuntimeError
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", _mod)

    adapter = YouTubeTranscriptApiAdapter()
    with pytest.raises(JSONTypeError):
        _ = adapter.get_transcript("vid", ["en"])


def test_list_transcripts_unexpected_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _API:
        @staticmethod
        def get_transcript(video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
            return []

        @staticmethod
        def list_transcripts(video_id: str) -> str:
            return "bad"

    _mod = _YTModule("youtube_transcript_api")
    _mod.YouTubeTranscriptApi = _API
    _mod.NoTranscriptFound = KeyError
    _mod.TranscriptsDisabled = RuntimeError
    _mod.VideoUnavailable = RuntimeError
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", _mod)
    adapter = YouTubeTranscriptApiAdapter()
    with pytest.raises(TranscriptListingError):
        _ = adapter.list_transcripts("vid")
