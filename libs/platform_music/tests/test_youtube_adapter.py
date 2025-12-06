from __future__ import annotations

from typing import Protocol

from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str
from pytest import MonkeyPatch


class _Req(Protocol):
    def add_header(self, name: str, value: str) -> None: ...


class _RespProto(Protocol):
    def read(self) -> bytes: ...

    def __enter__(self) -> _RespProto: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: BaseException | None,
    ) -> None: ...


class UrlProto(Protocol):
    def urlopen(self, req: _Req, timeout: float) -> _RespProto: ...


def test_youtube_adapter_fetch_history(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import youtube as yt

    doc = {
        "items": [
            {
                "title": "YSong",
                "artist": "YArtist",
                "videoId": "vid1",
                "playedAt": "2024-01-03T00:00:00Z",
                "durationSeconds": 180,
            }
        ]
    }

    class _Resp:
        def __init__(self, data: str) -> None:
            self._d = data.encode("utf-8")

        def read(self) -> bytes:
            return self._d

        def __enter__(self) -> _Resp:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: BaseException | None,
        ) -> None:
            return None

    class _Url:
        class Request:
            def __init__(self, url: str) -> None:
                self._url = url

            def add_header(self, name: str, value: str) -> None:
                _ = (name, value)

        def urlopen(self, req: _Req, timeout: float) -> _Resp:
            return _Resp(dump_json_str(doc))

    def _fake_import(name: str, fromlist: list[str]) -> UrlProto:
        if name == "urllib.request":
            return _Url()
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(yt, "__import__", _fake_import)
    cli = yt.youtube_client(sapisid="sid", cookies="a=b")
    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
    )
    assert len(out) == 1 and out[0]["track"]["service"] == "youtube_music"


def test_youtube_adapter_invalid(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import youtube as yt

    class _Resp:
        def read(self) -> bytes:
            return b"{}"

        def __enter__(self) -> _Resp:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: BaseException | None,
        ) -> None:
            return None

    class _Url:
        class Request:
            def __init__(self, url: str) -> None:
                self._url = url

            def add_header(self, name: str, value: str) -> None:
                _ = (name, value)

        def urlopen(self, req: _Req, timeout: float) -> _Resp:
            return _Resp()

    def _fake_import(name: str, fromlist: list[str]) -> UrlProto:
        if name == "urllib.request":
            return _Url()
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(yt, "__import__", _fake_import)
    cli = yt.youtube_client(sapisid="sid", cookies="a=b")
    try:
        cli.get_listening_history(
            start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
        )
    except AppError:
        return
    raise AssertionError("expected AppError")
