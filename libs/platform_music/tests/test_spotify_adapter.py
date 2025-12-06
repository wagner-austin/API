from __future__ import annotations

from typing import Protocol

from platform_core.errors import AppError
from platform_core.json_utils import JSONValue, dump_json_str
from pytest import MonkeyPatch


class UrlRequestProto(Protocol):
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
    def urlopen(self, req: UrlRequestProto, timeout: float) -> _RespProto: ...


def _build_fake_urllib_for_spotify(pages: list[dict[str, JSONValue]]) -> UrlProto:
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
        def __init__(self) -> None:
            self._i = 0

        class Request:
            def __init__(self, url: str) -> None:
                self._url = url

            def add_header(self, name: str, value: str) -> None:
                _ = (name, value)

        def urlopen(self, req: UrlRequestProto, timeout: float) -> _RespProto:
            out = dump_json_str(pages[self._i])

            self._i = min(self._i + 1, len(pages) - 1)

            return _Resp(out)

    return _Url()


def test_spotify_adapter_fetch_history(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import spotify as sp

    pages: list[dict[str, JSONValue]] = [
        {
            "items": [
                {
                    "track": {
                        "id": "t1",
                        "name": "Song1",
                        "artists": [{"name": "Artist1"}],
                        "duration_ms": 1000,
                    },
                    "played_at": "2024-01-01T00:00:00Z",
                }
            ],
            "cursors": {"before": "1704067200000"},
        },
        {"items": []},
    ]

    fake_url = _build_fake_urllib_for_spotify(pages)

    def _imp(name: str, fromlist: list[str]) -> UrlProto:
        if name == "urllib.request":
            return fake_url

        raise RuntimeError("unexpected import")

    monkeypatch.setattr(sp, "__import__", _imp)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
    )

    assert len(out) == 1 and out[0]["track"]["service"] == "spotify"


def test_spotify_adapter_invalid_json(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import spotify as sp

    class _Resp:
        def read(self) -> bytes:
            return b"[]"

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

        def urlopen(self, req: UrlRequestProto, timeout: float) -> _RespProto:
            return _Resp()

    def _imp2(name: str, fromlist: list[str]) -> UrlProto:
        if name == "urllib.request":
            return _Url()

        raise RuntimeError("unexpected import")

    monkeypatch.setattr(sp, "__import__", _imp2)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    try:
        cli.get_listening_history(
            start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
        )

    except AppError:
        return

    raise AssertionError("expected AppError")
