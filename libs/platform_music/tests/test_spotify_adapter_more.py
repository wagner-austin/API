from __future__ import annotations

from typing import Protocol

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


def _fake_urlopen_pages(pages: list[dict[str, JSONValue]]) -> UrlProto:
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


def test_spotify_limit_early_return(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import spotify as sp

    page: dict[str, JSONValue] = {
        "items": [
            {
                "track": {
                    "id": "t1",
                    "name": "Song1",
                    "artists": [{"name": "Artist1"}],
                    "duration_ms": 1000,
                },
                "played_at": "2024-01-01T00:00:00Z",
            },
            {
                "track": {
                    "id": "t2",
                    "name": "Song2",
                    "artists": [{"name": "Artist2"}],
                    "duration_ms": 1200,
                },
                "played_at": "2024-01-01T00:01:00Z",
            },
        ]
    }

    fake_url = _fake_urlopen_pages([page])

    def _imp(name: str, fromlist: list[str]) -> UrlProto:
        if name == "urllib.request":
            return fake_url

        raise RuntimeError("unexpected import")

    monkeypatch.setattr(sp, "__import__", _imp)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=1
    )

    assert len(out) == 1 and out[0]["track"]["id"] == "t1"


def test_spotify_no_cursors_break(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import spotify as sp

    page: dict[str, JSONValue] = {
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
        ]
        # Note: no "cursors" key to exercise the else: break path
    }

    fake_url = _fake_urlopen_pages([page])

    def _imp(name: str, fromlist: list[str]) -> UrlProto:
        if name == "urllib.request":
            return fake_url

        raise RuntimeError("unexpected import")

    monkeypatch.setattr(sp, "__import__", _imp)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
    )

    assert len(out) == 1 and out[0]["track"]["id"] == "t1"


def test_spotify_after_zero_branch(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import spotify as sp

    pages: list[dict[str, JSONValue]] = [
        {
            "items": [
                {
                    "track": {
                        "id": "t1",
                        "name": "S1",
                        "artists": [{"name": "A1"}],
                        "duration_ms": 1000,
                    },
                    "played_at": "1970-01-01T00:00:00Z",
                }
            ],
            "cursors": {"before": "1"},
        },
        {"items": []},
    ]

    fake_url = _fake_urlopen_pages(pages)

    def _imp(name: str, fromlist: list[str]) -> UrlProto:
        if name == "urllib.request":
            return fake_url

        raise RuntimeError("unexpected import")

    monkeypatch.setattr(sp, "__import__", _imp)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    out = cli.get_listening_history(
        start_date="1970-01-01T00:00:00Z", end_date="1970-01-02T00:00:00Z"
    )

    assert len(out) == 1 and out[0]["track"]["id"] == "t1"


def test_spotify_invalid_before_cursor(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import spotify as sp

    page: dict[str, JSONValue] = {
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
        # Invalid before cursor to exercise else: break path
        "cursors": {"before": "abc"},
    }

    fake_url = _fake_urlopen_pages([page])

    def _imp(name: str, fromlist: list[str]) -> UrlProto:
        if name == "urllib.request":
            return fake_url

        raise RuntimeError("unexpected import")

    monkeypatch.setattr(sp, "__import__", _imp)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
    )

    assert len(out) == 1 and out[0]["track"]["id"] == "t1"


def test_spotify_pages_limit_exit(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import spotify as sp

    # Build several pages with valid cursors to avoid early breaks
    pages: list[dict[str, JSONValue]] = []
    for i in range(6):
        pages.append(
            {
                "items": [
                    {
                        "track": {
                            "id": f"t{i}",
                            "name": f"Song{i}",
                            "artists": [{"name": f"Artist{i}"}],
                            "duration_ms": 1000 + i,
                        },
                        "played_at": "2024-01-01T00:00:00Z",
                    }
                ],
                "cursors": {"before": "123"},
            }
        )

    fake_url = _fake_urlopen_pages(pages)

    def _imp(name: str, fromlist: list[str]) -> UrlProto:
        if name == "urllib.request":
            return fake_url
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(sp, "__import__", _imp)
    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)
    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
    )
    # pages limit = 5 when limit is None; we should have 5 items
    assert len(out) == 5
