from __future__ import annotations

from typing import Protocol

import pytest
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


class _UrlLike(Protocol):
    def urlopen(self, req: _Req, timeout: float) -> _RespProto: ...


def test_apple_invalid_json_not_object(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import apple as ap

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
            return _Resp("[]")

    def _imp(name: str, fromlist: list[str]) -> _UrlLike:
        if name == "urllib.request":
            return _Url()
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(ap, "__import__", _imp)
    cli = ap.apple_client(music_user_token="u", developer_token="d")
    with pytest.raises(AppError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_apple_id_present_branch(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import apple as ap

    doc = {
        "data": [
            {
                "id": "a1",
                "attributes": {
                    "name": "SongA",
                    "artistName": "ArtistA",
                    "durationInMillis": 2000,
                    "lastPlayedDate": "2024-01-02T00:00:00Z",
                },
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

    def _imp(name: str, fromlist: list[str]) -> _UrlLike:
        if name == "urllib.request":
            return _Url()
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(ap, "__import__", _imp)
    cli = ap.apple_client(music_user_token="u", developer_token="d")
    out = cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31", limit=1)
    assert out[0]["track"]["id"] == "a1"
