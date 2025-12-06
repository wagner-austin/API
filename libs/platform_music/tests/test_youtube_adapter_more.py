from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str
from pytest import MonkeyPatch


class FakeUrl:
    class _Resp:
        def __init__(self, data: str) -> None:
            self._d = data.encode("utf-8")

        def read(self) -> bytes:
            return self._d

        def __enter__(self) -> FakeUrl._Resp:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: BaseException | None,
        ) -> None:
            return None

    class Request:
        def __init__(self, url: str) -> None:
            self._url = url

        def add_header(self, name: str, value: str) -> None:
            _ = (name, value)

    def __init__(self, data: str) -> None:
        self._data = data

    def urlopen(self, req: FakeUrl.Request, timeout: float) -> FakeUrl._Resp:
        return FakeUrl._Resp(self._data)


def _mk_url_for_payload(payload: str) -> FakeUrl:
    return FakeUrl(payload)


def test_youtube_invalid_json_not_object(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import youtube as yt

    url = _mk_url_for_payload("[]")

    def _imp(name: str, fromlist: list[str]) -> FakeUrl:
        if name == "urllib.request":
            return url
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(yt, "__import__", _imp)
    cli = yt.youtube_client(sapisid="s", cookies="c=1")
    with pytest.raises(AppError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_youtube_item_non_dict(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import youtube as yt

    payload = dump_json_str({"items": ["x"]})
    url = _mk_url_for_payload(payload)

    def _imp(name: str, fromlist: list[str]) -> FakeUrl:
        if name == "urllib.request":
            return url
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(yt, "__import__", _imp)
    cli = yt.youtube_client(sapisid="s", cookies="c=1")
    with pytest.raises(AppError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_youtube_invalid_fields(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import youtube as yt

    payload = dump_json_str({"items": [{"title": 1}]})
    url = _mk_url_for_payload(payload)

    def _imp(name: str, fromlist: list[str]) -> FakeUrl:
        if name == "urllib.request":
            return url
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(yt, "__import__", _imp)
    cli = yt.youtube_client(sapisid="s", cookies="c=1")
    with pytest.raises(AppError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_youtube_limit_early_return(monkeypatch: MonkeyPatch) -> None:
    from platform_music.services import youtube as yt

    payload = dump_json_str(
        {
            "items": [
                {
                    "title": "a",
                    "artist": "b",
                    "videoId": "v1",
                    "playedAt": "2024-01-01T00:00:00Z",
                    "durationSeconds": 10,
                },
                {
                    "title": "c",
                    "artist": "d",
                    "videoId": "v2",
                    "playedAt": "2024-01-01T00:01:00Z",
                    "durationSeconds": 12,
                },
            ]
        }
    )
    url = _mk_url_for_payload(payload)

    def _imp(name: str, fromlist: list[str]) -> FakeUrl:
        if name == "urllib.request":
            return url
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(yt, "__import__", _imp)
    cli = yt.youtube_client(sapisid="s", cookies="c=1")
    out = cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31", limit=1)
    assert len(out) == 1 and out[0]["track"]["id"] == "v1"
