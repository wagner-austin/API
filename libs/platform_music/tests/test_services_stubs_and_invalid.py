from __future__ import annotations

from typing import NoReturn

import pytest
from platform_core.errors import AppError


def test_spotify_adapter_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from platform_music.services import spotify as sp

    class _Err:
        class Request:
            def __init__(self, url: str) -> None:
                self._url = url

            def add_header(self, name: str, value: str) -> None:
                _ = (name, value)

        def urlopen(self, req: Request, timeout: float) -> NoReturn:
            raise RuntimeError("boom")

    def _imp(name: str, fromlist: list[str]) -> _Err:
        if name == "urllib.request":
            return _Err()
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(sp, "__import__", _imp)
    cli = sp.spotify_client(access_token="a", refresh_token="r", expires_in=3600)
    with pytest.raises(RuntimeError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_apple_adapter_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from platform_music.services import apple as ap

    class _Err:
        class Request:
            def __init__(self, url: str) -> None:
                self._url = url

            def add_header(self, name: str, value: str) -> None:
                _ = (name, value)

        def urlopen(self, req: Request, timeout: float) -> NoReturn:
            raise RuntimeError("boom")

    def _imp(name: str, fromlist: list[str]) -> _Err:
        if name == "urllib.request":
            return _Err()
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(ap, "__import__", _imp)
    cli = ap.apple_client(music_user_token="u", developer_token="d")
    with pytest.raises(RuntimeError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_youtube_adapter_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from platform_music.services import youtube as yt

    class _Err:
        class Request:
            def __init__(self, url: str) -> None:
                self._url = url

            def add_header(self, name: str, value: str) -> None:
                _ = (name, value)

        def urlopen(self, req: Request, timeout: float) -> NoReturn:
            raise RuntimeError("boom")

    def _imp(name: str, fromlist: list[str]) -> _Err:
        if name == "urllib.request":
            return _Err()
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(yt, "__import__", _imp)
    cli = yt.youtube_client(sapisid="s", cookies="c=1")
    with pytest.raises(RuntimeError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test__decode_error_paths_via_adapters() -> None:
    # Validate that adapters surface AppError on invalid shapes; precise
    # type validation is exercised in dedicated adapter tests.
    from platform_core.json_utils import JSONValue

    from platform_music.services.apple import _decode_apple_item
    from platform_music.services.youtube import _decode_yt_item

    bad_val: JSONValue = 1
    with pytest.raises(AppError):
        _decode_apple_item(bad_val)
    with pytest.raises(AppError):
        _decode_yt_item(bad_val)
