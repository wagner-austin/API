from __future__ import annotations

import pytest
from platform_core.errors import AppError

from platform_music.services import apple as ap
from platform_music.services import spotify as sp
from platform_music.services import youtube as yt
from platform_music.testing import (
    hooks,
    make_raising_apple_http_get,
    make_raising_spotify_http_get,
    make_raising_youtube_http_post,
)


def test_spotify_adapter_http_error() -> None:
    hooks.spotify_http_get = make_raising_spotify_http_get(RuntimeError("boom"))

    cli = sp.spotify_client(access_token="a", refresh_token="r", expires_in=3600)
    with pytest.raises(RuntimeError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_apple_adapter_http_error() -> None:
    hooks.apple_http_get = make_raising_apple_http_get(RuntimeError("boom"))

    cli = ap.apple_client(music_user_token="u", developer_token="d")
    with pytest.raises(RuntimeError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_youtube_adapter_http_error() -> None:
    hooks.youtube_http_post = make_raising_youtube_http_post(RuntimeError("boom"))

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
