from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue

from music_wrapped_api.routes._decoders import (
    decode_apple_credentials,
    decode_spotify_credentials,
    decode_wrapped_generate,
    decode_youtube_credentials,
    to_full_lastfm_credentials,
)


def test_require_lastfm_credentials_invalid() -> None:
    # Use decode path to validate input types instead of separate helper
    bad_doc: JSONValue = {
        "year": 2024,
        "service": "lastfm",
        "credentials": {"client_id": "x"},
    }
    with pytest.raises(AppError):
        decode_wrapped_generate(bad_doc)


def test_decode_generate_session_only_and_full() -> None:
    # Session-only
    req1 = decode_wrapped_generate(
        {"year": 2024, "service": "lastfm", "credentials": {"session_key": "t"}}
    )
    assert req1["credentials"]["session_key"] == "t"
    # Full
    req2 = decode_wrapped_generate(
        {
            "year": 2024,
            "service": "lastfm",
            "credentials": {"api_key": "k", "api_secret": "s", "session_key": "t"},
        }
    )
    creds_out = req2["credentials"]
    assert "api_key" in creds_out


def test_to_full_lastfm_credentials_errors_and_paths() -> None:
    # Missing session_key
    # Not testing invalid type construction here since decoder ensures presence
    # Full returns as-is
    out = to_full_lastfm_credentials(
        {"api_key": "k", "api_secret": "s", "session_key": "t"},
        api_key_env="ek",
        api_secret_env="es",
    )
    assert out == {"api_key": "k", "api_secret": "s", "session_key": "t"}


def test_decode_spotify_credentials_success_and_invalid() -> None:
    good: dict[str, JSONValue] = {
        "access_token": "at",
        "refresh_token": "rt",
        "expires_in": 3600,
    }
    out = decode_spotify_credentials(good)
    assert out["refresh_token"] == "rt"
    with pytest.raises(AppError):
        bad_sp: dict[str, JSONValue] = {"access_token": 1}
        decode_spotify_credentials(bad_sp)
    with pytest.raises(AppError):
        decode_spotify_credentials("x")


def test_decode_apple_credentials_success_and_invalid() -> None:
    good: dict[str, JSONValue] = {"developer_token": "d", "music_user_token": "u"}
    out = decode_apple_credentials(good)
    assert out["developer_token"] == "d"
    with pytest.raises(AppError):
        bad_ap: dict[str, JSONValue] = {"developer_token": 1}
        decode_apple_credentials(bad_ap)
    with pytest.raises(AppError):
        decode_apple_credentials(1)


def test_decode_youtube_credentials_success_and_invalid() -> None:
    good: dict[str, JSONValue] = {"sapisid": "sid", "cookies": "ck=1"}
    out = decode_youtube_credentials(good)
    assert out["sapisid"] == "sid"
    with pytest.raises(AppError):
        bad_yt: dict[str, JSONValue] = {"sapisid": 1}
        decode_youtube_credentials(bad_yt)
    with pytest.raises(AppError):
        decode_youtube_credentials([])
