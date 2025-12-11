from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue
from platform_workers.testing import FakeRedis

from music_wrapped_api import _test_hooks
from music_wrapped_api.api.routes.wrapped import _build_payload_for_service


def test_build_payload_spotify_token_and_full() -> None:
    fr = FakeRedis()
    fr.hset(
        "spotify:session:tk",
        {"access_token": "at", "refresh_token": "rt", "expires_in": "3600"},
    )
    _test_hooks.redis_factory = lambda url: fr

    # Token path
    doc_tok: dict[str, JSONValue] = {
        "year": 2024,
        "service": "spotify",
        "credentials": {"token_id": "tk"},
    }
    out_tok = _build_payload_for_service(doc_tok, redis_url="redis://ignored")
    assert out_tok["service"] == "spotify" and out_tok["credentials"] is not None

    # Full path
    doc_full: dict[str, JSONValue] = {
        "year": 2024,
        "service": "spotify",
        "credentials": {"access_token": "at2", "refresh_token": "rt2", "expires_in": 3600},
    }
    out_full = _build_payload_for_service(doc_full, redis_url="redis://ignored")
    assert out_full["service"] == "spotify"
    fr.assert_only_called({"hset", "expire", "hgetall"})


def test_build_payload_apple_token_and_full() -> None:
    fr = FakeRedis()
    fr.hset("apple:session:atk", {"music_user_token": "mut"})
    _test_hooks.redis_factory = lambda url: fr

    # Token path
    doc_tok: dict[str, JSONValue] = {
        "year": 2024,
        "service": "apple_music",
        "credentials": {"token_id": "atk"},
    }
    out_tok = _build_payload_for_service(doc_tok, redis_url="redis://ignored")
    assert out_tok["service"] == "apple_music"

    # Full path
    doc_full: dict[str, JSONValue] = {
        "year": 2024,
        "service": "apple_music",
        "credentials": {"developer_token": "d2", "music_user_token": "u2"},
    }
    out_full = _build_payload_for_service(doc_full, redis_url="redis://ignored")
    assert out_full["service"] == "apple_music"
    fr.assert_only_called({"hset", "expire", "hgetall"})


def test_build_payload_youtube_token_and_full() -> None:
    fr = FakeRedis()
    fr.hset("ytmusic:session:ytk", {"sapisid": "sid", "cookies": "a=b"})
    _test_hooks.redis_factory = lambda url: fr

    # Token path
    doc_tok: dict[str, JSONValue] = {
        "year": 2024,
        "service": "youtube_music",
        "credentials": {"token_id": "ytk"},
    }
    out_tok = _build_payload_for_service(doc_tok, redis_url="redis://ignored")
    assert out_tok["service"] == "youtube_music"

    # Full path
    doc_full: dict[str, JSONValue] = {
        "year": 2024,
        "service": "youtube_music",
        "credentials": {"sapisid": "sid2", "cookies": "c=d"},
    }
    out_full = _build_payload_for_service(doc_full, redis_url="redis://ignored")
    assert out_full["service"] == "youtube_music"
    fr.assert_only_called({"hset", "expire", "hgetall"})


def test_build_payload_unsupported_and_year_invalid() -> None:
    with pytest.raises(AppError):
        _build_payload_for_service({"service": "soundcloud", "year": 2024}, redis_url="r")
    with pytest.raises(AppError):
        _build_payload_for_service(
            {"service": "spotify", "year": "x", "credentials": {"token_id": "z"}}, redis_url="r"
        )


def test_decode_generate_any_variants() -> None:
    from music_wrapped_api.api.routes._decoders import decode_generate_any

    out_sp_tok = decode_generate_any(
        {"year": 2024, "service": "spotify", "credentials": {"token_id": "t"}}
    )
    assert out_sp_tok["service"] == "spotify"
    out_sp_full = decode_generate_any(
        {
            "year": 2024,
            "service": "spotify",
            "credentials": {"access_token": "a", "refresh_token": "r", "expires_in": 10},
        }
    )
    assert out_sp_full["service"] == "spotify"

    out_ap_tok = decode_generate_any(
        {"year": 2024, "service": "apple_music", "credentials": {"token_id": "t"}}
    )
    assert out_ap_tok["service"] == "apple_music"
    out_ap_full = decode_generate_any(
        {
            "year": 2024,
            "service": "apple_music",
            "credentials": {"developer_token": "d", "music_user_token": "u"},
        }
    )
    assert out_ap_full["service"] == "apple_music"

    out_yt_tok = decode_generate_any(
        {"year": 2024, "service": "youtube_music", "credentials": {"token_id": "t"}}
    )
    assert out_yt_tok["service"] == "youtube_music"
    out_yt_full = decode_generate_any(
        {
            "year": 2024,
            "service": "youtube_music",
            "credentials": {"sapisid": "s", "cookies": "c=1"},
        }
    )
    assert out_yt_full["service"] == "youtube_music"

    with pytest.raises(AppError):
        decode_generate_any({"year": 2024, "service": "unknown", "credentials": {}})

    # Last.fm path via decode_generate_any
    out_lf_sess = decode_generate_any(
        {"year": 2024, "service": "lastfm", "credentials": {"session_key": "s"}}
    )
    assert out_lf_sess["service"] == "lastfm"
    out_lf_full = decode_generate_any(
        {
            "year": 2024,
            "service": "lastfm",
            "credentials": {"api_key": "k", "api_secret": "s", "session_key": "t"},
        }
    )
    assert out_lf_full["service"] == "lastfm"

    # Early validation errors in decode_generate_any
    with pytest.raises(AppError):
        decode_generate_any(1)
    with pytest.raises(AppError):
        decode_generate_any({"year": "x"})


def test_build_payload_token_not_found_and_invalid_full() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    # Spotify token not found
    with pytest.raises(AppError):
        _build_payload_for_service(
            {"year": 2024, "service": "spotify", "credentials": {"token_id": "missing"}},
            redis_url="redis://ignored",
        )

    # Apple token not found
    with pytest.raises(AppError):
        _build_payload_for_service(
            {"year": 2024, "service": "apple_music", "credentials": {"token_id": "atk"}},
            redis_url="redis://ignored",
        )

    # YouTube token not found
    with pytest.raises(AppError):
        _build_payload_for_service(
            {"year": 2024, "service": "youtube_music", "credentials": {"token_id": "ytk"}},
            redis_url="redis://ignored",
        )

    # Apple invalid full types
    with pytest.raises(AppError):
        _build_payload_for_service(
            {
                "year": 2024,
                "service": "apple_music",
                "credentials": {"developer_token": 1, "music_user_token": "u"},
            },
            redis_url="redis://ignored",
        )

    # Apple invalid payload type
    with pytest.raises(AppError):
        _build_payload_for_service(
            {"year": 2024, "service": "apple_music", "credentials": 1},
            redis_url="redis://ignored",
        )

    # YouTube invalid full types
    with pytest.raises(AppError):
        _build_payload_for_service(
            {
                "year": 2024,
                "service": "youtube_music",
                "credentials": {"sapisid": 1, "cookies": "ck"},
            },
            redis_url="redis://ignored",
        )

    # YouTube invalid payload type
    with pytest.raises(AppError):
        _build_payload_for_service(
            {"year": 2024, "service": "youtube_music", "credentials": 1},
            redis_url="redis://ignored",
        )

    # Spotify invalid payload type
    with pytest.raises(AppError):
        _build_payload_for_service(
            {"year": 2024, "service": "spotify", "credentials": 1},
            redis_url="redis://ignored",
        )
    fr.assert_only_called({"hgetall"})


def test_decode_wrapped_generate_and_store_edge_cases() -> None:
    import pytest
    from platform_core.errors import AppError

    from music_wrapped_api.api.routes._decoders import (
        decode_apple_store,
        decode_wrapped_generate,
    )

    with pytest.raises(AppError):
        decode_wrapped_generate(1)
    with pytest.raises(AppError):
        decode_wrapped_generate({"year": "x", "service": "lastfm", "credentials": {}})
    with pytest.raises(AppError):
        decode_wrapped_generate({"year": 2024, "service": "spotify", "credentials": {}})
    with pytest.raises(AppError):
        decode_apple_store(1)


def test_decode_token_ref_errors() -> None:
    import pytest
    from platform_core.errors import AppError

    from music_wrapped_api.api.routes._decoders import _decode_token_ref

    with pytest.raises(AppError):
        _decode_token_ref("x")
    with pytest.raises(AppError):
        _decode_token_ref({"token_id": 1})
