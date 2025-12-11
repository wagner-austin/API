from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from platform_music.services import apple as ap
from platform_music.testing import hooks, make_fake_apple_http_get


def test_apple_adapter_fetch_history() -> None:
    doc = {
        "data": [
            {
                "attributes": {
                    "name": "SongA",
                    "artistName": "ArtistA",
                    "durationInMillis": 2000,
                    "lastPlayedDate": "2024-01-02T00:00:00Z",
                }
            },
            {
                "id": "a2",
                "attributes": {
                    "name": "SongB",
                    "artistName": "ArtistB",
                    "durationInMillis": 2500,
                    "lastPlayedDate": "2024-01-03T00:00:00Z",
                },
            },
        ]
    }

    hooks.apple_http_get = make_fake_apple_http_get(dump_json_str(doc))

    cli = ap.apple_client(music_user_token="u", developer_token="d")
    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=1
    )
    assert len(out) == 1 and out[0]["track"]["service"] == "apple_music"


def test_apple_adapter_fetch_history_limit_above_size() -> None:
    doc = {
        "data": [
            {
                "attributes": {
                    "name": "SongA",
                    "artistName": "ArtistA",
                    "durationInMillis": 2000,
                    "lastPlayedDate": "2024-01-02T00:00:00Z",
                }
            },
            {
                "id": "a2",
                "attributes": {
                    "name": "SongB",
                    "artistName": "ArtistB",
                    "durationInMillis": 2500,
                    "lastPlayedDate": "2024-01-03T00:00:00Z",
                },
            },
        ]
    }

    hooks.apple_http_get = make_fake_apple_http_get(dump_json_str(doc))

    cli = ap.apple_client(music_user_token="u", developer_token="d")
    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=10
    )
    assert len(out) == 2


def test_apple_decode_item_non_dict() -> None:
    doc = {"data": ["x"]}

    hooks.apple_http_get = make_fake_apple_http_get(dump_json_str(doc))

    cli = ap.apple_client(music_user_token="u", developer_token="d")
    with pytest.raises(AppError):
        cli.get_listening_history(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-12-31T23:59:59Z",
        )


def test_apple_decode_item_missing_attrs() -> None:
    doc = {"data": [{"id": "a", "bad": 1}]}

    hooks.apple_http_get = make_fake_apple_http_get(dump_json_str(doc))

    cli = ap.apple_client(music_user_token="u", developer_token="d")
    with pytest.raises(AppError):
        cli.get_listening_history(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-12-31T23:59:59Z",
        )


def test_apple_decode_item_bad_attrs() -> None:
    doc = {"data": [{"attributes": {"name": 1}}]}

    hooks.apple_http_get = make_fake_apple_http_get(dump_json_str(doc))

    cli = ap.apple_client(music_user_token="u", developer_token="d")
    with pytest.raises(AppError):
        cli.get_listening_history(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-12-31T23:59:59Z",
        )


def test_apple_adapter_invalid() -> None:
    hooks.apple_http_get = make_fake_apple_http_get("{}")

    cli = ap.apple_client(music_user_token="u", developer_token="d")
    with pytest.raises(AppError):
        cli.get_listening_history(
            start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
        )
