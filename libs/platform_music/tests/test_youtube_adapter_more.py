from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from platform_music.services import youtube as yt
from platform_music.testing import hooks, make_fake_youtube_http_post


def test_youtube_invalid_json_not_object() -> None:
    hooks.youtube_http_post = make_fake_youtube_http_post("[]")

    cli = yt.youtube_client(sapisid="s", cookies="c=1")
    with pytest.raises(AppError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_youtube_item_non_dict() -> None:
    payload = dump_json_str({"items": ["x"]})

    hooks.youtube_http_post = make_fake_youtube_http_post(payload)

    cli = yt.youtube_client(sapisid="s", cookies="c=1")
    with pytest.raises(AppError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_youtube_invalid_fields() -> None:
    payload = dump_json_str({"items": [{"title": 1}]})

    hooks.youtube_http_post = make_fake_youtube_http_post(payload)

    cli = yt.youtube_client(sapisid="s", cookies="c=1")
    with pytest.raises(AppError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_youtube_limit_early_return() -> None:
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

    hooks.youtube_http_post = make_fake_youtube_http_post(payload)

    cli = yt.youtube_client(sapisid="s", cookies="c=1")
    out = cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31", limit=1)
    assert len(out) == 1 and out[0]["track"]["id"] == "v1"
