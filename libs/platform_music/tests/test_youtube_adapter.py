from __future__ import annotations

from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from platform_music.services import youtube as yt
from platform_music.testing import hooks, make_fake_youtube_http_post


def test_youtube_adapter_fetch_history() -> None:
    doc = {
        "items": [
            {
                "title": "YSong",
                "artist": "YArtist",
                "videoId": "vid1",
                "playedAt": "2024-01-03T00:00:00Z",
                "durationSeconds": 180,
            }
        ]
    }

    hooks.youtube_http_post = make_fake_youtube_http_post(dump_json_str(doc))

    cli = yt.youtube_client(sapisid="sid", cookies="a=b")
    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
    )
    assert len(out) == 1 and out[0]["track"]["service"] == "youtube_music"


def test_youtube_adapter_invalid() -> None:
    hooks.youtube_http_post = make_fake_youtube_http_post("{}")

    cli = yt.youtube_client(sapisid="sid", cookies="a=b")
    try:
        cli.get_listening_history(
            start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
        )
    except AppError:
        return
    raise AssertionError("expected AppError")
