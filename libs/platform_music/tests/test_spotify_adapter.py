from __future__ import annotations

from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from platform_music.services import spotify as sp
from platform_music.testing import (
    hooks,
    make_fake_spotify_http_get,
    make_fake_spotify_http_get_pages,
)


def test_spotify_adapter_fetch_history() -> None:
    pages = [
        dump_json_str(
            {
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
                "cursors": {"before": "1704067200000"},
            }
        ),
        dump_json_str({"items": []}),
    ]

    hooks.spotify_http_get = make_fake_spotify_http_get_pages(pages)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
    )

    assert len(out) == 1 and out[0]["track"]["service"] == "spotify"


def test_spotify_adapter_invalid_json() -> None:
    hooks.spotify_http_get = make_fake_spotify_http_get("[]")

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    try:
        cli.get_listening_history(
            start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
        )

    except AppError:
        return

    raise AssertionError("expected AppError")
