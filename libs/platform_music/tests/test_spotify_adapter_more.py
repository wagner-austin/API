from __future__ import annotations

from platform_core.json_utils import dump_json_str

from platform_music.services import spotify as sp
from platform_music.testing import (
    hooks,
    make_fake_spotify_http_get,
    make_fake_spotify_http_get_pages,
)


def test_spotify_limit_early_return() -> None:
    page = dump_json_str(
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
                },
                {
                    "track": {
                        "id": "t2",
                        "name": "Song2",
                        "artists": [{"name": "Artist2"}],
                        "duration_ms": 1200,
                    },
                    "played_at": "2024-01-01T00:01:00Z",
                },
            ]
        }
    )

    hooks.spotify_http_get = make_fake_spotify_http_get(page)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=1
    )

    assert len(out) == 1 and out[0]["track"]["id"] == "t1"


def test_spotify_no_cursors_break() -> None:
    # Note: no "cursors" key to exercise the else: break path
    page = dump_json_str(
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
            ]
        }
    )

    hooks.spotify_http_get = make_fake_spotify_http_get(page)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
    )

    assert len(out) == 1 and out[0]["track"]["id"] == "t1"


def test_spotify_after_zero_branch() -> None:
    pages = [
        dump_json_str(
            {
                "items": [
                    {
                        "track": {
                            "id": "t1",
                            "name": "S1",
                            "artists": [{"name": "A1"}],
                            "duration_ms": 1000,
                        },
                        "played_at": "1970-01-01T00:00:00Z",
                    }
                ],
                "cursors": {"before": "1"},
            }
        ),
        dump_json_str({"items": []}),
    ]

    hooks.spotify_http_get = make_fake_spotify_http_get_pages(pages)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    out = cli.get_listening_history(
        start_date="1970-01-01T00:00:00Z", end_date="1970-01-02T00:00:00Z"
    )

    assert len(out) == 1 and out[0]["track"]["id"] == "t1"


def test_spotify_invalid_before_cursor() -> None:
    # Invalid before cursor to exercise else: break path
    page = dump_json_str(
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
            "cursors": {"before": "abc"},
        }
    )

    hooks.spotify_http_get = make_fake_spotify_http_get(page)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)

    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
    )

    assert len(out) == 1 and out[0]["track"]["id"] == "t1"


def test_spotify_pages_limit_exit() -> None:
    # Build several pages with valid cursors to avoid early breaks
    pages: list[str] = []
    for i in range(6):
        pages.append(
            dump_json_str(
                {
                    "items": [
                        {
                            "track": {
                                "id": f"t{i}",
                                "name": f"Song{i}",
                                "artists": [{"name": f"Artist{i}"}],
                                "duration_ms": 1000 + i,
                            },
                            "played_at": "2024-01-01T00:00:00Z",
                        }
                    ],
                    "cursors": {"before": "123"},
                }
            )
        )

    hooks.spotify_http_get = make_fake_spotify_http_get_pages(pages)

    cli = sp.spotify_client(access_token="at", refresh_token="rt", expires_in=3600)
    out = cli.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z"
    )
    # pages limit = 5 when limit is None; we should have 5 items
    assert len(out) == 5
