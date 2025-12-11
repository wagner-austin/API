from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from platform_music.services import apple as ap
from platform_music.testing import hooks, make_fake_apple_http_get


def test_apple_invalid_json_not_object() -> None:
    hooks.apple_http_get = make_fake_apple_http_get("[]")

    cli = ap.apple_client(music_user_token="u", developer_token="d")
    with pytest.raises(AppError):
        cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31")


def test_apple_id_present_branch() -> None:
    doc = {
        "data": [
            {
                "id": "a1",
                "attributes": {
                    "name": "SongA",
                    "artistName": "ArtistA",
                    "durationInMillis": 2000,
                    "lastPlayedDate": "2024-01-02T00:00:00Z",
                },
            }
        ]
    }

    hooks.apple_http_get = make_fake_apple_http_get(dump_json_str(doc))

    cli = ap.apple_client(music_user_token="u", developer_token="d")
    out = cli.get_listening_history(start_date="2024-01-01", end_date="2024-12-31", limit=1)
    assert out[0]["track"]["id"] == "a1"
