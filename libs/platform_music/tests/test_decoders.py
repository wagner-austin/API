from __future__ import annotations

import pytest
from platform_core.json_utils import JSONValue

from platform_music.services.decoders import (
    DecoderError,
    _decode_lastfm_scrobble,
    _decode_spotify_play,
)


def test_decode_lastfm_scrobble_success_str_uts() -> None:
    raw: JSONValue = {
        "artist": {"#text": "Artist"},
        "name": "Song",
        "date": {"uts": "1704067200"},  # 2024-01-01T00:00:00Z
    }
    rec = _decode_lastfm_scrobble(raw)
    assert rec["service"] == "lastfm"
    assert rec["track"]["artist_name"] == "Artist"
    assert rec["track"]["title"] == "Song"
    assert rec["played_at"].endswith("Z")


def test_decode_lastfm_scrobble_success_int_uts() -> None:
    raw: JSONValue = {
        "artist": {"#text": "A"},
        "name": "S",
        "date": {"uts": 1704067200},
    }
    rec = _decode_lastfm_scrobble(raw)
    assert rec["service"] == "lastfm"


_ERR_ARTIST_TEXT = "artist.#text must be a non-empty string"
_ERR_NAME = "name must be a non-empty string"
_ERR_UTS = "date.uts must be an integer or numeric string"


@pytest.mark.parametrize(
    ("raw", "msg"),
    [
        (1, "scrobble must be a dict"),
        ({}, "artist must be a dict"),
        ({"artist": 1}, "artist must be a dict"),
        ({"artist": {}, "name": "x", "date": {"uts": "1"}}, _ERR_ARTIST_TEXT),
        ({"artist": {"#text": ""}, "name": "x", "date": {"uts": "1"}}, _ERR_ARTIST_TEXT),
        ({"artist": {"#text": "A"}}, _ERR_NAME),
        ({"artist": {"#text": "A"}, "name": ""}, _ERR_NAME),
        ({"artist": {"#text": "A"}, "name": "S"}, "date must be a dict"),
        ({"artist": {"#text": "A"}, "name": "S", "date": {}}, _ERR_UTS),
        ({"artist": {"#text": "A"}, "name": "S", "date": {"uts": "xx"}}, _ERR_UTS),
    ],
)
def test_decode_lastfm_scrobble_invalid(raw: JSONValue, msg: str) -> None:
    with pytest.raises(DecoderError) as excinfo:
        _decode_lastfm_scrobble(raw)
    assert msg in str(excinfo.value)


def test_decode_spotify_play_success() -> None:
    raw: JSONValue = {
        "track": {
            "id": "t1",
            "name": "S",
            "artists": [{"name": "A"}],
            "duration_ms": 120000,
        },
        "played_at": "2024-01-01T00:00:00Z",
    }
    rec = _decode_spotify_play(raw)
    assert rec["service"] == "spotify"
    assert rec["track"]["id"] == "t1"
    assert rec["track"]["duration_ms"] == 120000


_ERR_TRACK_ID = "track.id must be a non-empty string"
_ERR_TRACK_NAME = "track.name must be a non-empty string"
_ERR_ARTISTS_EMPTY = "track.artists must be a non-empty list"
_ERR_ARTISTS0_TYPE = "track.artists[0] must be a dict"
_ERR_ARTISTS0_NAME = "track.artists[0].name must be a non-empty string"
_ERR_DUR = "track.duration_ms must be a non-negative int"
_ERR_PLAYED_AT = "played_at must be a non-empty string"


@pytest.mark.parametrize(
    ("raw", "msg"),
    [
        (1, "spotify item must be a dict"),
        ({}, "track must be a dict"),
        ({"track": 1}, "track must be a dict"),
        (
            {
                "track": {"name": "S", "artists": [{"name": "A"}], "duration_ms": 1},
                "played_at": "t",
            },
            _ERR_TRACK_ID,
        ),
        (
            {"track": {"id": "t", "artists": [{"name": "A"}], "duration_ms": 1}, "played_at": "t"},
            _ERR_TRACK_NAME,
        ),
        (
            {"track": {"id": "t", "name": "S", "artists": [], "duration_ms": 1}, "played_at": "t"},
            _ERR_ARTISTS_EMPTY,
        ),
        (
            {"track": {"id": "t", "name": "S", "artists": [1], "duration_ms": 1}, "played_at": "t"},
            _ERR_ARTISTS0_TYPE,
        ),
        (
            {
                "track": {"id": "t", "name": "S", "artists": [{}], "duration_ms": 1},
                "played_at": "t",
            },
            _ERR_ARTISTS0_NAME,
        ),
        (
            {
                "track": {
                    "id": "t",
                    "name": "S",
                    "artists": [{"name": "A"}],
                    "duration_ms": -1,
                },
                "played_at": "t",
            },
            _ERR_DUR,
        ),
        (
            {"track": {"id": "t", "name": "S", "artists": [{"name": "A"}], "duration_ms": 1}},
            _ERR_PLAYED_AT,
        ),
        (
            {
                "track": {"id": "t", "name": "S", "artists": [{"name": "A"}], "duration_ms": 1},
                "played_at": 1,
            },
            _ERR_PLAYED_AT,
        ),
    ],
)
def test_decode_spotify_play_invalid(raw: JSONValue, msg: str) -> None:
    with pytest.raises(DecoderError) as excinfo:
        _decode_spotify_play(raw)
    assert msg in str(excinfo.value)
