from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue

from platform_music.wrapped import decode_wrapped_result


def test_decode_wrapped_result_success() -> None:
    doc: JSONValue = {
        "service": "lastfm",
        "year": 2024,
        "generated_at": "2024-12-31T00:00:00Z",
        "total_scrobbles": 10,
        "top_artists": [{"artist_name": "A", "play_count": 5}],
        "top_songs": [{"title": "T", "artist_name": "A", "play_count": 5}],
        "top_by_month": [{"month": 1, "top_artists": []}],
    }
    out = decode_wrapped_result(doc)
    assert out["year"] == 2024
    assert out["service"] == "lastfm"


def test_decode_wrapped_result_other_services() -> None:
    for svc in ("spotify", "apple_music", "youtube_music"):
        doc: JSONValue = {
            "service": svc,
            "year": 2024,
            "generated_at": "2024-12-31T00:00:00Z",
            "total_scrobbles": 1,
            "top_artists": [],
            "top_songs": [],
            "top_by_month": [],
        }
        out = decode_wrapped_result(doc)
        assert out["service"] == svc


def test_decode_wrapped_result_invalid() -> None:
    bad: JSONValue = {
        "service": "lastfm",
        "year": "2024",
        "generated_at": "",
        "total_scrobbles": -1,
        "top_artists": [{}],
        "top_songs": [{}],
    }
    with pytest.raises(AppError):
        decode_wrapped_result(bad)


def _valid_doc() -> dict[str, JSONValue]:
    return {
        "service": "lastfm",
        "year": 2024,
        "generated_at": "2024-12-31T00:00:00Z",
        "total_scrobbles": 1,
        "top_artists": [{"artist_name": "A", "play_count": 1}],
        "top_songs": [{"title": "T", "artist_name": "A", "play_count": 1}],
        "top_by_month": [{"month": 1, "top_artists": [{"artist_name": "A", "play_count": 1}]}],
    }


def test_decode_wrapped_not_object() -> None:
    with pytest.raises(AppError):
        decode_wrapped_result(1)


def test_decode_wrapped_invalid_service() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    doc["service"] = "unknown"
    with pytest.raises(AppError):
        decode_wrapped_result(doc)


def test_decode_wrapped_missing_generated_at() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    doc["generated_at"] = ""
    with pytest.raises(AppError):
        decode_wrapped_result(doc)


def test_decode_wrapped_invalid_total_scrobbles() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    doc["total_scrobbles"] = -1
    with pytest.raises(AppError):
        decode_wrapped_result(doc)


def test_decode_wrapped_top_artists_not_list() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    doc["top_artists"] = 1
    with pytest.raises(AppError):
        decode_wrapped_result(doc)


def test_decode_wrapped_artist_entry_invalid() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    bad_list: list[JSONValue] = [1]
    doc["top_artists"] = bad_list
    with pytest.raises(AppError):
        decode_wrapped_result(doc)


def test_decode_wrapped_artist_fields_invalid() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    bad_artists: list[JSONValue] = [{"artist_name": 1, "play_count": "x"}]
    doc["top_artists"] = bad_artists
    with pytest.raises(AppError):
        decode_wrapped_result(doc)


def test_decode_wrapped_top_songs_not_list() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    doc["top_songs"] = 2
    with pytest.raises(AppError):
        decode_wrapped_result(doc)


def test_decode_wrapped_song_entry_invalid() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    bad_list2: list[JSONValue] = [1]
    doc["top_songs"] = bad_list2
    with pytest.raises(AppError):
        decode_wrapped_result(doc)


def test_decode_wrapped_song_fields_invalid() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    bad_songs: list[JSONValue] = [{"title": 1, "artist_name": 2, "play_count": "x"}]
    doc["top_songs"] = bad_songs
    with pytest.raises(AppError):
        decode_wrapped_result(doc)


def test_decode_wrapped_top_by_month_not_list() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    doc["top_by_month"] = 1
    with pytest.raises(AppError):
        decode_wrapped_result(doc)


def test_decode_wrapped_month_entry_invalid() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    doc["top_by_month"] = [1]
    with pytest.raises(AppError):
        decode_wrapped_result(doc)


def test_decode_wrapped_month_invalid_value() -> None:
    raw = _valid_doc()
    doc: dict[str, JSONValue] = dict(raw)
    doc["top_by_month"] = [{"month": 13, "top_artists": []}]
    with pytest.raises(AppError):
        decode_wrapped_result(doc)
