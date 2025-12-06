from __future__ import annotations

import io
import zipfile

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str

from platform_music.importers.youtube_takeout import (
    decode_stored_plays,
    decode_takeout_json,
    parse_takeout_bytes,
)


def test_parse_takeout_bytes_invalid_root_json() -> None:
    # JSON object instead of list should fail in _expect_list
    with pytest.raises(AppError):
        parse_takeout_bytes(b"{}", content_type="application/json")


def test_decode_takeout_json_missing_title_and_time_errors() -> None:
    # Missing title
    bad1: list[JSONValue] = [{"time": "2024-01-01T00:00:00Z", "products": ["YouTube Music"]}]
    with pytest.raises(AppError):
        decode_takeout_json(bad1)
    # Missing time
    bad2: list[JSONValue] = [{"title": "S", "products": ["YouTube Music"]}]
    with pytest.raises(AppError):
        decode_takeout_json(bad2)


def test_decode_takeout_json_item_not_dict() -> None:
    with pytest.raises(AppError):
        decode_takeout_json([1])


def test_decode_takeout_json_unknown_artist_when_no_subtitles() -> None:
    doc: list[JSONValue] = [
        {
            "title": "S",
            "titleUrl": "https://www.youtube.com/watch?v=v123&list=abc",
            "time": "2024-01-01T00:00:00Z",
            "products": ["YouTube Music"],
        }
    ]
    plays = decode_takeout_json(doc)
    assert plays[0]["track"]["artist_name"] == "Unknown"
    assert plays[0]["track"]["id"] == "v123"


def test_decode_takeout_json_titleurl_without_slash_or_v_param() -> None:
    doc: list[JSONValue] = [
        {
            "title": "NoId",
            "titleUrl": "opaque",
            "time": "2024-01-01T00:00:00Z",
            "products": ["YouTube Music"],
        }
    ]
    plays = decode_takeout_json(doc)
    # When extraction fails, we keep the original string as id
    assert plays[0]["track"]["id"] == "opaque"


def test_decode_takeout_json_short_form_with_trailing_slash() -> None:
    # Ensures branch where last path segment is empty
    doc: list[JSONValue] = [
        {
            "title": "Empty",
            "titleUrl": "https://youtu.be/",
            "time": "2024-01-01T00:00:00Z",
            "products": ["YouTube Music"],
        }
    ]
    plays = decode_takeout_json(doc)
    assert plays[0]["track"]["id"] == "https://youtu.be/"


def test_parse_takeout_bytes_magic_detection_without_content_type() -> None:
    items: list[dict[str, JSONValue]] = [
        {
            "title": "A",
            "time": "2024-01-01T00:00:00Z",
            "products": ["YouTube Music"],
        }
    ]
    data = dump_json_str(items).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("Takeout/YouTube and YouTube Music/history/watch-history.json", data)
    raw_zip = buf.getvalue()
    plays = parse_takeout_bytes(raw_zip, content_type="application/octet-stream")
    assert len(plays) == 1


def test_parse_takeout_bytes_zip_missing_history_json_errors() -> None:
    data = dump_json_str([{"x": 1}]).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("Takeout/nope.json", data)
    raw_zip = buf.getvalue()
    with pytest.raises(AppError):
        parse_takeout_bytes(raw_zip, content_type="application/zip")


def test_parse_takeout_bytes_zip_history_generic_name() -> None:
    # Covers second-pass candidate selection branch (history present without 'youtube' token)
    items: list[dict[str, JSONValue]] = [
        {
            "title": "A",
            "time": "2024-01-01T00:00:00Z",
            "products": ["YouTube Music"],
        }
    ]
    data = dump_json_str(items).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("SomeFolder/history.json", data)
    raw_zip = buf.getvalue()
    plays = parse_takeout_bytes(raw_zip, content_type="application/zip")
    assert len(plays) == 1


def test_static_service_limit_applied() -> None:
    doc: list[JSONValue] = [
        {"title": "S1", "time": "2024-01-01T00:00:00Z", "products": ["YouTube Music"]},
        {"title": "S2", "time": "2024-01-02T00:00:00Z", "products": ["YouTube Music"]},
        {"title": "S3", "time": "2024-01-03T00:00:00Z", "products": ["YouTube Music"]},
    ]
    plays = decode_takeout_json(doc)
    from platform_music.importers.youtube_takeout import static_service_from_plays

    svc = static_service_from_plays(plays)
    res = svc.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=2
    )
    assert len(res) == 2


def test_decode_stored_plays_invalid_shapes() -> None:
    # Not a list
    not_list: JSONValue = load_json_str("{}")
    with pytest.raises(AppError):
        decode_stored_plays(not_list)
    # Item not a dict
    bad_list: JSONValue = load_json_str("[1]")
    with pytest.raises(AppError):
        decode_stored_plays(bad_list)
    # Missing track
    with pytest.raises(AppError):
        decode_stored_plays([{"played_at": "t", "service": "youtube_music"}])
    # Wrong field types
    bad_item: JSONValue = {
        "track": {
            "id": 1,
            "title": "t",
            "artist_name": "a",
            "duration_ms": 0,
            "service": "youtube_music",
        },
        "played_at": "t",
        "service": "youtube_music",
    }
    with pytest.raises(AppError):
        decode_stored_plays([bad_item])
