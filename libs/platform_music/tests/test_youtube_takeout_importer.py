from __future__ import annotations

from platform_core.json_utils import JSONValue, dump_json_str, load_json_str

from platform_music.importers.youtube_takeout import (
    decode_stored_plays,
    decode_takeout_json,
    parse_takeout_bytes,
    static_service_from_plays,
)


def _sample_takeout_items() -> list[dict[str, JSONValue]]:
    return [
        {
            "title": "Song A",
            "titleUrl": "https://www.youtube.com/watch?v=vidA",
            "subtitles": [{"name": "Artist 1"}],
            "time": "2024-01-01T00:00:00Z",
            "products": ["YouTube Music"],
        },
        {
            "title": "Song B",
            "titleUrl": "https://youtu.be/vidB",
            "subtitles": [{"name": "Artist 2"}],
            "time": "2024-02-01T00:00:00Z",
            "products": ["YouTube Music"],
        },
        # Non-YouTube Music entry should be ignored
        {
            "title": "Video C",
            "time": "2024-03-01T00:00:00Z",
            "products": ["YouTube"],
        },
    ]


def test_decode_takeout_json_success() -> None:
    doc = _sample_takeout_items()
    plays = decode_takeout_json(doc)
    assert len(plays) == 2
    assert plays[0]["service"] == "youtube_music"
    assert plays[0]["track"]["id"] == "vidA"
    assert plays[1]["track"]["id"] == "vidB"


def test_parse_takeout_bytes_zip_and_json() -> None:
    # Build a ZIP in-memory containing a plausible history filename
    import io
    import zipfile

    doc = _sample_takeout_items()
    data = dump_json_str(doc).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("Takeout/YouTube and YouTube Music/history/watch-history.json", data)
    raw_zip = buf.getvalue()

    plays_zip = parse_takeout_bytes(raw_zip, content_type="application/zip")
    plays_json = parse_takeout_bytes(data, content_type="application/json")
    assert len(plays_zip) == 2
    assert len(plays_json) == 2


def test_static_service_filtering_and_decode_stored() -> None:
    doc = _sample_takeout_items()
    plays = decode_takeout_json(doc)
    # Re-validate through stored-plays path
    s = dump_json_str(plays)
    plays2 = decode_stored_plays(load_json_str(s))
    svc = static_service_from_plays(plays2)
    res = svc.get_listening_history(
        start_date="2024-02-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=None
    )
    assert len(res) == 1
    assert res[0]["track"]["title"] == "Song B"
