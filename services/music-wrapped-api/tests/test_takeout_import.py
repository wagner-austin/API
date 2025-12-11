from __future__ import annotations

import io
import zipfile

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from platform_workers.testing import FakeQueue, FakeRedis

from music_wrapped_api import _test_hooks
from music_wrapped_api.api.main import create_app


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
    ]


def test_import_youtube_takeout_json_success() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    fq = FakeQueue(job_id="takeout-1")
    _test_hooks.rq_queue_factory = lambda name, conn: fq

    client = TestClient(create_app())

    doc = _sample_takeout_items()
    payload = dump_json_str(doc)
    files = {"file": ("watch-history.json", payload, "application/json")}
    data = {"year": "2024"}

    r = client.post("/v1/wrapped/import/youtube-takeout", files=files, data=data)
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    assert body.get("status") == "queued"
    tok = body.get("token_id")
    if not isinstance(tok, str):
        raise AssertionError("expected token_id")

    saved = fr.get(f"ytmusic:takeout:{tok}")
    if saved is None:
        raise AssertionError("missing saved takeout")
    arr = load_json_str(saved)
    if not isinstance(arr, list):
        raise AssertionError("expected list of plays")
    assert len(arr) == 2
    assert fq.jobs and fq.jobs[0].func == "platform_music.jobs.process_import_youtube_takeout"
    fr.assert_only_called({"sadd", "set", "expire", "get"})


def test_import_youtube_takeout_zip_success() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    fq = FakeQueue(job_id="takeout-2")
    _test_hooks.rq_queue_factory = lambda name, conn: fq

    client = TestClient(create_app())

    doc = _sample_takeout_items()
    data = dump_json_str(doc).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("Takeout/YouTube and YouTube Music/history/watch-history.json", data)
    raw_zip = buf.getvalue()

    files = {"file": ("takeout.zip", raw_zip, "application/zip")}
    form = {"year": "2024"}
    r = client.post("/v1/wrapped/import/youtube-takeout", files=files, data=form)
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    tok = body.get("token_id")
    if not isinstance(tok, str):
        raise AssertionError("missing token_id")
    saved = fr.get(f"ytmusic:takeout:{tok}")
    if saved is None:
        raise AssertionError("missing saved takeout")
    fr.assert_only_called({"sadd", "set", "expire", "get"})


def test_import_youtube_takeout_invalid_fields() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    fq = FakeQueue(job_id="takeout-3")
    _test_hooks.rq_queue_factory = lambda name, conn: fq

    client = TestClient(create_app())
    doc = _sample_takeout_items()
    payload = dump_json_str(doc)

    # Missing year
    r1 = client.post(
        "/v1/wrapped/import/youtube-takeout",
        files={"file": ("watch-history.json", payload, "application/json")},
    )
    assert r1.status_code == 400

    # Extra field
    r2 = client.post(
        "/v1/wrapped/import/youtube-takeout",
        files={"file": ("watch-history.json", payload, "application/json")},
        data={"year": "2024", "extra": "x"},
    )
    assert r2.status_code == 400

    # Multiple files
    r3 = client.post(
        "/v1/wrapped/import/youtube-takeout",
        files=[
            ("file", ("a.json", payload, "application/json")),
            ("file", ("b.json", payload, "application/json")),
        ],
        data={"year": "2024"},
    )
    assert r3.status_code == 400
    fr.assert_only_called({"sadd"})


def test_import_youtube_takeout_invalid_year() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    fq = FakeQueue(job_id="takeout-4")
    _test_hooks.rq_queue_factory = lambda name, conn: fq

    client = TestClient(create_app())
    doc = _sample_takeout_items()
    payload = dump_json_str(doc)
    r = client.post(
        "/v1/wrapped/import/youtube-takeout",
        files={"file": ("watch-history.json", payload, "application/json")},
        data={"year": "oops"},
    )
    assert r.status_code == 400
    fr.assert_only_called({"sadd"})
