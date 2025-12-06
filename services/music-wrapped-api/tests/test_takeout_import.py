from __future__ import annotations

import io
import zipfile

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient
from pytest import MonkeyPatch

from music_wrapped_api.app import create_app


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


def test_import_youtube_takeout_json_success(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    import music_wrapped_api.routes.wrapped as routes

    fr = FakeRedis()

    def _conn(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    fq = FakeQueue(job_id="takeout-1")

    def _queue(name: str, *, connection: FakeRedisBytesClient) -> FakeQueue:
        return fq

    def _rf(url: str) -> FakeRedis:
        return fr

    monkeypatch.setattr(routes, "_rq_conn", _conn)
    monkeypatch.setattr(routes, "_rq_queue", _queue)
    monkeypatch.setattr(routes, "redis_for_kv", _rf)

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


def test_import_youtube_takeout_zip_success(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    import music_wrapped_api.routes.wrapped as routes

    fr = FakeRedis()

    def _conn(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    def _queue(name: str, *, connection: FakeRedisBytesClient) -> FakeQueue:
        return FakeQueue(job_id="takeout-2")

    def _rf(url: str) -> FakeRedis:
        return fr

    monkeypatch.setattr(routes, "_rq_conn", _conn)
    monkeypatch.setattr(routes, "_rq_queue", _queue)
    monkeypatch.setattr(routes, "redis_for_kv", _rf)

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


def test_import_youtube_takeout_invalid_fields(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    import music_wrapped_api.routes.wrapped as routes

    def _conn(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    def _queue(name: str, *, connection: FakeRedisBytesClient) -> FakeQueue:
        return FakeQueue(job_id="takeout-3")

    def _rf(url: str) -> FakeRedis:
        return FakeRedis()

    monkeypatch.setattr(routes, "_rq_conn", _conn)
    monkeypatch.setattr(routes, "_rq_queue", _queue)
    monkeypatch.setattr(routes, "redis_for_kv", _rf)

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


def test_import_youtube_takeout_invalid_year(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    import music_wrapped_api.routes.wrapped as routes

    def _conn(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    def _queue(name: str, *, connection: FakeRedisBytesClient) -> FakeQueue:
        return FakeQueue(job_id="takeout-4")

    def _rf(url: str) -> FakeRedis:
        return FakeRedis()

    monkeypatch.setattr(routes, "_rq_conn", _conn)
    monkeypatch.setattr(routes, "_rq_queue", _queue)
    monkeypatch.setattr(routes, "redis_for_kv", _rf)

    client = TestClient(create_app())
    doc = _sample_takeout_items()
    payload = dump_json_str(doc)
    r = client.post(
        "/v1/wrapped/import/youtube-takeout",
        files={"file": ("watch-history.json", payload, "application/json")},
        data={"year": "oops"},
    )
    assert r.status_code == 400
