from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient

from music_wrapped_api import _test_hooks
from music_wrapped_api.api.main import create_app


def test_generate_enqueues_job() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    fq = FakeQueue(job_id="mw-job-1")
    _test_hooks.rq_queue_factory = lambda name, conn: fq

    client = TestClient(create_app())
    payload: dict[str, JSONValue] = {
        "year": 2024,
        "service": "lastfm",
        "credentials": {"api_key": "k", "api_secret": "s", "session_key": "t"},
    }
    r = client.post("/v1/wrapped/generate", json=payload)
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    assert body.get("status") == "queued"
    assert body.get("job_id") == "mw-job-1"
    fr.assert_only_called({"sadd"})


def test_generate_enqueues_job_with_session_only() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    fq = FakeQueue(job_id="mw-job-2")
    _test_hooks.rq_queue_factory = lambda name, conn: fq

    client = TestClient(create_app())
    payload: dict[str, JSONValue] = {
        "year": 2024,
        "service": "lastfm",
        "credentials": {"session_key": "t"},
    }
    r = client.post("/v1/wrapped/generate", json=payload)
    assert r.status_code == 200
    body2 = load_json_str(r.text)
    if not isinstance(body2, dict):
        raise AssertionError("response must be an object")
    assert body2.get("status") == "queued"
    assert body2.get("job_id") == "mw-job-2"
    fr.assert_only_called({"sadd"})


def test_generate_invalid_body_and_year() -> None:
    client = TestClient(create_app())

    # Non-object body
    bad_list: list[int] = [1, 2]
    r1 = client.post("/v1/wrapped/generate", json=bad_list)
    assert r1.status_code == 400

    # Missing/invalid year
    bad_obj: dict[str, str] = {"year": "bad"}
    r2 = client.post("/v1/wrapped/generate", json=bad_obj)
    assert r2.status_code == 400


def test_generate_invalid_service_and_credentials() -> None:
    client = TestClient(create_app())

    # Invalid service (spotify credentials format mismatch)
    bad_service: dict[str, JSONValue] = {
        "year": 2024,
        "service": "spotify",
        "credentials": {"api_key": "k", "api_secret": "s", "session_key": "t"},
    }
    r1 = client.post("/v1/wrapped/generate", json=bad_service)
    assert r1.status_code == 400

    # Credentials not an object
    bad_creds1: dict[str, JSONValue] = {"year": 2024, "service": "lastfm", "credentials": 123}
    r2 = client.post("/v1/wrapped/generate", json=bad_creds1)
    assert r2.status_code == 400

    # Missing fields in credentials
    bad_creds2: dict[str, JSONValue] = {
        "year": 2024,
        "service": "lastfm",
        "credentials": {"api_key": 1},
    }
    r3 = client.post("/v1/wrapped/generate", json=bad_creds2)
    assert r3.status_code == 400

    # Wrong type for api_secret
    bad_creds3: dict[str, JSONValue] = {
        "year": 2024,
        "service": "lastfm",
        "credentials": {"api_key": "k", "api_secret": 1, "session_key": "t"},
    }
    r4 = client.post("/v1/wrapped/generate", json=bad_creds3)
    assert r4.status_code == 400

    # Wrong type for session_key
    bad_creds4: dict[str, JSONValue] = {
        "year": 2024,
        "service": "lastfm",
        "credentials": {"api_key": "k", "api_secret": "s", "session_key": 2},
    }
    r5 = client.post("/v1/wrapped/generate", json=bad_creds4)
    assert r5.status_code == 400


def test_result_not_found() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    client = TestClient(create_app())
    r = client.get("/v1/wrapped/result/foo")
    assert r.status_code == 404
    fr.assert_only_called({"sadd", "get"})


def test_result_ok() -> None:
    fr = FakeRedis()
    rid = "wrapped:1:2024"
    fr.set(
        rid,
        dump_json_str(
            {
                "service": "lastfm",
                "year": 2024,
                "generated_at": "2024-12-31T00:00:00Z",
                "total_scrobbles": 15,
                "top_artists": [],
            }
        ),
    )
    _test_hooks.redis_factory = lambda url: fr

    client = TestClient(create_app())
    r = client.get(f"/v1/wrapped/result/{rid}")
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    assert body.get("year") == 2024
    fr.assert_only_called({"sadd", "set", "expire", "get"})


def test_rq_hooks_work() -> None:
    fbc = FakeRedisBytesClient()
    fq = FakeQueue(job_id="test-q")

    _test_hooks.rq_conn = lambda url: fbc
    _test_hooks.rq_queue_factory = lambda name, conn: fq

    conn = _test_hooks.rq_conn("redis://ignored")
    q = _test_hooks.rq_queue_factory("music_wrapped", conn)
    if not isinstance(q, FakeQueue):
        raise AssertionError("queue must be FakeQueue")
