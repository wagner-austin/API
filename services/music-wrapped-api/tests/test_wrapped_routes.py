from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import dump_json_str, load_json_str
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient
from pytest import MonkeyPatch

from music_wrapped_api.app import create_app


def test_generate_enqueues_job(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    # Only needed for session-only flow; full creds path should not read env
    monkeypatch.setenv("LASTFM_API_KEY", "k")
    monkeypatch.setenv("LASTFM_API_SECRET", "s")

    import music_wrapped_api.routes.wrapped as routes

    def _conn(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    def _q(name: str, *, connection: FakeRedisBytesClient) -> FakeQueue:
        return FakeQueue(job_id="mw-job-1")

    monkeypatch.setattr(routes, "_rq_conn", _conn)
    monkeypatch.setattr(routes, "_rq_queue", _q)

    client = TestClient(create_app())
    payload = {
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


def test_generate_enqueues_job_with_session_only(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    monkeypatch.setenv("LASTFM_API_KEY", "k")
    monkeypatch.setenv("LASTFM_API_SECRET", "s")

    import music_wrapped_api.routes.wrapped as routes

    def _conn(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    def _q(name: str, *, connection: FakeRedisBytesClient) -> FakeQueue:
        return FakeQueue(job_id="mw-job-2")

    monkeypatch.setattr(routes, "_rq_conn", _conn)
    monkeypatch.setattr(routes, "_rq_queue", _q)

    client = TestClient(create_app())
    payload = {"year": 2024, "service": "lastfm", "credentials": {"session_key": "t"}}
    r = client.post("/v1/wrapped/generate", json=payload)
    assert r.status_code == 200
    body2 = load_json_str(r.text)
    if not isinstance(body2, dict):
        raise AssertionError("response must be an object")
    assert body2.get("status") == "queued"
    assert body2.get("job_id") == "mw-job-2"


def test_generate_invalid_body_and_year(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    client = TestClient(create_app())

    # Non-object body
    bad_list: list[int] = [1, 2]
    r1 = client.post("/v1/wrapped/generate", json=bad_list)
    assert r1.status_code == 400

    # Missing/invalid year
    bad_obj: dict[str, str] = {"year": "bad"}
    r2 = client.post("/v1/wrapped/generate", json=bad_obj)
    assert r2.status_code == 400


def test_generate_invalid_service_and_credentials(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    monkeypatch.setenv("LASTFM_API_KEY", "k")
    monkeypatch.setenv("LASTFM_API_SECRET", "s")
    client = TestClient(create_app())

    # Invalid service
    bad_service = {
        "year": 2024,
        "service": "spotify",
        "credentials": {"api_key": "k", "api_secret": "s", "session_key": "t"},
    }
    r1 = client.post("/v1/wrapped/generate", json=bad_service)
    assert r1.status_code == 400

    # Credentials not an object
    bad_creds1 = {"year": 2024, "service": "lastfm", "credentials": 123}
    r2 = client.post("/v1/wrapped/generate", json=bad_creds1)
    assert r2.status_code == 400

    # Missing fields in credentials
    bad_creds2 = {"year": 2024, "service": "lastfm", "credentials": {"api_key": 1}}
    r3 = client.post("/v1/wrapped/generate", json=bad_creds2)
    assert r3.status_code == 400

    # Wrong type for api_secret
    bad_creds3 = {
        "year": 2024,
        "service": "lastfm",
        "credentials": {"api_key": "k", "api_secret": 1, "session_key": "t"},
    }
    r4 = client.post("/v1/wrapped/generate", json=bad_creds3)
    assert r4.status_code == 400

    # Wrong type for session_key
    bad_creds4 = {
        "year": 2024,
        "service": "lastfm",
        "credentials": {"api_key": "k", "api_secret": "s", "session_key": 2},
    }
    r5 = client.post("/v1/wrapped/generate", json=bad_creds4)
    assert r5.status_code == 400


def test_result_not_found(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")

    import music_wrapped_api.routes.wrapped as routes

    fr = FakeRedis()

    def _rf(url: str) -> FakeRedis:
        return fr

    monkeypatch.setattr(routes, "redis_for_kv", _rf)
    client = TestClient(create_app())
    r = client.get("/v1/wrapped/result/foo")
    assert r.status_code == 404
    fr.assert_only_called({"get"})


def test_result_ok(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    import music_wrapped_api.routes.wrapped as routes

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

    def _rf(url: str) -> FakeRedis:
        return fr

    monkeypatch.setattr(routes, "redis_for_kv", _rf)
    client = TestClient(create_app())
    r = client.get(f"/v1/wrapped/result/{rid}")
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    assert body.get("year") == 2024
    fr.assert_only_called({"set", "expire", "get"})


def test_wrapped_route_wrappers(monkeypatch: MonkeyPatch) -> None:
    import music_wrapped_api.routes.wrapped as routes

    def _raw(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    def _rq(name: str, *, connection: FakeRedisBytesClient) -> FakeQueue:
        return FakeQueue()

    monkeypatch.setattr(routes, "redis_raw_for_rq", _raw)
    monkeypatch.setattr(routes, "rq_queue", _rq)

    conn = routes._rq_conn("redis://ignored")
    q = routes._rq_queue("music_wrapped", connection=conn)
    if not isinstance(q, FakeQueue):
        raise AssertionError("queue must be FakeQueue")
