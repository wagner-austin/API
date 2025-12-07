from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import dump_json_str
from platform_workers.testing import FakeRedis
from pytest import MonkeyPatch

from music_wrapped_api.app import create_app


def test_download_not_found(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")

    import music_wrapped_api.routes.wrapped as routes

    fr = FakeRedis()

    def _rf(url: str) -> FakeRedis:
        return fr

    monkeypatch.setattr(routes, "redis_for_kv", _rf)
    client = TestClient(create_app())
    r = client.get("/v1/wrapped/download/missing")
    assert r.status_code == 404
    fr.assert_only_called({"get"})


def test_download_ok(monkeypatch: MonkeyPatch) -> None:
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
                "top_songs": [],
                "top_by_month": [],
            }
        ),
    )

    def _rf(url: str) -> FakeRedis:
        return fr

    monkeypatch.setattr(routes, "redis_for_kv", _rf)
    client = TestClient(create_app())
    r = client.get(f"/v1/wrapped/download/{rid}")
    assert r.status_code == 200
    # PNG signature
    assert r.content.startswith(b"\x89PNG\r\n\x1a\n")
    fr.assert_only_called({"set", "expire", "get"})


def test_download_invalid_shape(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    import music_wrapped_api.routes.wrapped as routes

    fr = FakeRedis()
    rid = "wrapped:bad:shape"
    fr.set(rid, dump_json_str({}))

    def _rf(url: str) -> FakeRedis:
        return fr

    monkeypatch.setattr(routes, "redis_for_kv", _rf)
    client = TestClient(create_app())
    r = client.get(f"/v1/wrapped/download/{rid}")
    assert r.status_code == 400
    fr.assert_only_called({"set", "expire", "get"})
