from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import dump_json_str
from platform_music import WrappedResult
from platform_workers.testing import FakeRedis

from music_wrapped_api import _test_hooks
from music_wrapped_api.api.main import create_app


class _FakeRenderer:
    """Fake renderer that returns a minimal PNG."""

    def render_wrapped(self, result: WrappedResult) -> bytes:
        # Return minimal PNG signature + minimal data
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


def test_download_not_found() -> None:
    fr = FakeRedis()
    _test_hooks.redis_factory = lambda url: fr

    client = TestClient(create_app())
    r = client.get("/v1/wrapped/download/missing")
    assert r.status_code == 404
    fr.assert_only_called({"sadd", "get"})


def test_download_ok() -> None:
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
    _test_hooks.redis_factory = lambda url: fr
    _test_hooks.build_renderer = lambda: _FakeRenderer()

    client = TestClient(create_app())
    r = client.get(f"/v1/wrapped/download/{rid}")
    assert r.status_code == 200
    # PNG signature
    assert r.content.startswith(b"\x89PNG\r\n\x1a\n")
    fr.assert_only_called({"sadd", "set", "expire", "get"})


def test_download_invalid_shape() -> None:
    fr = FakeRedis()
    rid = "wrapped:bad:shape"
    fr.set(rid, dump_json_str({}))
    _test_hooks.redis_factory = lambda url: fr

    client = TestClient(create_app())
    r = client.get(f"/v1/wrapped/download/{rid}")
    assert r.status_code == 400
    fr.assert_only_called({"sadd", "set", "expire", "get"})
