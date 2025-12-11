from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis

from music_wrapped_api import _test_hooks
from music_wrapped_api.api.main import create_app


def test_readyz_degraded_without_worker() -> None:
    r = FakeRedis()
    # No workers registered
    _test_hooks.redis_factory = lambda url: r

    client = TestClient(create_app())
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body_raw = load_json_str(resp.text)
    if type(body_raw) is not dict:
        pytest.fail("expected dict response body")
    body: dict[str, JSONValue] = body_raw
    assert body.get("status") == "degraded"
    assert body.get("reason") == "no-worker"
    r.assert_only_called({"ping", "scard", "close"})


def test_readyz_ready_with_worker() -> None:
    r = FakeRedis()
    r.sadd("rq:workers", "worker-1")
    _test_hooks.redis_factory = lambda url: r

    client = TestClient(create_app())
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body_raw = load_json_str(resp.text)
    if type(body_raw) is not dict:
        pytest.fail("expected dict response body")
    body: dict[str, JSONValue] = body_raw
    assert body.get("status") == "ready"
    r.assert_only_called({"sadd", "ping", "scard", "close"})
