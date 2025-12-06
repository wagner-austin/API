from __future__ import annotations

from platform_core.health import HealthResponse
from platform_workers.testing import FakeRedis

from music_wrapped_api.health import healthz_endpoint, readyz_endpoint


def test_healthz_endpoint_ok() -> None:
    res: HealthResponse = healthz_endpoint()
    assert res["status"] == "ok"


def test_readyz_endpoint_ready() -> None:
    r = FakeRedis()
    r.sadd("rq:workers", "worker-a")
    res = readyz_endpoint(r)
    assert res["status"] == "ready"
