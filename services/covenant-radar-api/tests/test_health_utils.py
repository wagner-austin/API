"""Tests for health utility functions."""

from __future__ import annotations

from platform_workers.testing import FakeRedis

from covenant_radar_api.health import healthz_endpoint, readyz_endpoint


def test_healthz_endpoint_returns_ok() -> None:
    """Test healthz_endpoint returns ok status."""
    result = healthz_endpoint()
    assert result["status"] == "ok"


def test_readyz_endpoint_ready_with_workers() -> None:
    """Test readyz_endpoint returns ready when workers present."""
    fake_redis = FakeRedis()
    fake_redis.sadd("rq:workers", "worker-1")

    result = readyz_endpoint(fake_redis)

    assert result["status"] == "ready"
    assert result["reason"] is None
    fake_redis.assert_only_called({"sadd", "ping", "scard"})


def test_readyz_endpoint_degraded_no_workers() -> None:
    """Test readyz_endpoint returns degraded when no workers."""
    fake_redis = FakeRedis()

    result = readyz_endpoint(fake_redis)

    assert result["status"] == "degraded"
    assert result["reason"] == "no-worker"
    fake_redis.assert_only_called({"ping", "scard"})
