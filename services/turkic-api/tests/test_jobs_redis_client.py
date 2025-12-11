"""Tests for Redis client creation in jobs module."""

from __future__ import annotations

from platform_workers.testing import FakeRedis

from turkic_api import _test_hooks
from turkic_api.api.jobs import _get_redis_client


def test_get_redis_client_uses_shared_adapter() -> None:
    """Test that _get_redis_client uses the redis_factory hook."""
    seen: list[str] = []
    captured: list[FakeRedis] = []

    def fake_redis_for_kv(url: str) -> FakeRedis:
        seen.append(url)
        r = FakeRedis()
        captured.append(r)
        return r

    _test_hooks.redis_factory = fake_redis_for_kv

    client = _get_redis_client("redis://localhost:6379/0")
    assert client.hset("k", {"a": "1"}) == 1
    client.close()
    assert seen == ["redis://localhost:6379/0"]
    for r in captured:
        r.assert_only_called({"hset", "expire", "close"})
