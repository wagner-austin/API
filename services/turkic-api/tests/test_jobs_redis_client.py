from __future__ import annotations

import pytest
from platform_workers.testing import FakeRedis

import turkic_api.api.jobs as jobs


def test_get_redis_client_uses_shared_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    captured: list[FakeRedis] = []

    def fake_redis_for_kv(url: str) -> FakeRedis:
        seen.append(url)
        r = FakeRedis()
        captured.append(r)
        return r

    monkeypatch.setattr(jobs, "redis_for_kv", fake_redis_for_kv)

    client = jobs._get_redis_client("redis://localhost:6379/0")
    assert client.hset("k", {"a": "1"}) == 1
    client.close()
    assert seen == ["redis://localhost:6379/0"]
    for r in captured:
        r.assert_only_called({"hset", "expire", "close"})
