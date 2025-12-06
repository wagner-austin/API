from __future__ import annotations

import pytest
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis


@pytest.fixture(autouse=True)
def _readyz_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide typed Redis stub for /readyz in tests.

    Tests that exercise /readyz will set REDIS_URL explicitly.
    """
    import transcript_api.app as app_mod
    import transcript_api.events as events_mod

    def _rf(url: str) -> RedisStrProto:
        r = FakeRedis()
        r.sadd("rq:workers", "worker-1")
        return r

    monkeypatch.setattr(app_mod, "redis_for_kv", _rf)
    monkeypatch.setattr(events_mod, "redis_for_kv", _rf)
    monkeypatch.setenv("REDIS_URL", "redis://test-redis")
