from __future__ import annotations

import pytest
from platform_workers.testing import FakeRedis


@pytest.fixture(autouse=True)
def _readyz_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide typed Redis stub and REDIS_URL for /readyz in tests."""
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    import data_bank_api.api.routes.health as health_mod

    def _rf(url: str) -> FakeRedis:
        r = FakeRedis()
        r.sadd("rq:workers", "worker-1")  # Simulate one worker
        return r

    monkeypatch.setattr(health_mod, "redis_for_kv", _rf)
