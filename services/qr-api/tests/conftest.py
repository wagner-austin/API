from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from qr_api.api import _test_hooks


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore all hooks after each test."""
    original_redis = _test_hooks.redis_factory
    original_env = _test_hooks.get_env
    yield
    _test_hooks.redis_factory = original_redis
    _test_hooks.get_env = original_env


@pytest.fixture(autouse=True)
def _readyz_redis() -> None:
    """Provide typed Redis stub for /readyz in tests."""
    _env_values: dict[str, str] = {"REDIS_URL": "redis://ignored"}

    def _fake_env(key: str) -> str | None:
        return _env_values.get(key)

    def _rf(url: str) -> RedisStrProto:
        r = FakeRedis()
        r.sadd("rq:workers", "worker-1")  # Simulate one worker
        return r

    _test_hooks.get_env = _fake_env
    _test_hooks.redis_factory = _rf
