from __future__ import annotations

import pytest
from platform_workers.redis import _load_redis_error_class
from platform_workers.testing import FakeRedis

from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry

_RedisError: type[BaseException] = _load_redis_error_class()


class _FlakyGetRedis(FakeRedis):
    """FakeRedis that fails on first get call then returns value."""

    def __init__(self) -> None:
        super().__init__()
        self._get_calls = 0

    def get(self, key: str) -> str | None:
        self._record("get", key)
        self._get_calls += 1
        if self._get_calls == 1:
            raise _RedisError("boom")
        return "v"


class _AlwaysFailSetRedis(FakeRedis):
    """FakeRedis that always fails on set."""

    def set(self, key: str, value: str) -> bool:
        self._record("set", key, value)
        raise _RedisError("boom")


def test_get_with_retry_succeeds_after_retry() -> None:
    fake = _FlakyGetRedis()
    v = get_with_retry(fake, "k", attempts=2)
    assert v == "v"
    fake.assert_only_called({"get"})


def test_set_with_retry_exhausts_and_raises() -> None:
    fake = _AlwaysFailSetRedis()
    with pytest.raises(_RedisError):
        set_with_retry(fake, "k", "v", attempts=2)
    fake.assert_only_called({"set"})
