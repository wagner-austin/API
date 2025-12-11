from __future__ import annotations

from platform_workers.redis import _load_redis_error_class
from platform_workers.testing import FakeRedis as _FakeRedis

from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry

_RedisError: type[BaseException] = _load_redis_error_class()


class _FlakyGetRedis(_FakeRedis):
    """FakeRedis that fails on first get call."""

    def __init__(self) -> None:
        super().__init__()
        self._get_calls = 0

    def get(self, key: str) -> str | None:
        self._record("get", key)
        self._get_calls += 1
        if self._get_calls == 1:
            raise _RedisError("transient")
        return self._strings.get(key)


class _FlakySetRedis(_FakeRedis):
    """FakeRedis that fails on first set call."""

    def __init__(self) -> None:
        super().__init__()
        self._set_calls = 0

    def set(self, key: str, value: str) -> bool:
        self._record("set", key, value)
        self._set_calls += 1
        if self._set_calls == 1:
            raise _RedisError("transient")
        self._strings[key] = value
        return True


def test_get_with_retry_succeeds_after_transient() -> None:
    client = _FlakyGetRedis()
    client._strings["a"] = "b"  # Seed data directly
    val = get_with_retry(client, "a")
    assert val == "b"
    client.assert_only_called({"get"})


def test_set_with_retry_succeeds_after_transient() -> None:
    client = _FlakySetRedis()
    set_with_retry(client, "a", "b")
    v_after = client.get("a")
    assert v_after == "b"
    client.assert_only_called({"set", "get"})
