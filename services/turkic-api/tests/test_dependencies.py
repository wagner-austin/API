from __future__ import annotations

import sys
from types import ModuleType

import pytest

from turkic_api.api.dependencies import get_queue
from turkic_api.core.models import UnknownJson


class _RedisStub:
    def __init__(self) -> None:
        self.closed = False

    def ping(self, **_kwargs: UnknownJson) -> bool:
        return True

    def set(self, key: str, value: str) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        return len(mapping)

    def hgetall(self, key: str) -> dict[str, str]:
        return {}

    def publish(self, channel: str, message: str) -> int:
        return 1

    def close(self) -> None:
        self.closed = True

    def sadd(self, key: str, *values: str) -> int:
        return len(values)

    def scard(self, key: str) -> int:
        return 0


def test_get_redis_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[_RedisStub] = []

    def _from_url(_url: str) -> _RedisStub:
        stub = _RedisStub()
        created.append(stub)
        return stub

    import platform_workers.redis as pw_redis

    import turkic_api.api.dependencies as deps

    # Patch the internal factory used by get_redis
    monkeypatch.setattr(pw_redis, "_redis_from_url_str", _from_url)

    gen = deps.get_redis(deps.get_settings())
    client = next(gen)
    assert type(client).__name__ == "_RedisStub"
    with pytest.raises(StopIteration):
        gen.send(None)
    assert created
    assert created[0].closed is True


def test_get_queue_returns_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Q:
        def __init__(self, name: str, *, connection: UnknownJson) -> None:
            self.name = name
            self.connection = connection

    class _RQModule(ModuleType):
        Queue: type[_Q]

    dummy = _RQModule("rq")
    dummy.Queue = _Q
    sys.modules["rq"] = dummy

    import turkic_api.api.dependencies as deps

    # Call get_queue with settings dependency
    q = get_queue(deps.get_settings())
    assert callable(q.enqueue)


def test_get_queue_invalid_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import platform_workers.redis as pw_redis

    import turkic_api.api.dependencies as deps

    class _BadRedis:
        def ping(self, **_kwargs: UnknownJson) -> bool:
            return True

        def close(self) -> None:
            return None

    def _return_bad_client(_url: str) -> _BadRedis:
        return _BadRedis()

    monkeypatch.setattr(pw_redis, "redis_raw_for_rq", _return_bad_client)

    q = get_queue(deps.get_settings())
    # When the underlying RQ queue is backed by an invalid redis client,
    # the enqueue call raises AttributeError because the Queue object
    # created from rq_queue is incomplete without a proper connection.
    with pytest.raises(AttributeError):
        q.enqueue("task")
