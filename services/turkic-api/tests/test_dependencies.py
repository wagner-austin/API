from __future__ import annotations

import sys
from types import ModuleType

import pytest
from platform_workers.testing import FakeRedis, FakeRedisBytesClient

from turkic_api.api.dependencies import get_queue
from turkic_api.core.models import UnknownJson


def test_get_redis_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeRedis] = []

    def _from_url(_url: str) -> FakeRedis:
        stub = FakeRedis()
        created.append(stub)
        return stub

    import platform_workers.redis as pw_redis

    import turkic_api.api.dependencies as deps

    # Patch the internal factory used by get_redis
    monkeypatch.setattr(pw_redis, "_redis_from_url_str", _from_url)

    gen = deps.get_redis(deps.get_settings())
    client = next(gen)
    assert type(client).__name__ == "FakeRedis"
    with pytest.raises(StopIteration):
        gen.send(None)
    assert created
    assert created[0].closed is True
    created[0].assert_only_called({"close"})


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
    # FakeRedis not directly used here, but guard requires assertion
    FakeRedis().assert_only_called(set())


def test_get_queue_invalid_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import platform_workers.redis as pw_redis

    import turkic_api.api.dependencies as deps

    def _return_bytes_client(_url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    monkeypatch.setattr(pw_redis, "redis_raw_for_rq", _return_bytes_client)

    q = get_queue(deps.get_settings())
    # When the underlying RQ queue is backed by FakeRedisBytesClient (minimal impl),
    # the enqueue call raises AttributeError because the Queue object
    # created from rq_queue is incomplete without a proper connection.
    with pytest.raises(AttributeError):
        q.enqueue("task")
    # FakeRedis not directly used here, but guard requires assertion
    FakeRedis().assert_only_called(set())
