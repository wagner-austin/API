from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch
from platform_workers.testing import FakeRedis as _FakeRedis
from redis.exceptions import RedisError

from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry


def test_get_with_retry_succeeds_after_transient(monkeypatch: MonkeyPatch) -> None:
    client = _FakeRedis()
    client.set("a", "b")
    calls: dict[str, int] = {"n": 0}

    original_get = client.get

    def flaky_get(key: str) -> str | None:
        if calls["n"] == 0:
            calls["n"] += 1
            raise RedisError("transient")
        return original_get(key)

    monkeypatch.setattr(client, "get", flaky_get, raising=True)
    val = get_with_retry(client, "a")
    assert val == "b"
    client.assert_only_called({"set", "get"})


def test_set_with_retry_succeeds_after_transient(monkeypatch: MonkeyPatch) -> None:
    client = _FakeRedis()
    calls: dict[str, int] = {"n": 0}

    original_set = client.set

    def flaky_set(key: str, value: str) -> bool | str | None:
        if calls["n"] == 0:
            calls["n"] += 1
            raise RedisError("transient")
        return original_set(key, value)

    monkeypatch.setattr(client, "set", flaky_set, raising=True)
    set_with_retry(client, "a", "b")
    v_after = client.get("a")
    assert v_after == "b"
    client.assert_only_called({"set", "get"})
