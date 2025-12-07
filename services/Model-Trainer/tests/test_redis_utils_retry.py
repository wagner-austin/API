from __future__ import annotations

import pytest
from platform_workers.testing import FakeRedis
from pytest import MonkeyPatch
from redis.exceptions import RedisError

from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry


def test_get_with_retry_succeeds_after_retry(monkeypatch: MonkeyPatch) -> None:
    fake = FakeRedis()
    state: dict[str, int] = {"calls": 0}

    def flaky_get(key: str) -> str | None:
        state["calls"] += 1
        if state["calls"] == 1:
            raise RedisError("boom")
        return "v"

    monkeypatch.setattr(fake, "get", flaky_get)
    v = get_with_retry(fake, "k", attempts=2)
    assert v == "v"
    # get is monkeypatched so no calls recorded
    fake.assert_only_called(set())


def test_set_with_retry_exhausts_and_raises(monkeypatch: MonkeyPatch) -> None:
    fake = FakeRedis()

    def flaky_set(
        key: str,
        value: str,
    ) -> bool | str | None:
        raise RedisError("boom")

    monkeypatch.setattr(fake, "set", flaky_set)

    with pytest.raises(RedisError):
        set_with_retry(fake, "k", "v", attempts=2)
    # set is monkeypatched so no calls recorded
    fake.assert_only_called(set())
