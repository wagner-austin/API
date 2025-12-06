from __future__ import annotations

import pytest

import turkic_api.api.jobs as jobs


def test_get_redis_client_uses_shared_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class _Redis:
        def __init__(self) -> None:
            self.closed = False

        def hset(self, key: str, mapping: dict[str, str]) -> int:
            return len(mapping)

        def close(self) -> None:
            self.closed = True

        def hgetall(self, key: str) -> dict[str, str]:
            return {}

        def publish(self, channel: str, message: str) -> int:
            return 1

        def ping(self, **kwargs: str | int | float | bool | None) -> bool:
            return True

        def set(self, key: str, value: str) -> bool:
            return True

        def get(self, key: str) -> str | None:
            return None

    def fake_redis_for_kv(url: str) -> _Redis:
        seen.append(url)
        return _Redis()

    monkeypatch.setattr(jobs, "redis_for_kv", fake_redis_for_kv)

    client = jobs._get_redis_client("redis://localhost:6379/0")
    assert client.hset("k", {"a": "1"}) == 1
    client.close()
    assert seen == ["redis://localhost:6379/0"]
