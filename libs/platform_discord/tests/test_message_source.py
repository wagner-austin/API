from __future__ import annotations

import asyncio

import pytest
from platform_workers import redis as redis_mod
from platform_workers.redis import PubSubMessage

import platform_discord.message_source as mod


class _FakePubSub:
    def __init__(self) -> None:
        self._channel: str | None = None
        self._queue: list[PubSubMessage | None] = []

    async def subscribe(self, *channels: str) -> None:
        self._channel = channels[0]

    async def get_message(
        self,
        *,
        ignore_subscribe_messages: bool = True,
        timeout: float = 1.0,
    ) -> PubSubMessage | None:
        await asyncio.sleep(0)
        return self._queue.pop(0) if self._queue else None

    async def close(self) -> None:
        await asyncio.sleep(0)


class _FakeConn:
    def __init__(self) -> None:
        self.pubsub_obj = _FakePubSub()

    def pubsub(self) -> _FakePubSub:
        return self.pubsub_obj


@pytest.mark.asyncio
async def test_message_source_subscribe_get_close(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeConn()

    def _fake_factory(_url: str) -> redis_mod.RedisAsyncProto:
        return fake

    monkeypatch.setattr(mod, "_redis_async_from_url", _fake_factory, raising=True)

    src = mod.RedisPubSubSource("redis://x")
    assert await src.get() is None
    await src.close()
    await src.subscribe("chan")

    out1 = await src.get()
    assert out1 is None

    fake.pubsub_obj._queue.append({"type": "message", "data": 123})
    out2 = await src.get()
    assert out2 is None

    fake.pubsub_obj._queue.append({"type": "message", "data": "hello"})
    out3 = await src.get()
    assert out3 == "hello"

    await src.close()
