from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypeVar

import pytest

from platform_discord.subscriber import MessageSource, RedisEventSubscriber

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None

T = TypeVar("T")


@dataclass
class _FakeSource(MessageSource):
    messages: list[str | None]
    subscribed: str | None = None
    closed: bool = False

    async def subscribe(self, channel: str) -> None:
        self.subscribed = channel

    async def get(self) -> str | None:
        await asyncio.sleep(0)  # yield control
        if self.messages:
            return self.messages.pop(0)
        return None

    async def close(self) -> None:
        self.closed = True


async def _collect(ev: T, out: list[T]) -> None:
    out.append(ev)


@pytest.mark.asyncio
async def test_subscriber_processes_valid_and_ignores_invalid() -> None:
    def _decode(payload: str) -> dict[str, UnknownJson] | None:
        s = payload.strip()
        if not (s.startswith("{") and s.endswith("}")):
            return None
        # Minimal check: look for our marker type
        if '"digits.train.started.v1"' in s:
            return {"type": "digits.train.started.v1"}
        return None

    good = (
        '{"type":"digits.train.started.v1","request_id":"r","user_id":1,'
        '"model_id":"m","run_id":null,"ts":"t","total_epochs":1,"queue":"q"}'
    )
    bad = "not-json"
    src = _FakeSource(messages=[good, bad, None])
    got: list[dict[str, UnknownJson]] = []
    sub: RedisEventSubscriber[dict[str, UnknownJson]] = RedisEventSubscriber(
        channel="digits:events",
        source=src,
        decode=_decode,
        handle=lambda e: _collect(e, got),
    )
    await sub.run(limit=5)
    assert src.subscribed == "digits:events"
    assert src.closed is True
    assert len(got) == 1


@pytest.mark.asyncio
async def test_subscriber_handles_none_messages() -> None:
    def _decode(_payload: str) -> dict[str, UnknownJson] | None:
        return None

    src = _FakeSource(messages=[None, None])
    got: list[dict[str, UnknownJson]] = []
    sub: RedisEventSubscriber[dict[str, UnknownJson]] = RedisEventSubscriber(
        channel="digits:events",
        source=src,
        decode=_decode,
        handle=lambda e: _collect(e, got),
    )
    await sub.run(limit=3)
    assert src.closed is True
    assert got == []
