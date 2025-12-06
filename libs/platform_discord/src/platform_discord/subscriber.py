from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Generic, Protocol, TypeVar

E = TypeVar("E")


class MessageSource(Protocol):
    async def subscribe(self, channel: str) -> None: ...

    async def get(self) -> str | None: ...

    async def close(self) -> None: ...


Handler = Callable[[E], Awaitable[None]]
Decoder = Callable[[str], E | None]


class RedisEventSubscriber(Generic[E]):
    def __init__(
        self,
        channel: str,
        source: MessageSource,
        decode: Decoder[E],
        handle: Handler[E],
    ) -> None:
        self.channel = channel
        self.source = source
        self.decode = decode
        self.handle = handle

    async def run(self, *, limit: int | None = None) -> None:
        await self.source.subscribe(self.channel)
        count = 0
        while True:
            if limit is not None and count >= limit:
                break
            payload = await self.source.get()
            count += 1
            if payload is None:
                # Avoid busy spin; yield to event loop
                import asyncio as _asyncio

                await _asyncio.sleep(0)
                continue
            event = self.decode(payload)
            if event is None:
                import asyncio as _asyncio

                await _asyncio.sleep(0)
                continue
            await self.handle(event)
        await self.source.close()


__all__ = ["MessageSource", "RedisEventSubscriber"]
