from __future__ import annotations

from platform_workers.redis import (
    RedisAsyncProto,
    RedisPubSubProto,
)

from .subscriber import MessageSource as _MsgSourceProto


class RedisPubSubSource(_MsgSourceProto):
    def __init__(self, redis_url: str) -> None:
        from .testing import hooks

        self._redis: RedisAsyncProto = hooks.async_client_from_url(redis_url)
        self._pubsub: RedisPubSubProto | None = None

    async def subscribe(self, channel: str) -> None:
        ps = self._redis.pubsub()
        await ps.subscribe(channel)
        self._pubsub = ps

    async def get(self) -> str | None:
        ps = self._pubsub
        if ps is None:
            return None
        msg = await ps.get_message(ignore_subscribe_messages=True, timeout=1.0)
        if not msg:
            return None
        data = msg.get("data")
        return data if isinstance(data, str) else None

    async def close(self) -> None:
        ps = self._pubsub
        if ps is not None:
            await ps.close()


__all__ = ["RedisPubSubSource"]
