from __future__ import annotations

import logging

from platform_workers.redis import RedisAsyncProto, _redis_async_from_url


def test_redis_from_url_returns_client_without_connecting() -> None:
    # Import the helper and call it to cover import/runtime branch
    conn: RedisAsyncProto = _redis_async_from_url("redis://example")
    # The returned object should expose pubsub() per our typing expectations
    _ = conn.pubsub


logger = logging.getLogger(__name__)
