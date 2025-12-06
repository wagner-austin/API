from __future__ import annotations

import time

from platform_core.logging import get_logger
from platform_workers.redis import RedisStrProto, is_redis_error

_log = get_logger(__name__)


def get_with_retry(client: RedisStrProto, key: str, *, attempts: int = 3) -> str | None:
    delay = 0.01
    for i in range(attempts):
        try:
            return client.get(key)
        except Exception as exc:
            if not is_redis_error(exc):
                raise
            _log.warning("Redis get failed, attempt=%d/%d key=%s: %s", i + 1, attempts, key, exc)
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return None


def set_with_retry(client: RedisStrProto, key: str, value: str, *, attempts: int = 3) -> None:
    delay = 0.01
    for i in range(attempts):
        try:
            client.set(key, value)
            return
        except Exception as exc:
            if not is_redis_error(exc):
                raise
            _log.warning("Redis set failed, attempt=%d/%d key=%s: %s", i + 1, attempts, key, exc)
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
