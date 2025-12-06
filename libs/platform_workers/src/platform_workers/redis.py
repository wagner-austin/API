from __future__ import annotations

import importlib as _importlib
from typing import Protocol, TypedDict, runtime_checkable


class PubSubMessage(TypedDict, total=False):
    """Typed dict for Redis PubSub messages."""

    type: str
    pattern: str | None
    channel: str
    data: str | int | None


# Expose runtime redis module for test monkeypatching
redis = __import__("redis")


class RedisError(Exception):
    """Platform-wide Redis error base class.

    This wraps the underlying redis.exceptions.RedisError to provide a stable
    exception type that services can catch without importing redis directly.
    """


class _RedisExceptionsModule(Protocol):
    """Protocol for redis.exceptions module."""

    RedisError: type[BaseException]


class _RedisModuleWithExceptions(Protocol):
    """Protocol for redis module with exceptions."""

    exceptions: _RedisExceptionsModule


def _load_redis_error_class() -> type[BaseException]:
    """Load the redis.exceptions.RedisError class dynamically."""
    redis_mod: _RedisModuleWithExceptions = __import__("redis")
    error_cls: type[BaseException] = redis_mod.exceptions.RedisError
    return error_cls


def is_redis_error(exc: BaseException) -> bool:
    """Check if an exception is a Redis error.

    Use this instead of catching redis.exceptions.RedisError directly.
    """
    redis_error_cls = _load_redis_error_class()
    return isinstance(exc, redis_error_cls)


# Internal Protocols to avoid importing redis at module level
class _RedisStrClient(Protocol):
    """Protocol for redis.Redis[str] to avoid module-level import."""

    def ping(self, **kwargs: str | int | float | bool | None) -> bool | str: ...
    def set(self, name: str, value: str) -> bool | str | None: ...
    def get(self, name: str) -> str | None: ...
    def delete(self, *names: str) -> int: ...
    def expire(self, name: str, time: int) -> bool: ...
    def hset(self, name: str, mapping: dict[str, str]) -> int: ...
    def hget(self, name: str, key: str) -> str | None: ...
    def hgetall(self, name: str) -> dict[str, str]: ...
    def publish(self, channel: str, message: str) -> int: ...
    def scard(self, name: str) -> int: ...
    def sadd(self, name: str, value: str) -> int: ...
    def sismember(self, name: str, value: str) -> bool | int: ...
    def close(self) -> None: ...


class _RedisBytesClient(Protocol):
    """Protocol for redis.Redis[bytes] to avoid module-level import."""

    def ping(self, **kwargs: str | int | float | bool | None) -> bool | bytes: ...
    def close(self) -> None: ...


class _RedisBytesModule(Protocol):
    def from_url(
        self,
        url: str,
        *,
        decode_responses: bool,
        socket_connect_timeout: float,
        socket_timeout: float,
        retry_on_timeout: bool,
    ) -> _RedisBytesClient: ...


class _RedisStrModule(Protocol):
    def from_url(
        self,
        url: str,
        *,
        encoding: str,
        decode_responses: bool,
        socket_connect_timeout: float,
        socket_timeout: float,
        retry_on_timeout: bool,
    ) -> _RedisStrClient: ...


# Async Redis protocols for pubsub operations
class _RedisPubSubClient(Protocol):
    """Protocol for redis.asyncio PubSub to avoid module-level import."""

    async def subscribe(self, *channels: str) -> None: ...

    async def get_message(
        self, *, ignore_subscribe_messages: bool = True, timeout: float = 1.0
    ) -> PubSubMessage | None: ...

    async def close(self) -> None: ...


class _RedisAsyncClient(Protocol):
    """Protocol for redis.asyncio client to avoid module-level import."""

    def pubsub(self) -> _RedisPubSubClient: ...


class _RedisAsyncioModule(Protocol):
    """Protocol for redis.asyncio module to avoid module-level import."""

    def from_url(self, url: str, *, encoding: str, decode_responses: bool) -> _RedisAsyncClient: ...


def _load_redis_asyncio_module() -> _RedisAsyncioModule:
    mod: _RedisAsyncioModule = _importlib.import_module("redis.asyncio")
    return mod


def _load_redis_bytes_module() -> _RedisBytesModule:
    return __import__("redis")


def _load_redis_str_module() -> _RedisStrModule:
    return __import__("redis")


@runtime_checkable
class RedisStrProto(Protocol):
    def ping(self, **kwargs: str | int | float | bool | None) -> bool: ...

    def set(self, key: str, value: str) -> bool | str | None: ...

    def get(self, key: str) -> str | None: ...

    def delete(self, key: str) -> int: ...

    def expire(self, key: str, time: int) -> bool: ...

    def hset(self, key: str, mapping: dict[str, str]) -> int: ...

    def hget(self, key: str, field: str) -> str | None: ...

    def hgetall(self, key: str) -> dict[str, str]: ...

    def publish(self, channel: str, message: str) -> int: ...

    def scard(self, key: str) -> int: ...

    def sadd(self, key: str, member: str) -> int: ...

    def sismember(self, key: str, member: str) -> bool: ...

    def close(self) -> None: ...


@runtime_checkable
class RedisBytesProto(Protocol):
    def ping(self, **kwargs: str | int | float | bool | None) -> bool: ...

    def close(self) -> None: ...


@runtime_checkable
class RedisPubSubProto(Protocol):
    """Public protocol for async Redis PubSub operations."""

    async def subscribe(self, *channels: str) -> None: ...

    async def get_message(
        self, *, ignore_subscribe_messages: bool = True, timeout: float = 1.0
    ) -> PubSubMessage | None: ...

    async def close(self) -> None: ...


@runtime_checkable
class RedisAsyncProto(Protocol):
    """Public protocol for async Redis client with pubsub support."""

    def pubsub(self) -> RedisPubSubProto: ...


class _RedisStrAdapter(RedisStrProto):
    def __init__(self, inner: _RedisStrClient) -> None:
        self._inner = inner

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return bool(self._inner.ping(**kwargs))

    def set(self, key: str, value: str) -> bool | str:
        result = self._inner.set(name=key, value=value)
        return bool(result)

    def get(self, key: str) -> str | None:
        return self._inner.get(name=key)

    def delete(self, key: str) -> int:
        return int(self._inner.delete(key))

    def expire(self, key: str, time: int) -> bool:
        return bool(self._inner.expire(name=key, time=time))

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        return int(self._inner.hset(name=key, mapping=mapping))

    def hget(self, key: str, field: str) -> str | None:
        return self._inner.hget(name=key, key=field)

    def hgetall(self, key: str) -> dict[str, str]:
        raw = self._inner.hgetall(name=key)
        return {str(k): str(v) for k, v in raw.items()}

    def publish(self, channel: str, message: str) -> int:
        return self._inner.publish(channel, message)

    def scard(self, key: str) -> int:
        return int(self._inner.scard(name=key))

    def sadd(self, key: str, member: str) -> int:
        return int(self._inner.sadd(name=key, value=member))

    def sismember(self, key: str, member: str) -> bool:
        return bool(self._inner.sismember(name=key, value=member))

    def close(self) -> None:
        self._inner.close()


class _RedisBytesAdapter(RedisBytesProto):
    def __init__(self, inner: _RedisBytesClient) -> None:
        self._inner = inner

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return bool(self._inner.ping(**kwargs))

    def close(self) -> None:
        self._inner.close()


def _redis_client_for_rq(url: str) -> RedisBytesProto:
    redis_mod = _load_redis_bytes_module()
    client = redis_mod.from_url(
        url,
        decode_responses=False,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
        retry_on_timeout=True,
    )
    return _RedisBytesAdapter(client)


def redis_raw_for_rq(url: str) -> _RedisBytesClient:
    """Return a raw Redis client for use with RQ.

    RQ needs access to all Redis queue operations (lpush, rpop, etc.) which our
    RedisBytesProto adapter doesn't expose. This function returns the actual
    Redis client instance, not the adapter.

    Use `redis_for_rq()` for non-RQ binary Redis operations where only
    ping/close are needed.
    """
    redis_mod = _load_redis_bytes_module()
    return redis_mod.from_url(
        url,
        decode_responses=False,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
        retry_on_timeout=True,
    )


def _redis_from_url_str(url: str) -> RedisStrProto:
    redis_mod = _load_redis_str_module()
    client = redis_mod.from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
        retry_on_timeout=True,
    )
    return _RedisStrAdapter(client)


def redis_for_kv(url: str) -> RedisStrProto:
    """Strictly typed Redis client for string key-value and hashes."""
    return _redis_from_url_str(url)


def _redis_from_url_bytes(url: str) -> RedisBytesProto:
    return _redis_client_for_rq(url)


def redis_for_rq(url: str) -> RedisBytesProto:
    """Strictly typed Redis client for use with RQ (binary responses)."""
    return _redis_from_url_bytes(url)


def _redis_async_from_url(url: str) -> RedisAsyncProto:
    """Internal helper to create async Redis client from URL."""
    redis_asyncio = _load_redis_asyncio_module()
    client: RedisAsyncProto = redis_asyncio.from_url(url, encoding="utf-8", decode_responses=True)
    return client


def redis_for_pubsub(url: str) -> RedisAsyncProto:
    """Strictly typed async Redis client for pubsub operations."""
    return _redis_async_from_url(url)


__all__ = [
    "PubSubMessage",
    "RedisAsyncProto",
    "RedisBytesProto",
    "RedisError",
    "RedisPubSubProto",
    "RedisStrProto",
    "_RedisBytesClient",
    "_load_redis_error_class",
    "_redis_async_from_url",
    "_redis_client_for_rq",
    "is_redis_error",
    "redis_for_kv",
    "redis_for_pubsub",
    "redis_for_rq",
    "redis_raw_for_rq",
]
