"""_fakes: LoggerProtocol and related definitions."""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import NamedTuple, Protocol

from platform_workers.rq_harness import _JsonValue

from .redis import (
    PubSubMessage,
    RedisAsyncProto,
    RedisPubSubProto,
)


class LoggerProtocol(Protocol):
    """Protocol for a minimal structured logger interface."""

    def debug(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None: ...

    def info(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None: ...

    def warning(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None: ...

    def error(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None: ...


class LogRecord(NamedTuple):
    """Record of a log message."""

    level: str
    msg: str
    args: tuple[_JsonValue, ...]
    extra: Mapping[str, _JsonValue] | None


class FakeLogger:
    """Fake logger for testing."""

    def __init__(self) -> None:
        self.records: list[LogRecord] = []

    def debug(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None:
        self.records.append(LogRecord("debug", msg, args, extra))

    def info(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None:
        self.records.append(LogRecord("info", msg, args, extra))

    def warning(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None:
        self.records.append(LogRecord("warning", msg, args, extra))

    def error(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None:
        self.records.append(LogRecord("error", msg, args, extra))


# =============================================================================
# Async Redis Fakes for PubSub Testing
# =============================================================================


class FakePubSub(RedisPubSubProto):
    """Fake async Redis PubSub client for testing."""

    def __init__(self) -> None:
        self.subscriptions: list[str] = []
        self._messages: list[PubSubMessage] = []
        self._closed = False

    def inject_message(self, channel: str, data: str) -> None:
        """Inject a message to be returned by get_message."""
        msg: PubSubMessage = {"type": "message", "pattern": None, "channel": channel, "data": data}
        self._messages.append(msg)

    async def subscribe(self, *channels: str) -> None:
        self.subscriptions.extend(channels)

    async def get_message(
        self, *, ignore_subscribe_messages: bool = True, timeout: float = 1.0
    ) -> PubSubMessage | None:
        if self._messages:
            return self._messages.pop(0)
        return None

    async def close(self) -> None:
        self._closed = True


class FakeAsyncRedis(RedisAsyncProto):
    """Fake async Redis client for PubSub testing."""

    def __init__(self) -> None:
        self._pubsub = FakePubSub()

    def pubsub(self) -> FakePubSub:
        return self._pubsub


# =============================================================================
# Redis Module Fakes for Runtime Import Testing
# =============================================================================
