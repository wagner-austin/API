from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Annotated

from fastapi import Depends
from platform_workers.redis import RedisStrProto

from .config import Settings
from .dependencies import (
    get_queue as _get_queue_dep,
)
from .dependencies import (
    get_redis as _get_redis_dep,
)
from .dependencies import (
    get_request_logger as _get_request_logger,
)
from .dependencies import (
    get_settings as _get_settings_dep,
)
from .types import LoggerProtocol, QueueProtocol

SettingsProvider = Callable[[], Settings]
LoggerProvider = Callable[[], LoggerProtocol]

# Provider types for dependency injection
RedisProviderType = Callable[[Settings], RedisStrProto | Generator[RedisStrProto, None, None]]
QueueProviderType = Callable[[], QueueProtocol]


class _ProviderContext:
    def __init__(self) -> None:
        self.settings_provider: SettingsProvider | None = None
        self.redis_provider: RedisProviderType | None = None
        self.queue_provider: QueueProviderType | None = None
        self.logger_provider: LoggerProvider | None = None


provider_context = _ProviderContext()


def get_settings_from_context() -> Settings:
    if provider_context.settings_provider is not None:
        return provider_context.settings_provider()
    return _get_settings_dep()


def get_redis_from_context(
    settings: Annotated[Settings, Depends(get_settings_from_context)],
) -> Generator[RedisStrProto, None, None]:
    if provider_context.redis_provider is not None:
        result = provider_context.redis_provider(settings)
        if isinstance(result, Generator):
            yield from result
        else:
            yield result
    else:
        yield from _get_redis_dep(settings)


def get_queue_from_context(
    settings: Annotated[Settings, Depends(get_settings_from_context)],
) -> QueueProtocol:
    if provider_context.queue_provider is not None:
        return provider_context.queue_provider()
    return _get_queue_dep(settings)


def get_logger_from_context() -> LoggerProtocol:
    if provider_context.logger_provider is not None:
        return provider_context.logger_provider()
    return _get_request_logger()


__all__ = [
    "LoggerProvider",
    "QueueProviderType",
    "RedisProviderType",
    "SettingsProvider",
    "get_logger_from_context",
    "get_queue_from_context",
    "get_redis_from_context",
    "get_settings_from_context",
    "provider_context",
]
