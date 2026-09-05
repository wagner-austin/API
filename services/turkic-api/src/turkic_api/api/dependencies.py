from __future__ import annotations

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from platform_core.logging import get_logger
from platform_core.queues import TURKIC_QUEUE
from platform_workers.redis import RedisStrProto
from platform_workers.rq_harness import QueueProtocol, connecting_queue

from turkic_api import _test_hooks
from turkic_api.api.config import Settings, load_settings
from turkic_api.api.types import (
    LoggerProtocol,
)


def get_settings() -> Settings:
    """Dependency: typed application settings from environment."""
    return load_settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_redis(settings: SettingsDep) -> Generator[RedisStrProto, None, None]:
    """Dependency: typed Redis (strings) using URL from settings; closes on teardown."""
    client = _test_hooks.redis_factory(settings["redis_url"])
    try:
        yield client
    finally:
        client.close()


def get_request_logger() -> LoggerProtocol:
    """Dependency: request-scoped logger (delegates to global logger)."""
    return get_logger(__name__)


def get_queue(settings: SettingsDep) -> QueueProtocol:
    """Dependency: RQ queue bound to a dedicated binary Redis connection.

    Uses shared platform helpers and a fixed queue name from platform_core.
    Imports RQ at runtime to allow strict tests to inject fakes. Return type is
    a minimal JobLike object to avoid leaking untyped values.
    """
    return connecting_queue(TURKIC_QUEUE, settings["redis_url"])
