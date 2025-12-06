from __future__ import annotations

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from platform_core.logging import get_logger
from platform_core.queues import TURKIC_QUEUE
from platform_workers.redis import RedisStrProto, redis_for_kv, redis_raw_for_rq
from platform_workers.rq_harness import RQClientQueue, rq_queue

from turkic_api.api.config import Settings, settings_from_env
from turkic_api.api.types import (
    LoggerProtocol,
    QueueProtocol,
    RQJobLike,
    RQRetryLike,
    UnknownJson,
    _EnqCallable,
)


def get_settings() -> Settings:
    """Dependency: typed application settings from environment."""
    return settings_from_env()


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_redis(settings: SettingsDep) -> Generator[RedisStrProto, None, None]:
    """Dependency: typed Redis (strings) using URL from settings; closes on teardown."""
    client = redis_for_kv(settings["redis_url"])
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
    redis_url = settings["redis_url"]
    queue_name = TURKIC_QUEUE

    class _QueueAdapter:
        def __init__(self) -> None:
            self._url = redis_url
            self._name = queue_name

        def enqueue(
            self,
            func: str | _EnqCallable,
            *args: UnknownJson,
            job_timeout: int | None = None,
            result_ttl: int | None = None,
            failure_ttl: int | None = None,
            retry: RQRetryLike | None = None,
            description: str | None = None,
        ) -> RQJobLike:
            fref = func if isinstance(func, str) else str(func)
            conn = redis_raw_for_rq(self._url)
            q: RQClientQueue = rq_queue(self._name, connection=conn)
            job: RQJobLike = q.enqueue(
                fref,
                *args,
                job_timeout=job_timeout,
                result_ttl=result_ttl,
                failure_ttl=failure_ttl,
                retry=retry,
                description=description,
            )
            return job

    return _QueueAdapter()
