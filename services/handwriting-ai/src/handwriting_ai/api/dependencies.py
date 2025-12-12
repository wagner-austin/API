from __future__ import annotations

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from platform_core.config import _require_env_str
from platform_core.queues import DIGITS_QUEUE
from platform_workers.redis import RedisStrProto
from platform_workers.rq_harness import RQClientQueue, RQJobLike

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import LoggerInstanceProtocol
from handwriting_ai.api.types import (
    QueueProtocol,
    RQRetryLike,
    UnknownJson,
    _EnqCallable,
)
from handwriting_ai.config import Settings


def get_settings() -> Settings:
    """Dependency: typed application settings from environment."""
    return _test_hooks.load_settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]


def _get_redis_url() -> str:
    """Get Redis URL from environment, raising if not set."""
    return _require_env_str("REDIS_URL")


def get_redis() -> Generator[RedisStrProto, None, None]:
    """Dependency: typed Redis (strings) using URL from env; closes on teardown."""
    redis_url = _get_redis_url()
    client = _test_hooks.redis_factory(redis_url)
    try:
        yield client
    finally:
        client.close()


def get_request_logger() -> LoggerInstanceProtocol:
    """Dependency: request-scoped logger (delegates to global logger)."""
    return _test_hooks.get_logger(__name__)


def get_queue() -> QueueProtocol:
    """Dependency: RQ queue bound to a dedicated binary Redis connection.

    Uses shared platform helpers and a fixed queue name from platform_core.
    Imports RQ at runtime to allow strict tests to inject fakes. Return type is
    a minimal JobLike object to avoid leaking untyped values.
    """
    redis_url = _get_redis_url()
    queue_name = DIGITS_QUEUE

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
            conn = _test_hooks.rq_conn(self._url)
            q: RQClientQueue = _test_hooks.rq_queue_factory(self._name, conn)
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


RedisDep = Annotated[RedisStrProto, Depends(get_redis)]
LoggerDep = Annotated[LoggerInstanceProtocol, Depends(get_request_logger)]
QueueDep = Annotated[QueueProtocol, Depends(get_queue)]
