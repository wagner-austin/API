"""In-memory fakes for platform_workers.

The fakes are grouped by what they stand in for: _fakes_redis for the Redis
clients, _fakes_rq for the RQ jobs and queues, _fakes_logging for the logger,
and _fakes_modules for the importable module stand-ins."""

from __future__ import annotations

from platform_workers._fakes_logging import (
    FakeAsyncRedis,
    FakeLogger,
    FakePubSub,
    LoggerProtocol,
    LogRecord,
)
from platform_workers._fakes_modules import (
    FakeRedisAsyncioModule,
    FakeRedisBytesModule,
    FakeRedisStrModule,
    FakeRQModule,
    _FakeCurrentJob,
    _FakeRQQueueInternal,
    _FakeRQWorkerInternal,
)
from platform_workers._fakes_redis import (
    FakeRedis,
    FakeRedisBytesClient,
    FakeRedisClient,
    FakeRedisConditionalHsetError,
    FakeRedisConditionalHsetRedisError,
    FakeRedisError,
    FakeRedisHsetError,
    FakeRedisHsetRedisError,
    FakeRedisNonRedisError,
    FakeRedisNonRedisScardError,
    FakeRedisNoPong,
    FakeRedisPublishError,
    FakeRedisScardError,
    MethodCall,
    Published,
)
from platform_workers._fakes_rq import (
    EnqueuedJob,
    FakeFetchedJob,
    FakeJob,
    FakeQueue,
    FakeRetry,
)

__all__ = [
    "EnqueuedJob",
    "FakeAsyncRedis",
    "FakeFetchedJob",
    "FakeJob",
    "FakeLogger",
    "FakePubSub",
    "FakeQueue",
    "FakeRQModule",
    "FakeRedis",
    "FakeRedisAsyncioModule",
    "FakeRedisBytesClient",
    "FakeRedisBytesModule",
    "FakeRedisClient",
    "FakeRedisConditionalHsetError",
    "FakeRedisConditionalHsetRedisError",
    "FakeRedisError",
    "FakeRedisHsetError",
    "FakeRedisHsetRedisError",
    "FakeRedisNoPong",
    "FakeRedisNonRedisError",
    "FakeRedisNonRedisScardError",
    "FakeRedisPublishError",
    "FakeRedisScardError",
    "FakeRedisStrModule",
    "FakeRetry",
    "LogRecord",
    "LoggerProtocol",
    "MethodCall",
    "Published",
    "_FakeCurrentJob",
    "_FakeRQQueueInternal",
    "_FakeRQWorkerInternal",
]
