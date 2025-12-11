"""Additional tests for main.py routes."""

from __future__ import annotations

from fastapi.testclient import TestClient
from platform_workers.testing import FakeRedis, FakeRedisError

from turkic_api import _test_hooks
from turkic_api.api.config import Settings
from turkic_api.api.main import RedisCombinedProtocol, create_app
from turkic_api.api.types import JSONValue, QueueProtocol, RQJobLike, RQRetryLike, _EnqCallable


class _QueueStub:
    def enqueue(
        self,
        func: str | _EnqCallable,
        *args: JSONValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike:
        class _Job(RQJobLike):
            def get_id(self) -> str:
                return "test-job-id"

        return _Job()


_captured_redis: list[FakeRedis] = []


def _clear_captured_redis() -> None:
    _captured_redis.clear()


def _redis_provider_stub(settings: Settings) -> RedisCombinedProtocol:
    r = FakeRedis()
    _captured_redis.append(r)
    return r


def _queue_provider_stub() -> QueueProtocol:
    return _QueueStub()


def test_get_job_result_404() -> None:
    _clear_captured_redis()
    app = create_app(
        redis_provider=_redis_provider_stub,
        queue_provider=_queue_provider_stub,
    )
    with TestClient(app) as client:
        r = client.get("/api/v1/jobs/doesnotexist/result")
        assert r.status_code == 404
    for redis in _captured_redis:
        redis.assert_only_called({"hgetall"})


def test_readyz_handles_redis_error() -> None:
    captured_err: list[FakeRedisError] = []

    def _redis_err_provider(settings: Settings) -> RedisCombinedProtocol:
        r = FakeRedisError()
        captured_err.append(r)
        return r

    # Use hook to make path appear to exist
    _test_hooks.path_exists = lambda p: True

    app = create_app(
        redis_provider=_redis_err_provider,
        queue_provider=_queue_provider_stub,
    )
    with TestClient(app) as c:
        r = c.get("/readyz")
        assert r.status_code == 503
        body: dict[str, str | None] = r.json()
        assert body["status"] == "degraded"
        assert body["reason"] == "redis error"
    for redis in captured_err:
        redis.assert_only_called({"ping", "close"})
