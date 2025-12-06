from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from platform_workers.redis import _load_redis_error_class

from turkic_api.api.config import Settings
from turkic_api.api.main import RedisCombinedProtocol, create_app
from turkic_api.api.types import QueueProtocol, RQJobLike, RQRetryLike, _EnqCallable
from turkic_api.core.models import UnknownJson


class _RedisStub:
    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def close(self) -> None: ...
    def hgetall(self, _k: str) -> dict[str, str]:
        return {}

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        return 1

    def publish(self, channel: str, message: str) -> int:
        return 1

    def set(self, key: str, value: str) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def sadd(self, key: str, *values: str) -> int:
        return len(values)

    def scard(self, key: str) -> int:
        return 0

    def hget(self, key: str, field: str) -> str | None:
        return None

    def sismember(self, key: str, member: str) -> bool:
        return False


class _QueueStub:
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
        class _Job(RQJobLike):
            def get_id(self) -> str:
                return "test-job-id"

        return _Job()


class _RedisErr:
    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        error_cls = _load_redis_error_class()
        raise error_cls("unreachable")

    def close(self) -> None:
        pass

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        return 0

    def hgetall(self, _k: str) -> dict[str, str]:
        return {}

    def publish(self, channel: str, message: str) -> int:
        return 1

    def set(self, key: str, value: str) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def sadd(self, key: str, *values: str) -> int:
        return len(values)

    def scard(self, key: str) -> int:
        return 0

    def hget(self, key: str, field: str) -> str | None:
        return None

    def sismember(self, key: str, member: str) -> bool:
        return False


def _redis_provider_stub(settings: Settings) -> RedisCombinedProtocol:
    return _RedisStub()


def _queue_provider_stub() -> QueueProtocol:
    return _QueueStub()


def test_get_job_result_404() -> None:
    app = create_app(
        redis_provider=_redis_provider_stub,
        queue_provider=_queue_provider_stub,
    )
    with TestClient(app) as client:
        r = client.get("/api/v1/jobs/doesnotexist/result")
        assert r.status_code == 404


def test_readyz_handles_redis_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _redis_err_provider(settings: Settings) -> RedisCombinedProtocol:
        return _RedisErr()

    app = create_app(
        redis_provider=_redis_err_provider,
        queue_provider=_queue_provider_stub,
    )
    with TestClient(app) as c:
        # Force volume to appear mounted so we hit the Redis degraded branch
        from pathlib import Path

        def _exists(_: Path) -> bool:
            return True

        monkeypatch.setattr("pathlib.Path.exists", _exists, raising=False)
        r = c.get("/readyz")
        assert r.status_code == 503
        body: dict[str, str | None] = r.json()
        assert body["status"] == "degraded"
        assert body["reason"] == "redis error"
