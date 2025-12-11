"""Tests for turkic-api API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
from platform_workers.testing import FakeRedis, FakeRedisNoPong

from turkic_api import _test_hooks
from turkic_api.api.config import Settings
from turkic_api.api.main import RedisCombinedProtocol, create_app
from turkic_api.api.models import parse_job_response_json
from turkic_api.api.types import JSONValue, QueueProtocol, RQJobLike, RQRetryLike, _EnqCallable


class _QueueStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str | _EnqCallable, tuple[JSONValue, ...]]] = []

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
        self.calls.append((func, args))

        class _Job(RQJobLike):
            def get_id(self) -> str:
                return "test-job-id"

        return _Job()


# Capture created FakeRedis instances for assertion
_created_redis: list[FakeRedis] = []


def _redis_provider(settings: Settings) -> RedisCombinedProtocol:
    redis = FakeRedis()
    # Register a fake worker so health checks pass
    redis.sadd("rq:workers", "test-worker-1")
    _created_redis.append(redis)
    return redis


def _clear_redis_instances() -> None:
    _created_redis.clear()


def _queue_provider() -> QueueProtocol:
    return _QueueStub()


def test_healthz_always_ok() -> None:
    """Test /healthz liveness endpoint always returns ok."""
    _clear_redis_instances()
    app = create_app(redis_provider=_redis_provider, queue_provider=_queue_provider)
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body: dict[str, str] = resp.json()
    assert body == {"status": "ok"}
    for r in _created_redis:
        r.assert_only_called({"sadd", "close"})


def test_readyz_degraded_when_volume_missing() -> None:
    """Test /readyz returns degraded when data volume is missing."""
    _clear_redis_instances()

    # Use hook to fake path not existing
    _test_hooks.path_exists = lambda p: False

    app = create_app(redis_provider=_redis_provider, queue_provider=_queue_provider)
    client = TestClient(app)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    data: dict[str, str | None] = resp.json()
    assert data["status"] == "degraded"
    assert data["reason"] == "data volume not found"
    for r in _created_redis:
        r.assert_only_called({"sadd", "ping", "scard", "close"})


def test_create_job_enqueues_and_returns_id() -> None:
    _clear_redis_instances()
    app = create_app(redis_provider=_redis_provider, queue_provider=_queue_provider)
    client = TestClient(app)
    payload = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "max_sentences": 10,
        "transliterate": True,
        "confidence_threshold": 0.95,
    }
    resp = client.post("/api/v1/jobs", json=payload)
    assert resp.status_code == 200
    jr = parse_job_response_json(resp.text)
    assert jr["status"] == "queued"
    assert type(jr["job_id"]) is str
    assert jr["job_id"]
    for r in _created_redis:
        r.assert_only_called({"sadd", "hset", "expire", "close"})


def test_readyz_ready_and_degraded_paths() -> None:
    """Test /readyz ready when healthy, degraded when Redis fails."""
    _clear_redis_instances()
    app = create_app(redis_provider=_redis_provider, queue_provider=_queue_provider)
    client = TestClient(app)

    # Ready: redis True, volume True
    _test_hooks.path_exists = lambda p: True
    resp = client.get("/readyz")
    assert resp.status_code == 200
    ready_body: dict[str, str | None] = resp.json()
    assert ready_body["status"] == "ready"

    # Assert first part
    for r in _created_redis:
        r.assert_only_called({"sadd", "ping", "scard", "close"})

    # Degraded: redis ping returns False
    _clear_redis_instances()
    degraded_redis: list[FakeRedis] = []

    def _redis_false(settings: Settings) -> RedisCombinedProtocol:
        r = FakeRedisNoPong()
        degraded_redis.append(r)
        return r

    app2 = create_app(redis_provider=_redis_false, queue_provider=_queue_provider)
    with TestClient(app2) as alt:
        r2 = alt.get("/readyz")
        assert r2.status_code == 503
        degraded_body: dict[str, str | None] = r2.json()
        assert degraded_body["status"] == "degraded"
        assert degraded_body["reason"] == "redis no-pong"
    for r in degraded_redis:
        r.assert_only_called({"ping", "close"})
