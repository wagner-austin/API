from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from turkic_api.api.config import Settings
from turkic_api.api.main import RedisCombinedProtocol, create_app
from turkic_api.api.models import parse_job_response_json
from turkic_api.api.types import QueueProtocol, RQJobLike, RQRetryLike, _EnqCallable
from turkic_api.core.models import UnknownJson


class _RedisStub:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def close(self) -> None:
        pass

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._store[key] = {**mapping}
        return 1

    def hgetall(self, key: str) -> dict[str, str]:
        return self._store.get(key, {}).copy()

    def publish(self, channel: str, message: str) -> int:
        return 1

    def set(self, key: str, value: str) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def sadd(self, key: str, *values: str) -> int:
        return len(values)

    def scard(self, key: str) -> int:
        return 1

    def hget(self, key: str, field: str) -> str | None:
        return None

    def sismember(self, key: str, member: str) -> bool:
        return False


class _QueueStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str | _EnqCallable, tuple[UnknownJson, ...]]] = []

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
        self.calls.append((func, args))

        class _Job(RQJobLike):
            def get_id(self) -> str:
                return "test-job-id"

        return _Job()


def _redis_provider(settings: Settings) -> RedisCombinedProtocol:
    return _RedisStub()


def _queue_provider() -> QueueProtocol:
    return _QueueStub()


def test_healthz_always_ok() -> None:
    """Test /healthz liveness endpoint always returns ok."""
    app = create_app(redis_provider=_redis_provider, queue_provider=_queue_provider)
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body: dict[str, str] = resp.json()
    assert body == {"status": "ok"}


def test_readyz_degraded_when_volume_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test /readyz returns degraded when data volume is missing."""
    app = create_app(redis_provider=_redis_provider, queue_provider=_queue_provider)
    client = TestClient(app)
    from pathlib import Path

    def _exists(self: Path) -> bool:
        return False

    monkeypatch.setattr("pathlib.Path.exists", _exists, raising=False)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    data: dict[str, str | None] = resp.json()
    assert data["status"] == "degraded"
    assert data["reason"] == "data volume not found"


def test_create_job_enqueues_and_returns_id() -> None:
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


def test_readyz_ready_and_degraded_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test /readyz ready when healthy, degraded when Redis fails."""
    app = create_app(redis_provider=_redis_provider, queue_provider=_queue_provider)
    client = TestClient(app)
    # Ready: redis True, volume True
    from pathlib import Path

    def _exists_true(self: Path) -> bool:
        return True

    monkeypatch.setattr("pathlib.Path.exists", _exists_true, raising=False)
    resp = client.get("/readyz")
    assert resp.status_code == 200
    ready_body: dict[str, str | None] = resp.json()
    assert ready_body["status"] == "ready"

    # Degraded: redis ping returns False
    class _RedisFalse(_RedisStub):
        def ping(self, **kwargs: str | int | float | bool | None) -> bool:
            return False

    def _redis_false(settings: Settings) -> RedisCombinedProtocol:
        return _RedisFalse()

    app2 = create_app(redis_provider=_redis_false, queue_provider=_queue_provider)
    with TestClient(app2) as alt:
        r2 = alt.get("/readyz")
        assert r2.status_code == 503
        degraded_body: dict[str, str | None] = r2.json()
        assert degraded_body["status"] == "degraded"
        assert degraded_body["reason"] == "redis no-pong"
