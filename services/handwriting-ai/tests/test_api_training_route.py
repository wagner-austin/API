"""Router construction and the training job route."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from platform_core.config import _test_hooks as config_test_hooks
from platform_core.errors import ErrorCode
from platform_core.testing import make_fake_env
from platform_workers.rq_harness import _RedisBytesClient

from handwriting_ai import _test_hooks
from handwriting_ai.api.routes.training import (
    build_router,
)
from handwriting_ai.api.types import RQRetryLike, UnknownJson, _EnqCallable
from handwriting_ai.config import Settings


class _RedisConnectionProto(Protocol):
    """Protocol for Redis connection used by RQ."""

    pass


# --- Tests for dependencies.py ---


def test_build_router_creates_router() -> None:
    """Test build_router returns an APIRouter with the training endpoint."""
    from platform_core.security import create_api_key_dependency

    api_key_dep = create_api_key_dependency(
        required_key="",
        error_code=ErrorCode.UNAUTHORIZED,
        http_status=401,
    )
    router = build_router(api_key_dep)
    # Verify router has routes by checking it's not empty
    assert router.routes


class _FakeRQJobForTrainingTest:
    """Fake RQ job for training route test."""

    def get_id(self) -> str:
        return "test-job-id-123"


class _FakeQueueForTrainingTest:
    """Fake queue for training route test."""

    def __init__(self) -> None:
        self.call_count = 0

    def enqueue(
        self,
        func: str | _EnqCallable,
        *args: UnknownJson,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> _FakeRQJobForTrainingTest:
        self.call_count += 1
        return _FakeRQJobForTrainingTest()


class _FakeRQClientQueueForTest:
    """Fake RQClientQueue that matches the protocol."""

    def __init__(self, call_tracker: dict[str, int]) -> None:
        self._tracker = call_tracker

    def enqueue(
        self,
        func_ref: str,
        *args: UnknownJson,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> _FakeRQJobForTrainingTest:
        self._tracker["count"] += 1
        return _FakeRQJobForTrainingTest()

    def remove(self, job_or_id: str) -> int:
        """Report that nothing was pending; this fake tracks enqueues only.

        Args:
            job_or_id: The job id a caller would remove.

        Returns:
            0, since this double keeps no pending list to remove from.
        """
        _ = job_or_id
        return 0


def test_create_training_job_via_testclient(tmp_path: Path) -> None:
    """Test training job endpoint via TestClient with mocked dependencies."""
    from fastapi.testclient import TestClient
    from platform_core.json_utils import JSONValue, load_json_str
    from platform_workers.rq_harness import RQClientQueue

    from handwriting_ai.api.main import create_app
    from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig

    # Build settings
    app_cfg: AppConfig = {
        "data_root": tmp_path,
        "artifacts_root": tmp_path,
        "logs_root": tmp_path,
        "threads": 1,
        "port": 8081,
    }
    digits_cfg: DigitsConfig = {
        "model_dir": tmp_path / "models",
        "active_model": "test-model",
        "tta": False,
        "uncertain_threshold": 0.5,
        "max_image_mb": 1,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 1,
        "visualize_max_kb": 64,
        "retention_keep_runs": 1,
        "allowed_hosts": frozenset(["*"]),
    }
    security_cfg: SecurityConfig = {"api_key": "", "api_key_enabled": False}
    settings: Settings = {"app": app_cfg, "digits": digits_cfg, "security": security_cfg}

    # Track queue calls
    call_tracker: dict[str, int] = {"count": 0}
    fake_queue: RQClientQueue = _FakeRQClientQueueForTest(call_tracker)

    def _fake_rq_queue_factory(name: str, connection: _RedisBytesClient) -> RQClientQueue:
        _ = (name, connection)  # unused
        return fake_queue

    # Set REDIS_URL via hook
    config_test_hooks.get_env = make_fake_env({"REDIS_URL": "redis://fake:6379/0"})

    # Set rq_queue_factory hook
    _test_hooks.rq_queue_factory = _fake_rq_queue_factory

    # Create app and test client
    app = create_app(settings, enforce_api_key=False)
    client = TestClient(app)

    # Create valid payload
    payload: dict[str, str | int | float | bool | None] = {
        "user_id": 123,
        "model_id": "test-model",
        "epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "seed": 42,
        "augment": True,
        "notes": "Test training job",
    }

    # Make request
    resp = client.post("/api/v1/training/jobs", json=payload)

    # Parse response using established pattern
    body: JSONValue = load_json_str(resp.text)
    if type(body) is not dict:
        raise AssertionError("expected dict")

    # Verify response
    assert resp.status_code == 202
    assert body["status"] == "queued"
    assert body["job_id"] == "test-job-id-123"
    assert body["user_id"] == 123
    assert body["model_id"] == "test-model"

    # Verify queue was called
    assert call_tracker["count"] == 1
