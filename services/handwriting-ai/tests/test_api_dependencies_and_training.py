"""Tests for api/dependencies.py and api/routes/training.py coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest
from platform_core.errors import AppError, ErrorCode

from handwriting_ai.api.dependencies import (
    _get_redis_url,
    get_queue,
    get_redis,
    get_request_logger,
    get_settings,
)
from handwriting_ai.api.routes.training import (
    _validate_train_request,
    build_router,
)
from handwriting_ai.api.types import JsonDict, RQRetryLike, UnknownJson, _EnqCallable


class _RedisConnectionProto(Protocol):
    """Protocol for Redis connection used by RQ."""

    pass


# --- Tests for dependencies.py ---


def test_get_settings_returns_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_settings() returns loaded settings."""
    from pathlib import Path

    from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings

    fake_app: AppConfig = {
        "data_root": Path("/tmp/data"),
        "artifacts_root": Path("/tmp/artifacts"),
        "logs_root": Path("/tmp/logs"),
        "threads": 2,
        "port": 8080,
    }
    fake_digits: DigitsConfig = {
        "model_dir": Path("/tmp/models"),
        "active_model": "test-model",
        "tta": False,
        "uncertain_threshold": 0.5,
        "max_image_mb": 10,
        "max_image_side_px": 2048,
        "predict_timeout_seconds": 30,
        "visualize_max_kb": 128,
        "retention_keep_runs": 5,
        "allowed_hosts": frozenset(["*"]),
    }
    fake_security: SecurityConfig = {"api_key": "", "api_key_enabled": False}
    fake_settings: Settings = {"app": fake_app, "digits": fake_digits, "security": fake_security}

    def _fake_load_settings(*, create_dirs: bool = True) -> Settings:
        return fake_settings

    monkeypatch.setattr(
        "handwriting_ai.api.dependencies.load_settings", _fake_load_settings, raising=True
    )

    settings = get_settings()
    assert settings["app"]["threads"] == 2
    assert settings["app"]["port"] == 8080


def test_get_redis_url_returns_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _get_redis_url() returns REDIS_URL from environment."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    url = _get_redis_url()
    assert url == "redis://localhost:6379/0"


def test_get_redis_url_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _get_redis_url() raises when REDIS_URL not set."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(RuntimeError, match="REDIS_URL"):
        _get_redis_url()


def test_get_redis_yields_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_redis() yields Redis client and closes on teardown."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    closed = {"value": False}

    class _FakeRedis:
        def close(self) -> None:
            closed["value"] = True

        def ping(self) -> bool:
            return True

    def _fake_redis_for_kv(url: str) -> _FakeRedis:
        return _FakeRedis()

    monkeypatch.setattr(
        "handwriting_ai.api.dependencies.redis_for_kv", _fake_redis_for_kv, raising=True
    )

    gen = get_redis()
    client = next(gen)
    if client is None:
        raise AssertionError("expected redis client")
    assert not closed["value"]

    # Exhaust the generator to trigger finally block using gen.close()
    gen.close()

    assert closed["value"]


def test_get_request_logger_returns_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_request_logger() returns a logger instance."""

    class _FakeLoggerInstance:
        def info(self, msg: str, **kwargs: UnknownJson) -> None:
            pass

        def error(self, msg: str, **kwargs: UnknownJson) -> None:
            pass

    def _fake_get_logger(name: str) -> _FakeLoggerInstance:
        return _FakeLoggerInstance()

    monkeypatch.setattr(
        "handwriting_ai.api.dependencies.get_logger", _fake_get_logger, raising=True
    )

    logger = get_request_logger()
    if logger is None:
        raise AssertionError("expected logger")


def test_get_queue_returns_queue_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_queue() returns a QueueProtocol implementation."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    enqueued: list[dict[str, UnknownJson]] = []

    class _FakeRQJob:
        def get_id(self) -> str:
            return "fake-job-id"

    class _FakeRQQueue:
        def enqueue(
            self,
            func: str,
            *args: UnknownJson,
            job_timeout: int | None = None,
            result_ttl: int | None = None,
            failure_ttl: int | None = None,
            retry: RQRetryLike | None = None,
            description: str | None = None,
        ) -> _FakeRQJob:
            enqueued.append({"func": func, "args": list(args)})
            return _FakeRQJob()

    def _fake_redis_raw_for_rq(url: str) -> _RedisConnectionProto:
        class _FakeConn:
            pass

        return _FakeConn()

    def _fake_rq_queue(name: str, connection: _RedisConnectionProto) -> _FakeRQQueue:
        return _FakeRQQueue()

    monkeypatch.setattr(
        "handwriting_ai.api.dependencies.redis_raw_for_rq",
        _fake_redis_raw_for_rq,
        raising=True,
    )
    monkeypatch.setattr("handwriting_ai.api.dependencies.rq_queue", _fake_rq_queue, raising=True)

    queue = get_queue()
    job = queue.enqueue("some.func", {"key": "value"}, job_timeout=60)
    assert job.get_id() == "fake-job-id"
    assert len(enqueued) == 1
    assert enqueued[0]["func"] == "some.func"


# --- Tests for training.py _validate_train_request ---


def test_validate_train_request_valid_payload() -> None:
    """Test _validate_train_request with valid payload."""
    payload: JsonDict = {
        "user_id": 123,
        "model_id": "test-model",
        "epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "seed": 42,
        "augment": True,
        "notes": "Test notes",
    }
    result = _validate_train_request(payload)
    assert result["user_id"] == 123
    assert result["model_id"] == "test-model"
    assert result["epochs"] == 10
    assert result["batch_size"] == 32
    assert result["lr"] == 0.001
    assert result["seed"] == 42
    assert result["augment"] is True
    assert result["notes"] == "Test notes"


def test_validate_train_request_notes_null() -> None:
    """Test _validate_train_request with null notes."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
        "augment": False,
        "notes": None,
    }
    result = _validate_train_request(payload)
    assert result["notes"] is None


def test_validate_train_request_notes_missing() -> None:
    """Test _validate_train_request with missing notes (defaults to None)."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    result = _validate_train_request(payload)
    assert result["notes"] is None


def test_validate_train_request_augment_default() -> None:
    """Test _validate_train_request with missing augment (defaults to False)."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    result = _validate_train_request(payload)
    assert result["augment"] is False


def test_validate_train_request_lr_as_int() -> None:
    """Test _validate_train_request accepts lr as int."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 1,
        "seed": 1,
    }
    result = _validate_train_request(payload)
    assert result["lr"] == 1.0
    assert type(result["lr"]) is float


def test_validate_train_request_invalid_user_id_type() -> None:
    """Test _validate_train_request rejects non-int user_id."""
    payload: JsonDict = {
        "user_id": "not-an-int",
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="user_id must be an integer"):
        _validate_train_request(payload)


def test_validate_train_request_user_id_bool_rejected() -> None:
    """Test _validate_train_request rejects bool user_id."""
    payload: JsonDict = {
        "user_id": True,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="user_id must be an integer"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_model_id() -> None:
    """Test _validate_train_request rejects empty model_id."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="model_id must be a non-empty string"):
        _validate_train_request(payload)


def test_validate_train_request_model_id_whitespace() -> None:
    """Test _validate_train_request rejects whitespace-only model_id."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "   ",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="model_id must be a non-empty string"):
        _validate_train_request(payload)


def test_validate_train_request_model_id_not_string() -> None:
    """Test _validate_train_request rejects non-string model_id."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": 123,
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="model_id must be a non-empty string"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_epochs() -> None:
    """Test _validate_train_request rejects non-positive epochs."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 0,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="epochs must be a positive integer"):
        _validate_train_request(payload)


def test_validate_train_request_epochs_bool_rejected() -> None:
    """Test _validate_train_request rejects bool epochs."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": True,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="epochs must be a positive integer"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_batch_size() -> None:
    """Test _validate_train_request rejects non-positive batch_size."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 0,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="batch_size must be a positive integer"):
        _validate_train_request(payload)


def test_validate_train_request_batch_size_bool_rejected() -> None:
    """Test _validate_train_request rejects bool batch_size."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": True,
        "lr": 0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="batch_size must be a positive integer"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_lr_zero() -> None:
    """Test _validate_train_request rejects zero lr."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0,
        "seed": 1,
    }
    with pytest.raises(AppError, match="lr must be a positive number"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_lr_negative() -> None:
    """Test _validate_train_request rejects negative lr."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": -0.1,
        "seed": 1,
    }
    with pytest.raises(AppError, match="lr must be a positive number"):
        _validate_train_request(payload)


def test_validate_train_request_lr_bool_rejected() -> None:
    """Test _validate_train_request rejects bool lr."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": True,
        "seed": 1,
    }
    with pytest.raises(AppError, match="lr must be a positive number"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_seed() -> None:
    """Test _validate_train_request rejects non-int seed."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": "not-an-int",
    }
    with pytest.raises(AppError, match="seed must be an integer"):
        _validate_train_request(payload)


def test_validate_train_request_seed_bool_rejected() -> None:
    """Test _validate_train_request rejects bool seed."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": False,
    }
    with pytest.raises(AppError, match="seed must be an integer"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_augment() -> None:
    """Test _validate_train_request rejects non-bool augment."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
        "augment": "yes",
    }
    with pytest.raises(AppError, match="augment must be a boolean"):
        _validate_train_request(payload)


def test_validate_train_request_invalid_notes_type() -> None:
    """Test _validate_train_request rejects non-string notes."""
    payload: JsonDict = {
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.1,
        "seed": 1,
        "notes": 123,
    }
    with pytest.raises(AppError, match="notes must be a string or null"):
        _validate_train_request(payload)


# --- Tests for training.py build_router ---


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


def test_create_training_job_via_testclient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test training job endpoint via TestClient with mocked dependencies."""
    from fastapi.testclient import TestClient
    from platform_core.json_utils import JSONValue, load_json_str
    from platform_workers.rq_harness import RQClientQueue, _RedisBytesClient

    from handwriting_ai.api.app import create_app
    from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings

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

    def _fake_rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
        return fake_queue

    # Set REDIS_URL to something (required by get_queue)
    monkeypatch.setenv("REDIS_URL", "redis://fake:6379/0")

    # Monkeypatch rq_queue at the dependencies module level
    monkeypatch.setattr("handwriting_ai.api.dependencies.rq_queue", _fake_rq_queue, raising=True)

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
