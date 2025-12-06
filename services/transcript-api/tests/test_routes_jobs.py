"""Tests for STT job routes parsing functions."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.json_utils import JSONValue
from platform_workers.testing import FakeLogger, FakeQueue, FakeRedis

from transcript_api.dependencies import provider_context
from transcript_api.job_store import transcript_job_key
from transcript_api.routes.jobs import STTJobRequest, _parse_stt_job_request, build_router


def test_stt_job_request_class() -> None:
    """Test STTJobRequest class initialization."""
    req = STTJobRequest(url="https://youtu.be/abc", user_id=42)
    assert req.url == "https://youtu.be/abc"
    assert req.user_id == 42


def test_parse_stt_job_request_success() -> None:
    """Test _parse_stt_job_request with valid input."""
    payload: dict[str, JSONValue] = {"url": "https://youtu.be/dQw4w9WgXcQ", "user_id": 5}
    result = _parse_stt_job_request(payload)
    assert result.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert result.user_id == 5


def test_parse_stt_job_request_missing_url() -> None:
    """Test _parse_stt_job_request raises on missing url."""
    from platform_core.errors import AppError

    with pytest.raises(AppError) as excinfo:
        _parse_stt_job_request({"user_id": 1})
    assert "url is required" in str(excinfo.value)


def test_parse_stt_job_request_empty_url() -> None:
    """Test _parse_stt_job_request raises on empty url."""
    from platform_core.errors import AppError

    with pytest.raises(AppError) as excinfo:
        _parse_stt_job_request({"url": "   ", "user_id": 1})
    assert "url is required" in str(excinfo.value)


def test_parse_stt_job_request_non_string_url() -> None:
    """Test _parse_stt_job_request raises on non-string url."""
    from platform_core.errors import AppError

    with pytest.raises(AppError) as excinfo:
        _parse_stt_job_request({"url": 123, "user_id": 1})
    assert "url is required" in str(excinfo.value)


def test_parse_stt_job_request_missing_user_id() -> None:
    """Test _parse_stt_job_request raises on missing user_id."""
    from platform_core.errors import AppError

    with pytest.raises(AppError) as excinfo:
        _parse_stt_job_request({"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert "user_id must be an integer" in str(excinfo.value)


def test_parse_stt_job_request_non_int_user_id() -> None:
    """Test _parse_stt_job_request raises on non-integer user_id."""
    from platform_core.errors import AppError

    with pytest.raises(AppError) as excinfo:
        _parse_stt_job_request({"url": "https://youtu.be/dQw4w9WgXcQ", "user_id": "42"})
    assert "user_id must be an integer" in str(excinfo.value)


def test_build_router_returns_router_with_routes() -> None:
    """Test build_router returns an APIRouter with routes."""
    router = build_router()
    # Verify router has routes by accessing it directly
    routes = router.routes
    assert routes  # Non-empty list is truthy


def _make_test_client() -> tuple[TestClient, FakeRedis, FakeQueue, FakeLogger]:
    """Create a TestClient with fake providers."""
    redis = FakeRedis()
    queue = FakeQueue()
    logger = FakeLogger()

    provider_context.redis_provider = lambda: redis
    provider_context.queue_provider = lambda: queue
    provider_context.logger_provider = lambda: logger

    app = FastAPI()
    install_exception_handlers_fastapi(app, logger_name="test")
    app.include_router(build_router())
    return TestClient(app), redis, queue, logger


def _cleanup_providers() -> None:
    """Reset provider context after test."""
    provider_context.redis_provider = None
    provider_context.queue_provider = None
    provider_context.logger_provider = None


def test_create_stt_job_success() -> None:
    """Test POST /v1/stt/jobs with valid input."""
    client, _redis, queue, logger = _make_test_client()
    try:
        payload: dict[str, str | int] = {"url": "https://youtu.be/dQw4w9WgXcQ", "user_id": 42}
        resp = client.post("/v1/stt/jobs", json=payload)
        assert resp.status_code == 202
        data: dict[str, str | int] = resp.json()
        assert "job_id" in data
        assert data["user_id"] == 42
        assert data["status"] == "queued"
        url_val = data["url"]
        assert isinstance(url_val, str) and "youtube.com" in url_val
        assert len(queue.jobs) == 1
        assert queue.jobs[0].func == "transcript_api.jobs.process_stt"
        assert len(logger.records) == 1
        assert logger.records[0].level == "info"
    finally:
        _cleanup_providers()


def test_create_stt_job_invalid_json() -> None:
    """Test POST /v1/stt/jobs with non-object JSON."""
    client, _, _, _ = _make_test_client()
    try:
        resp = client.post(
            "/v1/stt/jobs",
            content=b'"just a string"',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
    finally:
        _cleanup_providers()


def test_create_stt_job_missing_url() -> None:
    """Test POST /v1/stt/jobs with missing url."""
    client, _, _, _ = _make_test_client()
    try:
        payload: dict[str, int] = {"user_id": 42}
        resp = client.post("/v1/stt/jobs", json=payload)
        assert resp.status_code == 400
    finally:
        _cleanup_providers()


def test_get_stt_job_status_not_found() -> None:
    """Test GET /v1/stt/jobs/{job_id} when job doesn't exist."""
    client, _, _, _ = _make_test_client()
    try:
        resp = client.get("/v1/stt/jobs/nonexistent-job")
        assert resp.status_code == 404
    finally:
        _cleanup_providers()


def test_get_stt_job_status_found() -> None:
    """Test GET /v1/stt/jobs/{job_id} when job exists."""
    client, redis, _, _ = _make_test_client()
    try:
        job_id = "test-job-123"
        now = datetime.utcnow().isoformat()
        redis.hset(
            transcript_job_key(job_id),
            {
                "user_id": "42",
                "status": "processing",
                "progress": "50",
                "message": "downloading",
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "video_id": "dQw4w9WgXcQ",
                "text": "",
                "created_at": now,
                "updated_at": now,
                "error": "",
            },
        )
        resp = client.get(f"/v1/stt/jobs/{job_id}")
        assert resp.status_code == 200
        data: dict[str, str | int | None] = resp.json()
        assert data["job_id"] == job_id
        assert data["user_id"] == 42
        assert data["status"] == "processing"
        assert data["progress"] == 50
        assert "text" not in data  # Not completed, so no text
    finally:
        _cleanup_providers()


def test_get_stt_job_status_completed_includes_text() -> None:
    """Test GET /v1/stt/jobs/{job_id} includes text when completed."""
    client, redis, _, _ = _make_test_client()
    try:
        job_id = "completed-job"
        now = datetime.utcnow().isoformat()
        redis.hset(
            transcript_job_key(job_id),
            {
                "user_id": "1",
                "status": "completed",
                "progress": "100",
                "message": "done",
                "url": "https://www.youtube.com/watch?v=xyz",
                "video_id": "xyz",
                "text": "Hello world transcript",
                "created_at": now,
                "updated_at": now,
                "error": "",
            },
        )
        resp = client.get(f"/v1/stt/jobs/{job_id}")
        assert resp.status_code == 200
        data: dict[str, str | int | None] = resp.json()
        assert data["status"] == "completed"
        assert data["text"] == "Hello world transcript"
    finally:
        _cleanup_providers()
