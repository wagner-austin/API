"""Tests for Redis key helpers."""

from __future__ import annotations

from art_trainer.core.infra.redis_keys import (
    JOB_CANCEL_PREFIX,
    JOB_PROGRESS_PREFIX,
    JOB_RESULT_PREFIX,
    JOB_STATUS_PREFIX,
    cancel_key,
    progress_key,
    result_key,
    status_key,
)


def test_status_key() -> None:
    """Test status_key returns correct key."""
    result = status_key("test-job-123")
    assert result == f"{JOB_STATUS_PREFIX}test-job-123"
    assert result == "art:job:status:test-job-123"


def test_progress_key() -> None:
    """Test progress_key returns correct key."""
    result = progress_key("test-job-456")
    assert result == f"{JOB_PROGRESS_PREFIX}test-job-456"
    assert result == "art:job:progress:test-job-456"


def test_cancel_key() -> None:
    """Test cancel_key returns correct key."""
    result = cancel_key("test-job-789")
    assert result == f"{JOB_CANCEL_PREFIX}test-job-789"
    assert result == "art:job:cancel:test-job-789"


def test_result_key() -> None:
    """Test result_key returns correct key."""
    result = result_key("test-job-000")
    assert result == f"{JOB_RESULT_PREFIX}test-job-000"
    assert result == "art:job:result:test-job-000"
