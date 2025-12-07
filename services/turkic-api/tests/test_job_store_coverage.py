"""Coverage tests for job_store.py missing branches."""

from __future__ import annotations

from datetime import datetime

import pytest
from platform_core.turkic_jobs import turkic_job_key
from platform_workers.testing import FakeRedis

from turkic_api.api.job_store import TurkicJobStore


def test_parse_status_failed_branch() -> None:
    """Cover line 37: return 'failed' in _parse_status."""
    r = FakeRedis()
    key = turkic_job_key("failed_job")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "failed",
            "progress": "100",
            "created_at": now,
            "updated_at": now,
            "error": "something went wrong",
        },
    )
    store = TurkicJobStore(r)
    loaded = store.load("failed_job")
    if loaded is None:
        pytest.fail("expected loaded job")
    assert loaded["status"] == "failed"
    assert loaded["error"] == "something went wrong"
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_parse_user_id_invalid_string() -> None:
    """Cover line 47: invalid user_id in _parse_user_id (non-digit string)."""
    r = FakeRedis()
    key = turkic_job_key("bad_user")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "abc",
            "status": "queued",
            "progress": "0",
            "created_at": now,
            "updated_at": now,
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(ValueError, match="invalid user_id"):
        store.load("bad_user")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_parse_user_id_empty_string() -> None:
    """Cover line 47: invalid user_id when empty string."""
    r = FakeRedis()
    key = turkic_job_key("empty_user")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "",
            "status": "queued",
            "progress": "0",
            "created_at": now,
            "updated_at": now,
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(ValueError, match="invalid user_id"):
        store.load("empty_user")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_parse_status_missing_with_user_id() -> None:
    """Cover line 29: missing status in _parse_status (with user_id present)."""
    r = FakeRedis()
    key = turkic_job_key("no_status")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "progress": "0",
            "created_at": now,
            "updated_at": now,
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(ValueError, match="missing status"):
        store.load("no_status")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_parse_progress_missing_with_user_id() -> None:
    """Cover line 54: missing progress in _parse_progress (with user_id and status)."""
    r = FakeRedis()
    key = turkic_job_key("no_progress")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "queued",
            "created_at": now,
            "updated_at": now,
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(ValueError, match="missing progress"):
        store.load("no_progress")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_parse_progress_invalid_with_user_id() -> None:
    """Cover line 57: invalid progress in _parse_progress (non-digit string)."""
    r = FakeRedis()
    key = turkic_job_key("bad_progress")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "queued",
            "progress": "xyz",
            "created_at": now,
            "updated_at": now,
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(ValueError, match="invalid progress"):
        store.load("bad_progress")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_parse_datetime_missing_with_user_id() -> None:
    """Cover line 64: missing datetime in _parse_datetime (empty created_at)."""
    r = FakeRedis()
    key = turkic_job_key("no_datetime")
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "queued",
            "progress": "0",
            "created_at": "",
            "updated_at": "2024-01-01T00:00:00",
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(ValueError, match="missing created_at"):
        store.load("no_datetime")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_parse_upload_status_invalid_with_user_id() -> None:
    """Cover line 74: invalid upload_status in _parse_upload_status."""
    r = FakeRedis()
    key = turkic_job_key("bad_upload")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "processing",
            "progress": "50",
            "created_at": now,
            "updated_at": now,
            "upload_status": "pending",
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(ValueError, match="invalid upload_status"):
        store.load("bad_upload")
    r.assert_only_called({"hset", "expire", "hgetall"})
