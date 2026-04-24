"""Tests for doc_extract_api.job_store."""

from __future__ import annotations

from platform_workers.testing import FakeRedis

from doc_extract_api.job_store import (
    _job_key,
    load_job,
    mark_completed,
    mark_failed,
    save_job,
    update_progress,
)
from doc_extract_api.types import ExtractionJob


def _make_job(job_id: str = "j1") -> ExtractionJob:
    return ExtractionJob(
        job_id=job_id,
        status="queued",
        title="Test",
        source="",
        category="general",
        file_path="/tmp/test.pdf",
        pages_total=10,
        pages_done=0,
        document_id="",
        error="",
    )


class TestJobKey:
    def test_format(self) -> None:
        assert _job_key("abc") == "doc-extract:job:abc"


class TestSaveAndLoad:
    def test_save_and_load(self) -> None:
        redis = FakeRedis()
        job = _make_job("j1")
        save_job(redis, job)

        loaded = load_job(redis, "j1")
        assert loaded is not None and loaded["job_id"] == "j1"
        assert loaded["status"] == "queued"
        assert loaded["title"] == "Test"
        assert loaded["pages_total"] == 10
        redis.assert_only_called({"hset", "hgetall"})

    def test_load_missing(self) -> None:
        redis = FakeRedis()
        assert load_job(redis, "nonexistent") is None
        redis.assert_only_called({"hgetall"})


class TestUpdateProgress:
    def test_updates_pages_done(self) -> None:
        redis = FakeRedis()
        job = _make_job("j1")
        save_job(redis, job)
        update_progress(redis, "j1", 5)

        loaded = load_job(redis, "j1")
        assert loaded is not None and loaded["pages_done"] == 5
        redis.assert_only_called({"hset", "hgetall"})


class TestMarkCompleted:
    def test_marks_completed(self) -> None:
        redis = FakeRedis()
        job = _make_job("j1")
        save_job(redis, job)
        mark_completed(redis, "j1", "doc-uuid-123")

        loaded = load_job(redis, "j1")
        assert loaded is not None and loaded["status"] == "completed"
        assert loaded["document_id"] == "doc-uuid-123"
        redis.assert_only_called({"hset", "hgetall"})


class TestMarkFailed:
    def test_marks_failed(self) -> None:
        redis = FakeRedis()
        job = _make_job("j1")
        save_job(redis, job)
        mark_failed(redis, "j1", "something went wrong")

        loaded = load_job(redis, "j1")
        assert loaded is not None and loaded["status"] == "failed"
        assert loaded["error"] == "something went wrong"
        redis.assert_only_called({"hset", "hgetall"})
