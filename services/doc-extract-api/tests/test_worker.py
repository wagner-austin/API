"""Tests for doc_extract_api.worker."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from doc_extract_api import _test_hooks
from doc_extract_api.job_store import load_job, save_job
from doc_extract_api.types import ExtractionJob
from doc_extract_api.worker import _resolve_tenant, process_extraction

from ._test_hooks import (
    FakeDbConnection,
    FakePdfPlumberPage,
    make_fake_connect_db,
    make_fake_pdfplumber_open,
    make_fake_read_file,
)


def _make_redis_factory(redis: FakeRedis) -> Callable[[str], RedisStrProto]:
    """Create a fake redis factory returning the given FakeRedis."""

    def _factory(url: str) -> RedisStrProto:
        _ = url
        return redis

    return _factory


class TestResolveTenant:
    def test_resolves_tenant(self) -> None:
        conn = FakeDbConnection()
        conn.add_response([("tenant-uuid-123",)])
        _resolve_tenant(conn, "user@example.com")
        assert len(conn.executed) == 2
        assert "SELECT tenant_id" in conn.executed[0][0]
        assert "SET app.tenant_id" in conn.executed[1][0]

    def test_raises_on_missing_tenant(self) -> None:
        conn = FakeDbConnection()
        with pytest.raises(RuntimeError, match="No tenant found"):
            _resolve_tenant(conn, "unknown@example.com")


class TestProcessExtraction:
    def setup_method(self) -> None:
        _test_hooks.ocr_pdf = None

    def test_full_pipeline(self) -> None:
        redis = FakeRedis()
        conn = FakeDbConnection()
        conn.add_response([("tenant-uuid",)])  # tenant lookup
        conn.add_response([])  # SET LOCAL (no result needed)
        conn.add_response([(0,)])  # document_exists count
        conn.add_response([("new-doc-uuid",)])  # insert_document RETURNING id

        fake_pages = [
            FakePdfPlumberPage(1, text="page one content"),
            FakePdfPlumberPage(2, text="page two content"),
        ]

        _test_hooks.redis_factory = _make_redis_factory(redis)
        _test_hooks.connect_db = make_fake_connect_db(conn)
        _test_hooks.pdfplumber_open = make_fake_pdfplumber_open(fake_pages)
        _test_hooks.read_file = make_fake_read_file({"/tmp/test.pdf": b"fake pdf"})

        job = ExtractionJob(
            job_id="test-job",
            status="queued",
            title="Test Document",
            source="https://example.com",
            category="general",
            file_path="/tmp/test.pdf",
            pages_total=0,
            pages_done=0,
            document_id="",
            error="",
        )
        save_job(redis, job)

        process_extraction("test-job")

        result = load_job(redis, "test-job")
        assert result is not None and result["status"] == "completed"
        assert result["document_id"] == "new-doc-uuid"
        assert conn.closed is True
        redis.assert_only_called({"hset", "hgetall"})

    def test_missing_job(self) -> None:
        redis = FakeRedis()
        _test_hooks.redis_factory = _make_redis_factory(redis)
        process_extraction("nonexistent-job")
        redis.assert_only_called({"hgetall"})

    def test_duplicate_document(self) -> None:
        redis = FakeRedis()
        conn = FakeDbConnection()
        conn.add_response([("tenant-uuid",)])  # tenant lookup
        conn.add_response([])  # SET LOCAL
        conn.add_response([(1,)])  # document_exists count = 1 (duplicate)

        _test_hooks.redis_factory = _make_redis_factory(redis)
        _test_hooks.connect_db = make_fake_connect_db(conn)

        job = ExtractionJob(
            job_id="dup-job",
            status="queued",
            title="Existing Doc",
            source="",
            category="general",
            file_path="/tmp/test.pdf",
            pages_total=0,
            pages_done=0,
            document_id="",
            error="",
        )
        save_job(redis, job)

        process_extraction("dup-job")

        result = load_job(redis, "dup-job")
        assert result is not None and result["status"] == "failed"
        assert "already exists" in result["error"]
        assert conn.closed is True
        redis.assert_only_called({"hset", "hgetall"})
