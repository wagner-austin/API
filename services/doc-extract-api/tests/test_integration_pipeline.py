"""End-to-end integration test for the extraction pipeline.

Creates a real PDF with fpdf2, runs it through the full
process_extraction worker against corvis_test and real Redis,
then verifies the document and pages exist in Postgres.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from platform_core.config import _test_hooks as platform_hooks
from platform_workers.redis import RedisStrProto

from doc_extract_api import _test_hooks
from doc_extract_api._test_hooks import DbConnection, _default_connect_db, _default_redis_for_kv
from doc_extract_api.job_store import load_job, save_job
from doc_extract_api.types import ExtractionJob

_TEST_TENANT_ID = "00000000-0000-0000-0000-000000000099"
_TEST_EMAIL = "test-pipeline@example.com"


def _load_test_dsn() -> str:
    """Load DATABASE_TEST_URL from the MCPs repo .env file."""
    get_env = platform_hooks.get_env
    dsn = get_env("DATABASE_TEST_URL")
    if dsn is not None and len(dsn) > 0:
        return dsn

    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "libs").is_dir():
            mcps_env = current.parent / "MCPs" / ".env"
            if mcps_env.exists():
                for line in mcps_env.read_text(encoding="utf-8").splitlines():
                    if line.startswith("DATABASE_TEST_URL="):
                        return line.split("=", 1)[1].strip()
            break
        current = current.parent

    import pytest

    return pytest.skip("DATABASE_TEST_URL not found")


class _FpdfProto(Protocol):
    """Protocol for fpdf2 FPDF class."""

    def add_page(self) -> None: ...
    def set_font(self, family: str, size: int) -> None: ...
    def cell(self, text: str) -> None: ...
    def output(self) -> bytes: ...


def _create_test_pdf() -> bytes:
    """Create a multi-page PDF with known text content.

    Returns:
        Raw PDF bytes with 3 pages of known content.
    """
    _fpdf2 = __import__("fpdf", fromlist=["FPDF"])
    fpdf: _FpdfProto = _fpdf2.FPDF()

    for i in range(3):
        fpdf.add_page()
        fpdf.set_font("Helvetica", size=14)
        fpdf.cell(text=f"Integration test page {i + 1} with unique content alpha-{i + 1}")

    return fpdf.output()


def _ensure_test_user(conn: DbConnection) -> None:
    """Ensure the test user exists in corvis_test for tenant resolution.

    Args:
        conn: Open database connection.
    """
    conn.execute(
        "INSERT INTO users (tenant_id, email) VALUES (%s::uuid, %s) ON CONFLICT (email) DO NOTHING",
        (_TEST_TENANT_ID, _TEST_EMAIL),
    )
    conn.commit()


def _cleanup_test_doc(conn: DbConnection, title: str, category: str) -> None:
    """Remove test documents and pages from corvis_test.

    Args:
        conn: Open database connection.
        title: Document title to clean up.
        category: Document category.
    """
    conn.execute(
        "DELETE FROM document_pages WHERE document_id IN "
        "(SELECT id FROM documents WHERE title = %s AND category = %s)",
        (title, category),
    )
    conn.execute(
        "DELETE FROM documents WHERE title = %s AND category = %s",
        (title, category),
    )
    conn.commit()


class TestEndToEndPipeline:
    """Full pipeline integration test against real infrastructure."""

    def test_extract_real_pdf_to_real_db(self) -> None:
        """Extract a real PDF, write pages to corvis_test, verify results."""
        dsn = _load_test_dsn()

        # Create a real PDF file on disk
        pdf_bytes = _create_test_pdf()
        fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.write(fd, pdf_bytes)
        os.close(fd)

        title = "integration-pipeline-test-doc"
        category = "general"

        # Set up real Redis
        redis = _default_redis_for_kv("redis://127.0.0.1:6379/0")

        # Set up real DB connection and ensure test user exists
        conn = _default_connect_db(dsn)
        conn.execute(f"SET app.tenant_id = '{_TEST_TENANT_ID}'")
        _ensure_test_user(conn)
        _cleanup_test_doc(conn, title, category)
        conn.close()

        # Wire hooks to real implementations
        def _real_redis_factory(url: str) -> RedisStrProto:
            return _default_redis_for_kv(url)

        real_factory: Callable[[str], RedisStrProto] = _real_redis_factory
        _test_hooks.redis_factory = real_factory
        _test_hooks.connect_db = _default_connect_db

        # Set env vars for the worker
        env_vars = {
            "REDIS_URL": "redis://127.0.0.1:6379/0",
            "DATABASE_URL": dsn,
            "DOC_EXTRACT_TENANT_EMAIL": _TEST_EMAIL,
        }
        from platform_core.testing import make_fake_env

        platform_hooks.get_env = make_fake_env(env_vars)

        # Ensure OCR is disabled (pdfplumber-only for this test)
        _test_hooks.ocr_pdf = None

        # Create and save the job
        job = ExtractionJob(
            job_id="integration-pipeline-job",
            status="queued",
            title=title,
            source="integration-test",
            category=category,
            file_path=pdf_path,
            pages_total=0,
            pages_done=0,
            document_id="",
            error="",
        )
        save_job(redis, job)

        # Run the full extraction pipeline
        from doc_extract_api.worker import process_extraction

        process_extraction("integration-pipeline-job")

        # Verify the job completed
        result = load_job(redis, "integration-pipeline-job")
        assert result is not None and result["status"] == "completed"
        assert len(result["document_id"]) == 36  # UUID

        # Verify the document and pages exist in Postgres
        conn = _default_connect_db(dsn)
        conn.execute(f"SET app.tenant_id = '{_TEST_TENANT_ID}'")

        doc_cursor = conn.execute(
            "SELECT title, page_count, category FROM documents WHERE id = %s::uuid",
            (result["document_id"],),
        )
        doc_row = doc_cursor.fetchone()
        assert doc_row is not None and doc_row[0] == title
        assert doc_row[1] == 3  # 3 pages
        assert doc_row[2] == category

        pages_cursor = conn.execute(
            "SELECT page_number, content FROM document_pages "
            "WHERE document_id = %s::uuid ORDER BY page_number",
            (result["document_id"],),
        )
        pages = pages_cursor.fetchall()
        assert len(pages) == 3
        page_1_content = str(pages[0][1])
        assert "page 1" in page_1_content.lower() or "alpha-1" in page_1_content

        # Clean up
        _cleanup_test_doc(conn, title, category)
        conn.close()
        redis.delete("doc-extract:job:integration-pipeline-job")
        redis.close()
        os.unlink(pdf_path)
