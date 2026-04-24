"""RQ worker task for PDF extraction.

Processes a queued extraction job: reads the PDF, extracts pages,
writes to Postgres, and updates job progress in Redis.
"""

from __future__ import annotations

from . import _test_hooks
from .db import document_exists, insert_document, insert_pages_batch
from .extract import extract_pdf_pages
from .job_store import mark_completed, mark_failed, save_job, update_progress
from .settings import get_database_url, get_redis_url, get_tenant_email
from .types import ExtractionJob

_PROGRESS_INTERVAL: int = 10


def _resolve_tenant(conn: _test_hooks.DbConnection, email: str) -> None:
    """Set the Postgres tenant context from an email address.

    Args:
        conn: Open database connection.
        email: Tenant email to resolve.

    Raises:
        RuntimeError: If no tenant is found for the email.
    """
    cursor = conn.execute(
        "SELECT tenant_id::text FROM users WHERE email = %s",
        (email,),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"No tenant found for email: {email}")
    tenant_id = str(row[0])
    conn.execute(f"SET app.tenant_id = '{tenant_id}'")


def process_extraction(job_id: str) -> None:
    """Process a single extraction job.

    Reads the PDF file, extracts pages via pdfplumber + docTR,
    writes the document and pages to Postgres, and updates
    progress in Redis throughout.

    Args:
        job_id: The job identifier to process.
    """
    redis_url = get_redis_url()
    redis = _test_hooks.redis_factory(redis_url)

    from .job_store import load_job

    job = load_job(redis, job_id)
    if job is None:
        return

    updated_job = ExtractionJob(
        job_id=job["job_id"],
        status="processing",
        title=job["title"],
        source=job["source"],
        category=job["category"],
        file_path=job["file_path"],
        pages_total=job["pages_total"],
        pages_done=0,
        document_id="",
        error="",
    )
    save_job(redis, updated_job)

    database_url = get_database_url()
    tenant_email = get_tenant_email()
    conn = _test_hooks.connect_db(database_url)

    _resolve_tenant(conn, tenant_email)

    if document_exists(conn, job["title"], job["category"]):
        mark_failed(redis, job_id, f"Document already exists: {job['title']}")
        conn.close()
        return

    pdf_bytes = _test_hooks.read_file(job["file_path"])
    pages = extract_pdf_pages(pdf_bytes)

    update_progress(redis, job_id, 0)

    doc_id = insert_document(
        conn,
        job["title"],
        job["source"],
        len(pages),
        job["category"],
    )

    for i, page in enumerate(pages):
        insert_pages_batch(conn, doc_id, [page])
        if (i + 1) % _PROGRESS_INTERVAL == 0 or i == len(pages) - 1:
            update_progress(redis, job_id, i + 1)

    conn.close()
    mark_completed(redis, job_id, doc_id)


__all__ = [
    "_PROGRESS_INTERVAL",
    "_resolve_tenant",
    "process_extraction",
]
