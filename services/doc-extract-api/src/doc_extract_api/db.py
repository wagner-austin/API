"""Database operations for document ingestion.

Writes extracted pages to the corvis Postgres documents/document_pages
tables. These tables are managed by corvis-db migrations.
"""

from __future__ import annotations

from . import _test_hooks
from .types import ExtractedPage


def document_exists(conn: _test_hooks.DbConnection, title: str, category: str) -> bool:
    """Check if a document with the given title and category already exists.

    Args:
        conn: Open database connection.
        title: Document title to check.
        category: Document category.

    Returns:
        True if a document with this title exists in the given category.
    """
    cursor = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE title = %s AND category = %s",
        (title, category),
    )
    row = cursor.fetchone()
    return row is not None and int(row[0] or 0) > 0


def insert_document(
    conn: _test_hooks.DbConnection,
    title: str,
    source: str,
    page_count: int,
    category: str,
) -> str:
    """Insert a document record.

    Args:
        conn: Open database connection with tenant context set.
        title: Document title.
        source: Source identifier (URL, file path, etc.).
        page_count: Number of pages in the document.
        category: Document category.

    Returns:
        The UUID of the inserted document.
    """
    cursor = conn.execute(
        "INSERT INTO documents (tenant_id, title, source, page_count, format, category) "
        "VALUES (current_setting('app.tenant_id')::uuid, %s, %s, %s, %s, %s) "
        "RETURNING id::text",
        (title, source, page_count, "pdf", category),
    )
    rows = cursor.fetchall()
    doc_id = str(rows[0][0])
    conn.commit()
    return doc_id


def insert_pages_batch(
    conn: _test_hooks.DbConnection,
    document_id: str,
    pages: list[ExtractedPage],
) -> int:
    """Insert multiple document pages in a single transaction.

    Args:
        conn: Open database connection.
        document_id: UUID of the parent document.
        pages: List of ExtractedPage objects to insert.

    Returns:
        Number of pages inserted.
    """
    for page in pages:
        conn.execute(
            "INSERT INTO document_pages (document_id, page_number, content) "
            "VALUES (%s::uuid, %s, %s)",
            (document_id, page["page_number"], page["content"]),
        )
    conn.commit()
    return len(pages)


__all__ = [
    "document_exists",
    "insert_document",
    "insert_pages_batch",
]
