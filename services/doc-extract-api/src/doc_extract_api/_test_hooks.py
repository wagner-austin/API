"""Internal test hooks for doc-extract-api — dependency injection.

Production code sets hooks to real implementations at startup.
Tests set hooks to fakes before running.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from platform_workers.redis import RedisStrProto, redis_for_kv
from platform_workers.rq_harness import WorkerConfig

# =========================================================================
# PdfPlumber Protocols
# =========================================================================


class PdfPlumberPage(Protocol):
    """Protocol for a single pdfplumber page."""

    @property
    def page_number(self) -> int:
        """1-based page number."""
        ...

    def extract_text(self, *, x_tolerance: int = 3) -> str | None:
        """Extract text from the page.

        Args:
            x_tolerance: Horizontal tolerance for grouping characters.

        Returns:
            Extracted text, or None if no text found.
        """
        ...

    def extract_tables(
        self,
        table_settings: dict[str, int] | None = None,
    ) -> list[list[list[str | None]]]:
        """Extract tables from the page.

        Args:
            table_settings: Optional settings for table detection.

        Returns:
            List of tables, each a list of rows, each a list of cell values.
        """
        ...


class PdfPlumberPdf(Protocol):
    """Protocol for an opened pdfplumber PDF."""

    @property
    def pages(self) -> list[PdfPlumberPage]:
        """All pages in the PDF."""
        ...

    def close(self) -> None:
        """Close the PDF."""
        ...


class PdfPlumberOpenProtocol(Protocol):
    """Protocol for pdfplumber.open()."""

    def __call__(self, path_or_fp: str | Path | io.BytesIO) -> PdfPlumberPdf:
        """Open a PDF from a file path or file-like object.

        Args:
            path_or_fp: A path string or BytesIO containing PDF data.

        Returns:
            An opened PDF object.
        """
        ...


# =========================================================================
# OCR Protocol
# =========================================================================


class OcrPdfProtocol(Protocol):
    """Protocol for extracting text from PDFs via OCR.

    The real implementation uses docTR with GPU acceleration
    to detect and recognize text from specific PDF page images.
    """

    def __call__(self, pdf_bytes: bytes, pages: list[int]) -> dict[int, str]:
        """Extract text from specific PDF pages using OCR.

        Args:
            pdf_bytes: Raw PDF file bytes.
            pages: 0-based page indices to OCR.

        Returns:
            Dict mapping 0-based page index to OCR text for that page.
        """
        ...


# =========================================================================
# Database Protocol
# =========================================================================


class DbCursor(Protocol):
    """Protocol for a database cursor."""

    @property
    def rowcount(self) -> int:
        """Number of rows affected by the last operation."""
        ...

    def fetchall(self) -> list[tuple[str | int | float | bytes | None, ...]]:
        """Fetch all remaining rows.

        Returns:
            List of row tuples.
        """
        ...

    def fetchone(self) -> tuple[str | int | float | bytes | None, ...] | None:
        """Fetch the next row.

        Returns:
            A row tuple, or None if no more rows.
        """
        ...


class DbConnection(Protocol):
    """Protocol for a Postgres database connection."""

    def execute(
        self,
        query: str,
        params: tuple[str | int | float | bytes | None, ...] | None = None,
    ) -> DbCursor:
        """Execute a SQL statement.

        Args:
            query: SQL statement to execute.
            params: Parameters to bind.

        Returns:
            A cursor with results.
        """
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


class ConnectDbProtocol(Protocol):
    """Protocol for opening a database connection."""

    def __call__(self, conninfo: str) -> DbConnection:
        """Open a database connection.

        Args:
            conninfo: Postgres connection string.

        Returns:
            An open database connection.
        """
        ...


# =========================================================================
# Worker runner
# =========================================================================


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


# =========================================================================
# File system
# =========================================================================


class ReadFileProtocol(Protocol):
    """Protocol for reading a file as bytes."""

    def __call__(self, path: str) -> bytes:
        """Read a file and return its contents.

        Args:
            path: Absolute path to the file.

        Returns:
            Raw file bytes.
        """
        ...


# =========================================================================
# Default implementations
# =========================================================================


class _AutocommitConnectProtocol(Protocol):
    """Protocol for psycopg.connect() with autocommit kwarg."""

    def __call__(self, conninfo: str, *, autocommit: bool = False) -> DbConnection:
        """Open a database connection with autocommit option.

        Args:
            conninfo: Postgres connection string.
            autocommit: Whether to enable autocommit mode.

        Returns:
            An open database connection.
        """
        ...


def _default_connect_db(conninfo: str) -> DbConnection:
    """Production implementation — opens a real Postgres connection.

    Args:
        conninfo: Postgres connection string.

    Returns:
        An open psycopg connection with autocommit enabled.
    """
    _psycopg = __import__("psycopg")
    _connect: _AutocommitConnectProtocol = _psycopg.connect
    conn: DbConnection = _connect(conninfo, autocommit=True)
    return conn


def _default_pdfplumber_open(path_or_fp: str | Path | io.BytesIO) -> PdfPlumberPdf:
    """Production implementation — opens a PDF with pdfplumber.

    Args:
        path_or_fp: A path string or BytesIO containing PDF data.

    Returns:
        An opened pdfplumber PDF object.
    """
    _pdfplumber = __import__("pdfplumber")
    _open: PdfPlumberOpenProtocol = _pdfplumber.open
    return _open(path_or_fp)


def _default_redis_for_kv(url: str) -> RedisStrProto:
    """Production implementation — creates real Redis client.

    Args:
        url: Redis connection URL.

    Returns:
        A Redis client for key-value operations.
    """
    return redis_for_kv(url)


def _default_read_file(path: str) -> bytes:
    """Production implementation — reads a file from disk.

    Args:
        path: Absolute path to the file.

    Returns:
        Raw file bytes.
    """
    with open(path, "rb") as f:
        return f.read()


# =========================================================================
# Module-level hooks
# =========================================================================

connect_db: ConnectDbProtocol = _default_connect_db
pdfplumber_open: PdfPlumberOpenProtocol = _default_pdfplumber_open
redis_factory: Callable[[str], RedisStrProto] = _default_redis_for_kv
read_file: ReadFileProtocol = _default_read_file
ocr_pdf: OcrPdfProtocol | None = None
test_runner: WorkerRunnerProtocol | None = None


__all__ = [
    "ConnectDbProtocol",
    "DbConnection",
    "DbCursor",
    "OcrPdfProtocol",
    "PdfPlumberOpenProtocol",
    "PdfPlumberPage",
    "PdfPlumberPdf",
    "ReadFileProtocol",
    "WorkerRunnerProtocol",
    "_AutocommitConnectProtocol",
    "_default_connect_db",
    "_default_pdfplumber_open",
    "_default_read_file",
    "_default_redis_for_kv",
    "connect_db",
    "ocr_pdf",
    "pdfplumber_open",
    "read_file",
    "redis_factory",
    "test_runner",
]
