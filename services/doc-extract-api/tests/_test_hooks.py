"""Test fakes for doc-extract-api tests."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

from platform_workers.redis import RedisStrProto

from doc_extract_api import _test_hooks
from doc_extract_api._test_hooks import (
    DbConnection,
    PdfPlumberOpenProtocol,
    PdfPlumberPage,
    PdfPlumberPdf,
)

# =========================================================================
# Fake PdfPlumber
# =========================================================================


class FakePdfPlumberPage:
    """Fake pdfplumber page for testing."""

    def __init__(
        self,
        page_number: int,
        text: str | None = None,
        tables: list[list[list[str | None]]] | None = None,
    ) -> None:
        self._page_number = page_number
        self._text = text
        self._tables = tables if tables is not None else []

    @property
    def page_number(self) -> int:
        return self._page_number

    def extract_text(self, *, x_tolerance: int = 3) -> str | None:
        _ = x_tolerance
        return self._text

    def extract_tables(
        self,
        table_settings: dict[str, int] | None = None,
    ) -> list[list[list[str | None]]]:
        _ = table_settings
        return self._tables


class FakePdfPlumberPdf:
    """Fake pdfplumber PDF for testing."""

    def __init__(self, fake_pages: list[FakePdfPlumberPage]) -> None:
        self._pages = fake_pages
        self.closed = False

    @property
    def pages(self) -> list[PdfPlumberPage]:
        result: list[PdfPlumberPage] = []
        for p in self._pages:
            page: PdfPlumberPage = p
            result.append(page)
        return result

    def close(self) -> None:
        self.closed = True


def make_fake_pdfplumber_open(
    fake_pages: list[FakePdfPlumberPage],
) -> PdfPlumberOpenProtocol:
    """Create a fake pdfplumber.open that returns predefined pages.

    Args:
        fake_pages: List of fake pages to return.

    Returns:
        A callable matching PdfPlumberOpenProtocol.
    """
    pdf: PdfPlumberPdf = FakePdfPlumberPdf(fake_pages)

    def _open(path_or_fp: str | Path | io.BytesIO) -> PdfPlumberPdf:
        _ = path_or_fp
        return pdf

    opener: PdfPlumberOpenProtocol = _open
    return opener


# =========================================================================
# Fake OCR
# =========================================================================


def make_fake_ocr(results: dict[int, str]) -> _test_hooks.OcrPdfProtocol:
    """Create a fake OCR function that returns predefined results.

    Args:
        results: Dict mapping page index to OCR text.

    Returns:
        A callable matching OcrPdfProtocol.
    """

    def _ocr(pdf_bytes: bytes, pages: list[int]) -> dict[int, str]:
        _ = pdf_bytes
        _ = pages
        return results

    ocr: _test_hooks.OcrPdfProtocol = _ocr
    return ocr


# =========================================================================
# Fake Database
# =========================================================================


class FakeDbCursor:
    """Fake database cursor for testing."""

    def __init__(
        self,
        rows: list[tuple[str | int | float | bytes | None, ...]] | None = None,
    ) -> None:
        self._rows = rows if rows is not None else []
        self._index = 0
        self.rowcount = len(self._rows)

    def fetchall(self) -> list[tuple[str | int | float | bytes | None, ...]]:
        return self._rows

    def fetchone(self) -> tuple[str | int | float | bytes | None, ...] | None:
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row


class FakeDbConnection:
    """Fake database connection for testing."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[str | int | float | bytes | None, ...] | None]] = []
        self.committed = 0
        self.closed = False
        self._responses: list[FakeDbCursor] = []

    def add_response(
        self,
        rows: list[tuple[str | int | float | bytes | None, ...]],
    ) -> None:
        """Queue a cursor response for the next execute call.

        Args:
            rows: Rows to return from fetchall/fetchone.
        """
        self._responses.append(FakeDbCursor(rows))

    def execute(
        self,
        query: str,
        params: tuple[str | int | float | bytes | None, ...] | None = None,
    ) -> FakeDbCursor:
        self.executed.append((query, params))
        if len(self._responses) > 0:
            return self._responses.pop(0)
        return FakeDbCursor()

    def commit(self) -> None:
        self.committed += 1

    def close(self) -> None:
        self.closed = True


def make_fake_connect_db(conn: FakeDbConnection) -> _test_hooks.ConnectDbProtocol:
    """Create a fake connect_db that returns a predefined connection.

    Args:
        conn: The fake connection to return.

    Returns:
        A callable matching ConnectDbProtocol.
    """

    def _connect(conninfo: str) -> DbConnection:
        _ = conninfo
        result: DbConnection = conn
        return result

    connect: _test_hooks.ConnectDbProtocol = _connect
    return connect


# =========================================================================
# Fake Redis
# =========================================================================


class FakeRedis:
    """Fake Redis client matching RedisStrProto."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, str]] = {}
        self._strings: dict[str, str] = {}
        self._sets: dict[str, set[str]] = {}

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        _ = kwargs
        return True

    def set(self, key: str, value: str) -> bool | str | None:
        self._strings[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self._strings.get(key)

    def delete(self, key: str) -> int:
        removed = 0
        if key in self._data:
            del self._data[key]
            removed += 1
        if key in self._strings:
            del self._strings[key]
            removed += 1
        return min(removed, 1)

    def expire(self, key: str, time: int) -> bool:
        _ = key
        _ = time
        return True

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        if key not in self._data:
            self._data[key] = {}
        self._data[key].update(mapping)
        return len(mapping)

    def hget(self, key: str, field: str) -> str | None:
        return self._data.get(key, {}).get(field)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._data.get(key, {}))

    def publish(self, channel: str, message: str) -> int:
        _ = channel
        _ = message
        return 0

    def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    def sadd(self, key: str, member: str) -> int:
        if key not in self._sets:
            self._sets[key] = set()
        before = len(self._sets[key])
        self._sets[key].add(member)
        return len(self._sets[key]) - before

    def sismember(self, key: str, member: str) -> bool:
        return member in self._sets.get(key, set())

    def close(self) -> None:
        pass


def make_fake_redis_factory(
    redis: FakeRedis,
) -> Callable[[str], RedisStrProto]:
    """Create a fake Redis factory.

    Args:
        redis: The fake Redis instance to return.

    Returns:
        A callable that returns the fake Redis.
    """

    def _factory(url: str) -> RedisStrProto:
        _ = url
        result: RedisStrProto = redis
        return result

    return _factory


# =========================================================================
# Fake file reader
# =========================================================================


def make_fake_read_file(files: dict[str, bytes]) -> _test_hooks.ReadFileProtocol:
    """Create a fake file reader.

    Args:
        files: Dict mapping file paths to file contents.

    Returns:
        A callable matching ReadFileProtocol.
    """

    def _read(path: str) -> bytes:
        if path not in files:
            raise FileNotFoundError(f"Fake file not found: {path}")
        return files[path]

    reader: _test_hooks.ReadFileProtocol = _read
    return reader


__all__ = [
    "FakeDbConnection",
    "FakeDbCursor",
    "FakePdfPlumberPage",
    "FakePdfPlumberPdf",
    "FakeRedis",
    "make_fake_connect_db",
    "make_fake_ocr",
    "make_fake_pdfplumber_open",
    "make_fake_read_file",
    "make_fake_redis_factory",
]
