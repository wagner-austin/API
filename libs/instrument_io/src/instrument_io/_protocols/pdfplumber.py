"""Protocol definitions for pdfplumber library.

Provides type-safe interfaces to pdfplumber PDF and Page classes
without importing pdfplumber directly.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Protocol


class PageProtocol(Protocol):
    """Protocol for pdfplumber Page."""

    @property
    def page_number(self) -> int:
        """Return 1-based page number."""
        ...

    @property
    def width(self) -> float:
        """Return page width in points."""
        ...

    @property
    def height(self) -> float:
        """Return page height in points."""
        ...

    def extract_text(self) -> str:
        """Extract text from page.

        Returns:
            Text content of the page.
        """
        ...

    def extract_tables(self) -> list[list[list[str | None]]]:
        """Extract tables from page.

        Returns:
            List of tables, where each table is a list of rows,
            and each row is a list of cell values (strings or None).
        """
        ...


class PDFProtocol(Protocol):
    """Protocol for pdfplumber PDF."""

    @property
    def pages(self) -> list[PageProtocol]:
        """Return list of pages in the PDF."""
        ...

    @property
    def metadata(self) -> dict[str, str | int | float | bool | None]:
        """Return PDF metadata dictionary."""
        ...

    def close(self) -> None:
        """Close the PDF file."""
        ...

    def __enter__(self) -> PDFProtocol:
        """Enter context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, closing the PDF file."""
        ...


class _OpenPDFFn(Protocol):
    """Protocol for pdfplumber.open function."""

    def __call__(self, path: str | Path) -> PDFProtocol: ...


def _open_pdf(path: Path) -> PDFProtocol:
    """Open PDF file with proper typing via Protocol.

    Args:
        path: Path to PDF file.

    Returns:
        PDFProtocol for the opened PDF.
    """
    pdfplumber_mod = __import__("pdfplumber")
    open_fn: _OpenPDFFn = pdfplumber_mod.open
    return open_fn(path)


__all__ = [
    "PDFProtocol",
    "PageProtocol",
    "_open_pdf",
]
