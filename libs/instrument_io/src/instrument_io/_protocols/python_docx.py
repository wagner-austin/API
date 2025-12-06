"""Protocol definitions for python-docx library.

Provides type-safe interfaces to python-docx Document, Paragraph, Table, and related classes
without importing python-docx directly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class RunProtocol(Protocol):
    """Protocol for docx Run (text run within a paragraph)."""

    @property
    def text(self) -> str:
        """Return text content of the run."""
        ...


class StyleProtocol(Protocol):
    """Protocol for docx paragraph style."""

    @property
    def name(self) -> str:
        """Return style name."""
        ...


class ParagraphProtocol(Protocol):
    """Protocol for docx Paragraph."""

    @property
    def text(self) -> str:
        """Return paragraph text content."""
        ...

    @property
    def runs(self) -> list[RunProtocol]:
        """Return list of runs in the paragraph."""
        ...

    @property
    def style(self) -> StyleProtocol:
        """Return paragraph style."""
        ...


class CellProtocol(Protocol):
    """Protocol for docx table Cell."""

    @property
    def text(self) -> str:
        """Return cell text content."""
        ...

    @property
    def paragraphs(self) -> list[ParagraphProtocol]:
        """Return list of paragraphs in the cell."""
        ...


class RowProtocol(Protocol):
    """Protocol for docx table Row."""

    @property
    def cells(self) -> list[CellProtocol]:
        """Return list of cells in the row."""
        ...


class TableProtocol(Protocol):
    """Protocol for docx Table."""

    @property
    def rows(self) -> list[RowProtocol]:
        """Return list of rows in the table."""
        ...


class DocumentProtocol(Protocol):
    """Protocol for docx Document."""

    @property
    def paragraphs(self) -> list[ParagraphProtocol]:
        """Return list of paragraphs in the document."""
        ...

    @property
    def tables(self) -> list[TableProtocol]:
        """Return list of tables in the document."""
        ...


def _open_docx(path: Path) -> DocumentProtocol:
    """Open Word document with proper typing via Protocol.

    Args:
        path: Path to .docx file.

    Returns:
        DocumentProtocol for the opened document.
    """
    docx_mod = __import__("docx")
    document_func: Callable[[str | Path], DocumentProtocol] = docx_mod.Document
    doc: DocumentProtocol = document_func(path)
    return doc


__all__ = [
    "CellProtocol",
    "DocumentProtocol",
    "ParagraphProtocol",
    "RowProtocol",
    "RunProtocol",
    "StyleProtocol",
    "TableProtocol",
    "_open_docx",
]
