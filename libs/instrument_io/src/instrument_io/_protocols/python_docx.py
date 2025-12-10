"""Protocol definitions for python-docx library.

Provides type-safe interfaces to python-docx Document, Paragraph, Table, and related classes
without importing python-docx directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class LengthProtocol(Protocol):
    """Protocol for docx.shared length types (Inches, Pt, Emu).

    All length types have an emu property representing the value
    in English Metric Units (914400 EMUs = 1 inch).
    """

    @property
    def emu(self) -> int:
        """Return value in EMUs (English Metric Units)."""
        ...


class InlineShapeProtocol(Protocol):
    """Protocol for docx InlineShape (images)."""

    @property
    def width(self) -> int:
        """Return width in EMUs."""
        ...

    @width.setter
    def width(self, value: int) -> None:
        """Set width in EMUs."""
        ...

    @property
    def height(self) -> int:
        """Return height in EMUs."""
        ...

    @height.setter
    def height(self, value: int) -> None:
        """Set height in EMUs."""
        ...


class WdAlignParagraphProtocol(Protocol):
    """Protocol for WD_ALIGN_PARAGRAPH enum values."""

    CENTER: int
    LEFT: int
    RIGHT: int
    JUSTIFY: int


class RunProtocol(Protocol):
    """Protocol for docx Run (text run within a paragraph)."""

    @property
    def text(self) -> str:
        """Return text content of the run."""
        ...

    @property
    def bold(self) -> bool | None:
        """Return whether run is bold."""
        ...

    @bold.setter
    def bold(self, value: bool) -> None:
        """Set bold property."""
        ...

    @property
    def italic(self) -> bool | None:
        """Return whether run is italic."""
        ...

    @italic.setter
    def italic(self, value: bool) -> None:
        """Set italic property."""
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

    @property
    def alignment(self) -> int | None:
        """Return paragraph alignment."""
        ...

    @alignment.setter
    def alignment(self, value: int) -> None:
        """Set paragraph alignment."""
        ...

    def add_run(self, text: str | None = None) -> RunProtocol:
        """Add a run to the paragraph."""
        ...


class CellProtocol(Protocol):
    """Protocol for docx table Cell."""

    @property
    def text(self) -> str:
        """Return cell text content."""
        ...

    @text.setter
    def text(self, value: str) -> None:
        """Set cell text content."""
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

    @property
    def style(self) -> str | None:
        """Return table style name."""
        ...

    @style.setter
    def style(self, value: str) -> None:
        """Set table style."""
        ...

    def add_row(self) -> RowProtocol:
        """Add a row to the table."""
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

    def add_heading(self, text: str, level: int = 1) -> ParagraphProtocol:
        """Add a heading to the document."""
        ...

    def add_paragraph(self, text: str = "", style: str | None = None) -> ParagraphProtocol:
        """Add a paragraph to the document."""
        ...

    def add_table(self, rows: int, cols: int) -> TableProtocol:
        """Add a table to the document."""
        ...

    def add_picture(
        self,
        image_path_or_stream: str | Path,
        width: LengthProtocol | None = None,
        height: LengthProtocol | None = None,
    ) -> InlineShapeProtocol:
        """Add an image to the document.

        Args:
            image_path_or_stream: Path to image file.
            width: Optional width (maintains aspect ratio if height not set).
            height: Optional height.

        Returns:
            InlineShapeProtocol for the added image.
        """
        ...

    def add_page_break(self) -> ParagraphProtocol:
        """Add a page break to the document.

        Returns:
            ParagraphProtocol containing the page break.
        """
        ...

    def save(self, path: str | Path) -> None:
        """Save the document to a file."""
        ...


def _open_docx(path: Path) -> DocumentProtocol:
    """Open Word document with proper typing via Protocol.

    Args:
        path: Path to .docx file.

    Returns:
        DocumentProtocol for the opened document.
    """
    docx_mod = __import__("docx")
    doc: DocumentProtocol = docx_mod.Document(path)
    return doc


def _create_document() -> DocumentProtocol:
    """Create a new Word document with proper typing via Protocol.

    Returns:
        DocumentProtocol for the new document.
    """
    docx_mod = __import__("docx")
    doc: DocumentProtocol = docx_mod.Document()
    return doc


def _get_wd_align_center() -> int:
    """Get WD_ALIGN_PARAGRAPH.CENTER value.

    Returns:
        Integer value for center alignment.
    """
    enum_mod = __import__("docx.enum.text", fromlist=["WD_ALIGN_PARAGRAPH"])
    align_enum: WdAlignParagraphProtocol = enum_mod.WD_ALIGN_PARAGRAPH
    center_value: int = align_enum.CENTER
    return center_value


class _LengthCtor(Protocol):
    """Protocol for length constructor (Inches, Pt)."""

    def __call__(self, value: float) -> LengthProtocol:
        """Create length from value."""
        ...


def _get_inches(value: float) -> LengthProtocol:
    """Get Inches length object.

    Args:
        value: Size in inches.

    Returns:
        LengthProtocol representing the size.
    """
    shared_mod = __import__("docx.shared", fromlist=["Inches"])
    inches_fn: _LengthCtor = shared_mod.Inches
    result: LengthProtocol = inches_fn(value)
    return result


def _get_pt(value: float) -> LengthProtocol:
    """Get Pt (points) length object.

    Args:
        value: Size in points.

    Returns:
        LengthProtocol representing the size.
    """
    shared_mod = __import__("docx.shared", fromlist=["Pt"])
    pt_fn: _LengthCtor = shared_mod.Pt
    result: LengthProtocol = pt_fn(value)
    return result


__all__ = [
    "CellProtocol",
    "DocumentProtocol",
    "InlineShapeProtocol",
    "LengthProtocol",
    "ParagraphProtocol",
    "RowProtocol",
    "RunProtocol",
    "StyleProtocol",
    "TableProtocol",
    "WdAlignParagraphProtocol",
    "_create_document",
    "_get_inches",
    "_get_pt",
    "_get_wd_align_center",
    "_open_docx",
]
