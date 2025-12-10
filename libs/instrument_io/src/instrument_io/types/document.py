"""Document content type definitions for Word and PDF writers.

Provides TypedDicts for document sections that can be rendered
to Word (.docx) or PDF format by respective writers.

All types are immutable TypedDicts with strict typing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict, TypeGuard

from instrument_io.types.common import CellValue


class HeadingContent(TypedDict):
    """Heading section content.

    Attributes:
        type: Always "heading".
        text: Heading text.
        level: Heading level (1-6, where 1 is largest).
    """

    type: Literal["heading"]
    text: str
    level: int


class ParagraphContent(TypedDict):
    """Paragraph section content.

    Attributes:
        type: Always "paragraph".
        text: Paragraph text.
        bold: Whether entire paragraph is bold.
        italic: Whether entire paragraph is italic.
    """

    type: Literal["paragraph"]
    text: str
    bold: bool
    italic: bool


class TableContent(TypedDict):
    """Table section content.

    Attributes:
        type: Always "table".
        headers: Column header names.
        rows: List of row dictionaries with CellValue values.
        caption: Table caption (empty string if none).
    """

    type: Literal["table"]
    headers: list[str]
    rows: list[dict[str, CellValue]]
    caption: str


class FigureContent(TypedDict):
    """Figure/image section content.

    Attributes:
        type: Always "figure".
        path: Path to image file (PNG, JPEG).
        caption: Figure caption (empty string if none).
        width_inches: Width in inches (maintains aspect ratio). Use 0.0 for auto.
    """

    type: Literal["figure"]
    path: Path
    caption: str
    width_inches: float


class ListContent(TypedDict):
    """Bulleted or numbered list content.

    Attributes:
        type: Always "list".
        items: List item strings.
        ordered: True for numbered list, False for bullet list.
    """

    type: Literal["list"]
    items: list[str]
    ordered: bool


class PageBreakContent(TypedDict):
    """Page break marker.

    Attributes:
        type: Always "page_break".
    """

    type: Literal["page_break"]


# Union of all document content section types
DocumentSection = (
    HeadingContent
    | ParagraphContent
    | TableContent
    | FigureContent
    | ListContent
    | PageBreakContent
)

# A complete document is a sequence of sections
DocumentContent = list[DocumentSection]


# Page size type
PageSize = Literal["letter", "a4", "legal"]


# Page dimensions in points (72 points = 1 inch)
PAGE_SIZES: dict[PageSize, tuple[float, float]] = {
    "letter": (612.0, 792.0),  # 8.5 x 11 inches
    "a4": (595.28, 841.89),  # 210 x 297 mm (precise conversion)
    "legal": (612.0, 1008.0),  # 8.5 x 14 inches
}


def is_heading(section: DocumentSection) -> TypeGuard[HeadingContent]:
    """Check if section is a heading.

    Args:
        section: Document section to check.

    Returns:
        True if section is HeadingContent.
    """
    return section["type"] == "heading"


def is_paragraph(section: DocumentSection) -> TypeGuard[ParagraphContent]:
    """Check if section is a paragraph.

    Args:
        section: Document section to check.

    Returns:
        True if section is ParagraphContent.
    """
    return section["type"] == "paragraph"


def is_table(section: DocumentSection) -> TypeGuard[TableContent]:
    """Check if section is a table.

    Args:
        section: Document section to check.

    Returns:
        True if section is TableContent.
    """
    return section["type"] == "table"


def is_figure(section: DocumentSection) -> TypeGuard[FigureContent]:
    """Check if section is a figure.

    Args:
        section: Document section to check.

    Returns:
        True if section is FigureContent.
    """
    return section["type"] == "figure"


def is_list(section: DocumentSection) -> TypeGuard[ListContent]:
    """Check if section is a list.

    Args:
        section: Document section to check.

    Returns:
        True if section is ListContent.
    """
    return section["type"] == "list"


def is_page_break(section: DocumentSection) -> TypeGuard[PageBreakContent]:
    """Check if section is a page break.

    Args:
        section: Document section to check.

    Returns:
        True if section is PageBreakContent.
    """
    return section["type"] == "page_break"


__all__ = [
    "PAGE_SIZES",
    "DocumentContent",
    "DocumentSection",
    "FigureContent",
    "HeadingContent",
    "ListContent",
    "PageBreakContent",
    "PageSize",
    "ParagraphContent",
    "TableContent",
    "is_figure",
    "is_heading",
    "is_list",
    "is_page_break",
    "is_paragraph",
    "is_table",
]
