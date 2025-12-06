"""Protocol definitions for python-pptx library.

Provides type-safe interfaces to python-pptx Presentation, Slide, Shape, and related classes
without importing python-pptx directly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class CellProtocol(Protocol):
    """Protocol for pptx table Cell."""

    @property
    def text(self) -> str:
        """Return cell text content."""
        ...


class RowProtocol(Protocol):
    """Protocol for pptx table Row."""

    @property
    def cells(self) -> list[CellProtocol]:
        """Return list of cells in the row."""
        ...


class TableProtocol(Protocol):
    """Protocol for pptx Table shape."""

    @property
    def rows(self) -> list[RowProtocol]:
        """Return list of rows in the table."""
        ...


class TextFrameProtocol(Protocol):
    """Protocol for pptx TextFrame."""

    @property
    def text(self) -> str:
        """Return text content."""
        ...


class ShapeProtocol(Protocol):
    """Protocol for pptx Shape."""

    @property
    def has_text_frame(self) -> bool:
        """Return True if shape has text frame."""
        ...

    @property
    def text_frame(self) -> TextFrameProtocol:
        """Return text frame."""
        ...

    @property
    def has_table(self) -> bool:
        """Return True if shape is a table."""
        ...

    @property
    def table(self) -> TableProtocol:
        """Return table."""
        ...

    @property
    def shape_type(self) -> int:
        """Return shape type enum value."""
        ...


class ShapesIteratorProtocol(Protocol):
    """Protocol for pptx shapes collection iterator."""

    def __next__(self) -> ShapeProtocol:
        """Return next shape."""
        ...


class ShapesProtocol(Protocol):
    """Protocol for pptx Shapes collection."""

    def __iter__(self) -> ShapesIteratorProtocol:
        """Iterate over shapes."""
        ...

    def __len__(self) -> int:
        """Return number of shapes."""
        ...


class SlideProtocol(Protocol):
    """Protocol for pptx Slide."""

    @property
    def shapes(self) -> ShapesProtocol:
        """Return shapes collection."""
        ...


class SlidesIteratorProtocol(Protocol):
    """Protocol for pptx slides collection iterator."""

    def __next__(self) -> SlideProtocol:
        """Return next slide."""
        ...


class SlidesProtocol(Protocol):
    """Protocol for pptx Slides collection."""

    def __iter__(self) -> SlidesIteratorProtocol:
        """Iterate over slides."""
        ...

    def __len__(self) -> int:
        """Return number of slides."""
        ...


class PresentationProtocol(Protocol):
    """Protocol for pptx Presentation."""

    @property
    def slides(self) -> SlidesProtocol:
        """Return slides collection."""
        ...


def _open_pptx(path: Path) -> PresentationProtocol:
    """Open PowerPoint presentation with proper typing via Protocol.

    Args:
        path: Path to .pptx file.

    Returns:
        PresentationProtocol for the opened presentation.
    """
    pptx_mod = __import__("pptx")
    presentation_func: Callable[[str | Path], PresentationProtocol] = pptx_mod.Presentation
    prs: PresentationProtocol = presentation_func(path)
    return prs


__all__ = [
    "CellProtocol",
    "PresentationProtocol",
    "RowProtocol",
    "ShapeProtocol",
    "ShapesProtocol",
    "SlideProtocol",
    "SlidesProtocol",
    "TableProtocol",
    "TextFrameProtocol",
    "_open_pptx",
]
