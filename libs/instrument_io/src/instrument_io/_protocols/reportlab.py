"""Protocol definitions for reportlab library.

Provides type-safe interfaces to reportlab Platypus document generation
without importing reportlab directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict


class TableStyleCommand4(TypedDict):
    """Table style command with 4 elements (cmd, start, stop, value)."""

    cmd: str
    start: tuple[int, int]
    stop: tuple[int, int]
    value: str | int | float


class TableStyleCommand5(TypedDict):
    """Table style command with 5 elements (cmd, start, stop, value1, value2)."""

    cmd: str
    start: tuple[int, int]
    stop: tuple[int, int]
    value1: int | float
    value2: str


class CanvasProtocol(Protocol):
    """Protocol for reportlab Canvas."""

    def drawString(self, x: float, y: float, text: str) -> None:
        """Draw string at position."""
        ...

    def drawCentredString(self, x: float, y: float, text: str) -> None:
        """Draw string centered at position."""
        ...

    def saveState(self) -> None:
        """Save current graphics state."""
        ...

    def restoreState(self) -> None:
        """Restore saved graphics state."""
        ...

    def setFont(self, font_name: str, font_size: int) -> None:
        """Set current font."""
        ...

    def getPageNumber(self) -> int:
        """Get current page number."""
        ...


class StyleProtocol(Protocol):
    """Protocol for reportlab paragraph/table styles."""

    name: str
    fontSize: float
    leading: float


class ParagraphStyleProtocol(Protocol):
    """Protocol for reportlab ParagraphStyle."""

    name: str
    fontSize: float
    leading: float
    alignment: int
    spaceAfter: float
    spaceBefore: float


class TableStyleProtocol(Protocol):
    """Protocol for reportlab TableStyle."""

    pass


class FlowableProtocol(Protocol):
    """Protocol for reportlab Flowable base class."""

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:
        """Calculate space needed."""
        ...

    def drawOn(self, canvas: CanvasProtocol, x: float, y: float) -> None:
        """Draw flowable on canvas."""
        ...


class PageCallbackProtocol(Protocol):
    """Protocol for page callback functions used with onFirstPage/onLaterPages."""

    def __call__(self, canvas: CanvasProtocol, doc: SimpleDocTemplateProtocol) -> None:
        """Handle page rendering callback."""
        ...


class SimpleDocTemplateProtocol(Protocol):
    """Protocol for reportlab SimpleDocTemplate."""

    def build(
        self,
        flowables: list[FlowableProtocol],
        onFirstPage: PageCallbackProtocol | None = None,
        onLaterPages: PageCallbackProtocol | None = None,
    ) -> None:
        """Build PDF from flowables."""
        ...


class StyleSheetProtocol(Protocol):
    """Protocol for reportlab StyleSheet1."""

    def __getitem__(self, key: str) -> ParagraphStyleProtocol:
        """Get style by name."""
        ...


# Constructor protocols


class _SimpleDocTemplateCtor(Protocol):
    """Protocol for SimpleDocTemplate constructor."""

    def __call__(
        self,
        filename: str | Path,
        pagesize: tuple[float, float] | None = None,
        leftMargin: float = 72,
        rightMargin: float = 72,
        topMargin: float = 72,
        bottomMargin: float = 72,
    ) -> SimpleDocTemplateProtocol:
        """Create SimpleDocTemplate."""
        ...


class _ParagraphCtor(Protocol):
    """Protocol for Paragraph constructor."""

    def __call__(
        self,
        text: str,
        style: ParagraphStyleProtocol,
    ) -> FlowableProtocol:
        """Create Paragraph."""
        ...


class _TableCtor(Protocol):
    """Protocol for Table constructor."""

    def __call__(
        self,
        data: list[list[str]],
        colWidths: list[float] | None = None,
        rowHeights: list[float] | None = None,
        style: TableStyleProtocol | None = None,
    ) -> FlowableProtocol:
        """Create Table."""
        ...


class _ImageCtor(Protocol):
    """Protocol for Image constructor."""

    def __call__(
        self,
        filename: str | Path,
        width: float | None = None,
        height: float | None = None,
        kind: str | None = None,
    ) -> FlowableProtocol:
        """Create Image."""
        ...


class _SpacerCtor(Protocol):
    """Protocol for Spacer constructor."""

    def __call__(self, width: float, height: float) -> FlowableProtocol:
        """Create Spacer."""
        ...


class _PageBreakCtor(Protocol):
    """Protocol for PageBreak constructor."""

    def __call__(self) -> FlowableProtocol:
        """Create PageBreak."""
        ...


class _GetSampleStyleSheetFn(Protocol):
    """Protocol for getSampleStyleSheet function."""

    def __call__(self) -> StyleSheetProtocol:
        """Get sample stylesheet."""
        ...


class _RawTableStyleCtor(Protocol):
    """Protocol for raw TableStyle constructor (internal use only).

    Reportlab accepts a list of tuples. We use typed wrappers externally.
    """

    def __call__(
        self,
        cmds: list[
            tuple[str, tuple[int, int], tuple[int, int], str | int | float]
            | tuple[str, tuple[int, int], tuple[int, int], int | float, str]
        ],
    ) -> TableStyleProtocol:
        """Create TableStyle."""
        ...


class _ListFlowableCtor(Protocol):
    """Protocol for ListFlowable constructor."""

    def __call__(
        self,
        flowables: list[FlowableProtocol],
        bulletType: str = "bullet",
        bulletFormat: str | None = None,
        bulletFontSize: int | None = None,
        start: int | None = None,
        leftIndent: int | None = None,
    ) -> FlowableProtocol:
        """Create ListFlowable."""
        ...


class _ListItemCtor(Protocol):
    """Protocol for ListItem constructor."""

    def __call__(
        self,
        flowable: FlowableProtocol,
        leftIndent: float = 18,
        value: str | None = None,
    ) -> FlowableProtocol:
        """Create ListItem."""
        ...


class _ParagraphStyleCtor(Protocol):
    """Protocol for ParagraphStyle constructor."""

    def __call__(
        self,
        name: str,
        parent: ParagraphStyleProtocol | None = None,
        fontName: str | None = None,
        fontSize: float | None = None,
        leading: float | None = None,
        alignment: int | None = None,
        firstLineIndent: float | None = None,
        leftIndent: float | None = None,
        rightIndent: float | None = None,
        spaceBefore: float | None = None,
        spaceAfter: float | None = None,
        textColor: str | None = None,
    ) -> ParagraphStyleProtocol:
        """Create ParagraphStyle."""
        ...


# Helper functions
