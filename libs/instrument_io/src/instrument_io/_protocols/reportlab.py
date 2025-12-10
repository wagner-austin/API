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


class SimpleDocTemplateProtocol(Protocol):
    """Protocol for reportlab SimpleDocTemplate."""

    def build(self, flowables: list[FlowableProtocol]) -> None:
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
        start: int | None = None,
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


def _create_simple_doc_template(
    filename: str | Path,
    pagesize: tuple[float, float],
    margins: tuple[float, float, float, float],
) -> SimpleDocTemplateProtocol:
    """Create a SimpleDocTemplate with specified settings.

    Args:
        filename: Output file path.
        pagesize: (width, height) in points.
        margins: (left, right, top, bottom) in points.

    Returns:
        SimpleDocTemplateProtocol for building the document.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["SimpleDocTemplate"])
    ctor: _SimpleDocTemplateCtor = platypus_mod.SimpleDocTemplate
    left, right, top, bottom = margins
    doc: SimpleDocTemplateProtocol = ctor(
        str(filename),
        pagesize=pagesize,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
    )
    return doc


def _get_sample_stylesheet() -> StyleSheetProtocol:
    """Get the default reportlab stylesheet.

    Returns:
        StyleSheetProtocol with standard styles.
    """
    styles_mod = __import__("reportlab.lib.styles", fromlist=["getSampleStyleSheet"])
    get_fn: _GetSampleStyleSheetFn = styles_mod.getSampleStyleSheet
    stylesheet: StyleSheetProtocol = get_fn()
    return stylesheet


def _create_paragraph(
    text: str,
    style: ParagraphStyleProtocol,
) -> FlowableProtocol:
    """Create a Paragraph flowable.

    Args:
        text: Paragraph text (can include basic HTML tags).
        style: ParagraphStyle to apply.

    Returns:
        FlowableProtocol for the paragraph.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["Paragraph"])
    ctor: _ParagraphCtor = platypus_mod.Paragraph
    para: FlowableProtocol = ctor(text, style)
    return para


def _create_table(
    data: list[list[str]],
    col_widths: list[float] | None = None,
    style: TableStyleProtocol | None = None,
) -> FlowableProtocol:
    """Create a Table flowable.

    Args:
        data: 2D list of cell values.
        col_widths: Optional column widths in points.
        style: Optional TableStyle.

    Returns:
        FlowableProtocol for the table.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["Table"])
    ctor: _TableCtor = platypus_mod.Table
    table: FlowableProtocol = ctor(data, colWidths=col_widths, style=style)
    return table


def _command4_to_tuple(
    cmd: TableStyleCommand4,
) -> tuple[str, tuple[int, int], tuple[int, int], str | int | float]:
    """Convert 4-element command TypedDict to tuple.

    Args:
        cmd: TableStyleCommand4 TypedDict.

    Returns:
        Tuple suitable for reportlab TableStyle.
    """
    return (cmd["cmd"], cmd["start"], cmd["stop"], cmd["value"])


def _command5_to_tuple(
    cmd: TableStyleCommand5,
) -> tuple[str, tuple[int, int], tuple[int, int], int | float, str]:
    """Convert 5-element command TypedDict to tuple.

    Args:
        cmd: TableStyleCommand5 TypedDict.

    Returns:
        Tuple suitable for reportlab TableStyle.
    """
    return (cmd["cmd"], cmd["start"], cmd["stop"], cmd["value1"], cmd["value2"])


def _create_table_style_from_commands4(
    commands: list[TableStyleCommand4],
) -> TableStyleProtocol:
    """Create a TableStyle from 4-element commands.

    Args:
        commands: List of 4-element style command TypedDicts.

    Returns:
        TableStyleProtocol for table formatting.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["TableStyle"])
    ctor: _RawTableStyleCtor = platypus_mod.TableStyle
    raw_cmds: list[
        tuple[str, tuple[int, int], tuple[int, int], str | int | float]
        | tuple[str, tuple[int, int], tuple[int, int], int | float, str]
    ] = [_command4_to_tuple(c) for c in commands]
    style: TableStyleProtocol = ctor(raw_cmds)
    return style


def _create_table_style_from_commands5(
    commands: list[TableStyleCommand5],
) -> TableStyleProtocol:
    """Create a TableStyle from 5-element commands.

    Args:
        commands: List of 5-element style command TypedDicts.

    Returns:
        TableStyleProtocol for table formatting.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["TableStyle"])
    ctor: _RawTableStyleCtor = platypus_mod.TableStyle
    raw_cmds: list[
        tuple[str, tuple[int, int], tuple[int, int], str | int | float]
        | tuple[str, tuple[int, int], tuple[int, int], int | float, str]
    ] = [_command5_to_tuple(c) for c in commands]
    style: TableStyleProtocol = ctor(raw_cmds)
    return style


def _create_table_style_mixed(
    commands4: list[TableStyleCommand4],
    commands5: list[TableStyleCommand5],
) -> TableStyleProtocol:
    """Create a TableStyle from mixed 4 and 5-element commands.

    Args:
        commands4: List of 4-element style command TypedDicts.
        commands5: List of 5-element style command TypedDicts.

    Returns:
        TableStyleProtocol for table formatting.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["TableStyle"])
    ctor: _RawTableStyleCtor = platypus_mod.TableStyle
    raw_cmds: list[
        tuple[str, tuple[int, int], tuple[int, int], str | int | float]
        | tuple[str, tuple[int, int], tuple[int, int], int | float, str]
    ] = []
    for cmd4 in commands4:
        raw_cmds.append(_command4_to_tuple(cmd4))
    for cmd5 in commands5:
        raw_cmds.append(_command5_to_tuple(cmd5))
    style: TableStyleProtocol = ctor(raw_cmds)
    return style


def _create_image(
    path: str | Path,
    width: float | None = None,
    height: float | None = None,
) -> FlowableProtocol:
    """Create an Image flowable.

    Args:
        path: Path to image file.
        width: Optional width in points.
        height: Optional height in points.

    Returns:
        FlowableProtocol for the image.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["Image"])
    ctor: _ImageCtor = platypus_mod.Image
    img: FlowableProtocol = ctor(str(path), width=width, height=height)
    return img


def _create_spacer(width: float, height: float) -> FlowableProtocol:
    """Create a Spacer flowable.

    Args:
        width: Width in points.
        height: Height in points.

    Returns:
        FlowableProtocol for the spacer.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["Spacer"])
    ctor: _SpacerCtor = platypus_mod.Spacer
    spacer: FlowableProtocol = ctor(width, height)
    return spacer


def _create_page_break() -> FlowableProtocol:
    """Create a PageBreak flowable.

    Returns:
        FlowableProtocol for the page break.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["PageBreak"])
    ctor: _PageBreakCtor = platypus_mod.PageBreak
    page_break: FlowableProtocol = ctor()
    return page_break


def _create_list_flowable(
    items: list[FlowableProtocol],
    ordered: bool = False,
) -> FlowableProtocol:
    """Create a ListFlowable (bulleted or numbered list).

    Args:
        items: List of ListItem flowables.
        ordered: True for numbered, False for bullets.

    Returns:
        FlowableProtocol containing the items.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["ListFlowable"])
    ctor: _ListFlowableCtor = platypus_mod.ListFlowable
    bullet_type = "1" if ordered else "bullet"
    start = 1 if ordered else None
    list_flow: FlowableProtocol = ctor(
        items,
        bulletType=bullet_type,
        start=start,
    )
    return list_flow


def _create_list_item(
    flowable: FlowableProtocol,
    left_indent: float = 18,
) -> FlowableProtocol:
    """Create a ListItem wrapper for list content.

    Args:
        flowable: Content flowable for the item.
        left_indent: Left indentation in points.

    Returns:
        FlowableProtocol wrapped as list item.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["ListItem"])
    ctor: _ListItemCtor = platypus_mod.ListItem
    item: FlowableProtocol = ctor(flowable, leftIndent=left_indent)
    return item


def _create_paragraph_style(
    name: str,
    parent: ParagraphStyleProtocol | None = None,
    font_name: str | None = None,
    font_size: float | None = None,
    leading: float | None = None,
    alignment: int | None = None,
    first_line_indent: float | None = None,
    left_indent: float | None = None,
    right_indent: float | None = None,
    space_before: float | None = None,
    space_after: float | None = None,
) -> ParagraphStyleProtocol:
    """Create a custom ParagraphStyle.

    Args:
        name: Style name.
        parent: Parent style to inherit from.
        font_name: Font name (e.g., 'Times-Roman', 'Helvetica').
        font_size: Font size in points.
        leading: Line height in points (for double-spacing, use font_size * 2).
        alignment: Text alignment (0=left, 1=center, 2=right, 4=justify).
        first_line_indent: First line indentation in points.
        left_indent: Left margin indent in points.
        right_indent: Right margin indent in points.
        space_before: Space before paragraph in points.
        space_after: Space after paragraph in points.

    Returns:
        ParagraphStyleProtocol with specified settings.
    """
    styles_mod = __import__("reportlab.lib.styles", fromlist=["ParagraphStyle"])

    # Build kwargs dict with only non-None values
    # reportlab doesn't handle None values well
    kwargs: dict[str, str | float | int | ParagraphStyleProtocol] = {}
    if parent is not None:
        kwargs["parent"] = parent
    if font_name is not None:
        kwargs["fontName"] = font_name
    if font_size is not None:
        kwargs["fontSize"] = font_size
    if leading is not None:
        kwargs["leading"] = leading
    if alignment is not None:
        kwargs["alignment"] = alignment
    if first_line_indent is not None:
        kwargs["firstLineIndent"] = first_line_indent
    if left_indent is not None:
        kwargs["leftIndent"] = left_indent
    if right_indent is not None:
        kwargs["rightIndent"] = right_indent
    if space_before is not None:
        kwargs["spaceBefore"] = space_before
    if space_after is not None:
        kwargs["spaceAfter"] = space_after

    # Define a callable type for the constructor with **kwargs
    class _ParagraphStyleKwargsCtor(Protocol):
        def __call__(
            self,
            name: str,
            **kwargs: str | float | int | ParagraphStyleProtocol,
        ) -> ParagraphStyleProtocol: ...

    ctor: _ParagraphStyleKwargsCtor = styles_mod.ParagraphStyle
    style: ParagraphStyleProtocol = ctor(name, **kwargs)
    return style


__all__ = [
    "CanvasProtocol",
    "FlowableProtocol",
    "ParagraphStyleProtocol",
    "SimpleDocTemplateProtocol",
    "StyleProtocol",
    "StyleSheetProtocol",
    "TableStyleCommand4",
    "TableStyleCommand5",
    "TableStyleProtocol",
    "_command4_to_tuple",
    "_command5_to_tuple",
    "_create_image",
    "_create_list_flowable",
    "_create_list_item",
    "_create_page_break",
    "_create_paragraph",
    "_create_paragraph_style",
    "_create_simple_doc_template",
    "_create_spacer",
    "_create_table",
    "_create_table_style_from_commands4",
    "_create_table_style_from_commands5",
    "_create_table_style_mixed",
    "_get_sample_stylesheet",
]
