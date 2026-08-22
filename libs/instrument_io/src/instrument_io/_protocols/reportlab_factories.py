"""ReportLab factory helpers built on the typed protocols.

The Protocols themselves live in :mod:`instrument_io._protocols.reportlab`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from instrument_io._protocols.reportlab import (
    CanvasProtocol,
    FlowableProtocol,
    PageCallbackProtocol,
    ParagraphStyleProtocol,
    SimpleDocTemplateProtocol,
    StyleProtocol,
    StyleSheetProtocol,
    TableStyleCommand4,
    TableStyleCommand5,
    TableStyleProtocol,
    _GetSampleStyleSheetFn,
    _ImageCtor,
    _ListFlowableCtor,
    _ListItemCtor,
    _PageBreakCtor,
    _ParagraphCtor,
    _RawTableStyleCtor,
    _SimpleDocTemplateCtor,
    _SpacerCtor,
    _TableCtor,
)


class PageNumberCallback:
    """Callback class that draws page numbers at bottom center."""

    def __init__(self, pagesize: tuple[float, float]) -> None:
        """Initialize with page dimensions.

        Args:
            pagesize: (width, height) in points, used to center the page number.
        """
        self._page_width = pagesize[0]

    def __call__(self, canvas: CanvasProtocol, doc: SimpleDocTemplateProtocol) -> None:
        """Draw page number at bottom center."""
        canvas.saveState()
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.setFont("Times-Roman", 11)
        canvas.drawCentredString(self._page_width / 2, 30, text)
        canvas.restoreState()


def _create_page_number_callback(
    pagesize: tuple[float, float],
) -> PageCallbackProtocol:
    """Create a page callback that draws page numbers at bottom center.

    Args:
        pagesize: (width, height) in points, used to center the page number.

    Returns:
        PageCallbackProtocol that draws "Page N" centered at the bottom.
    """
    return PageNumberCallback(pagesize)


def _create_simple_doc_template(
    filename: str | Path,
    pagesize: tuple[float, float],
    margins: tuple[float, float, float, float],
    show_page_numbers: bool = True,
) -> tuple[SimpleDocTemplateProtocol, PageCallbackProtocol | None]:
    """Create a SimpleDocTemplate with specified settings.

    Args:
        filename: Output file path.
        pagesize: (width, height) in points.
        margins: (left, right, top, bottom) in points.
        show_page_numbers: Whether to include a page number callback.

    Returns:
        Tuple of (SimpleDocTemplateProtocol, page_callback or None).
        Use the callback with doc.build(flowables, onFirstPage=cb, onLaterPages=cb).
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

    page_callback: PageCallbackProtocol | None = None
    if show_page_numbers:
        page_callback = _create_page_number_callback(pagesize)

    return doc, page_callback


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
        width: Optional width in points (maintains aspect ratio if height not set).
        height: Optional height in points.

    Returns:
        FlowableProtocol for the image.
    """
    platypus_mod = __import__("reportlab.platypus", fromlist=["Image"])
    ctor: _ImageCtor = platypus_mod.Image
    # If only width is specified, use proportional with a large height bound
    if width is not None and height is None:
        img: FlowableProtocol = ctor(str(path), width=width, height=width, kind="proportional")
    else:
        img = ctor(str(path), width=width, height=height)
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
        bulletFormat="%s." if ordered else None,
        bulletFontSize=11,
        start=start,
        leftIndent=6,
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
