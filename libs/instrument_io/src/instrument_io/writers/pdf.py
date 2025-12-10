"""PDF document writer implementation.

Provides typed writing of PDF documents via reportlab Platypus.
Uses Protocol-based dynamic imports for external libraries.
"""

from __future__ import annotations

from pathlib import Path

from instrument_io._exceptions import WriterError
from instrument_io._protocols.reportlab import (
    FlowableProtocol,
    ParagraphStyleProtocol,
    TableStyleCommand4,
    TableStyleCommand5,
    _create_image,
    _create_list_flowable,
    _create_list_item,
    _create_page_break,
    _create_paragraph,
    _create_paragraph_style,
    _create_simple_doc_template,
    _create_spacer,
    _create_table,
    _create_table_style_mixed,
)
from instrument_io.types.common import CellValue
from instrument_io.types.document import (
    PAGE_SIZES,
    DocumentContent,
    DocumentSection,
    FigureContent,
    HeadingContent,
    ListContent,
    PageSize,
    ParagraphContent,
    TableContent,
    is_figure,
    is_heading,
    is_list,
    is_paragraph,
    is_table,
)

POINTS_PER_INCH = 72.0

# Research paper format constants
BODY_FONT = "Times-Roman"
BODY_FONT_BOLD = "Times-Bold"
BODY_FONT_ITALIC = "Times-Italic"
BODY_FONT_SIZE = 11.0
BODY_LEADING = 14.0  # Single-spaced with comfortable line height
PARAGRAPH_SPACE = 10.0  # Space between paragraphs
HEADING_SPACE_BEFORE = 12.0  # Space before headings
HEADING_SPACE_AFTER = 2.0  # Space after headings


def _create_research_styles() -> dict[str, ParagraphStyleProtocol]:
    """Create research paper paragraph styles.

    Research format uses:
    - Times Roman 11pt body text
    - Single-spaced with space between paragraphs
    - No first-line indent (block paragraphs)
    - Centered title, left-aligned section headings

    Returns:
        Dictionary of style name to ParagraphStyleProtocol.
    """
    # Title style - centered, bold, 16pt
    title = _create_paragraph_style(
        "Title",
        font_name=BODY_FONT_BOLD,
        font_size=16.0,
        leading=20.0,
        alignment=1,  # CENTER
        space_before=0.0,
        space_after=18.0,
    )

    # Heading1 - bold, 14pt, left-aligned
    heading1 = _create_paragraph_style(
        "Heading1",
        font_name=BODY_FONT_BOLD,
        font_size=14.0,
        leading=18.0,
        alignment=0,  # LEFT
        space_before=HEADING_SPACE_BEFORE,
        space_after=HEADING_SPACE_AFTER,
    )

    # Heading2 - bold, 12pt, left-aligned
    heading2 = _create_paragraph_style(
        "Heading2",
        font_name=BODY_FONT_BOLD,
        font_size=12.0,
        leading=16.0,
        alignment=0,  # LEFT
        space_before=HEADING_SPACE_BEFORE,
        space_after=HEADING_SPACE_AFTER,
    )

    # Heading3 - italic, 11pt, left-aligned
    heading3 = _create_paragraph_style(
        "Heading3",
        font_name=BODY_FONT_ITALIC,
        font_size=BODY_FONT_SIZE,
        leading=BODY_LEADING,
        alignment=0,  # LEFT
        space_before=12.0,
        space_after=4.0,
    )

    # Normal body text - no indent, space between paragraphs
    normal = _create_paragraph_style(
        "Normal",
        font_name=BODY_FONT,
        font_size=BODY_FONT_SIZE,
        leading=BODY_LEADING,
        alignment=0,  # LEFT
        first_line_indent=0.0,
        space_before=0.0,
        space_after=PARAGRAPH_SPACE,
    )

    # Caption style - italic, centered
    caption = _create_paragraph_style(
        "Caption",
        font_name=BODY_FONT_ITALIC,
        font_size=10.0,
        leading=12.0,
        alignment=1,  # CENTER
        first_line_indent=0.0,
        space_before=4.0,
        space_after=12.0,
    )

    # List item style - no first-line indent, slight left margin
    list_item = _create_paragraph_style(
        "ListItem",
        font_name=BODY_FONT,
        font_size=BODY_FONT_SIZE,
        leading=BODY_LEADING,
        alignment=0,  # LEFT
        first_line_indent=0.0,
        left_indent=18.0,
        space_before=0.0,
        space_after=2.0,
    )

    return {
        "Title": title,
        "Heading1": heading1,
        "Heading2": heading2,
        "Heading3": heading3,
        "Normal": normal,
        "Caption": caption,
        "ListItem": list_item,
    }


def _heading_style_name(level: int, is_title: bool = False) -> str:
    """Map heading level to style name.

    Args:
        level: Heading level (1-6).
        is_title: Whether this is the document title (uses centered Title style).

    Returns:
        Style name for the heading level.
    """
    if is_title and level == 1:
        return "Title"
    if level == 1:
        return "Heading1"
    if level == 2:
        return "Heading2"
    if level == 3:
        return "Heading3"
    return "Heading3"


def _render_heading_pdf(
    content: HeadingContent,
    styles: dict[str, ParagraphStyleProtocol],
    is_title: bool = False,
) -> FlowableProtocol:
    """Render heading to PDF flowable.

    Args:
        content: Heading content.
        styles: Available styles.
        is_title: Whether this is the document title.

    Returns:
        Paragraph flowable for the heading.
    """
    level = content["level"]
    clamped_level = max(1, min(level, 6))
    style_name = _heading_style_name(clamped_level, is_title=is_title)
    style = styles.get(style_name, styles["Normal"])
    return _create_paragraph(content["text"], style)


def _render_paragraph_pdf(
    content: ParagraphContent,
    styles: dict[str, ParagraphStyleProtocol],
) -> FlowableProtocol:
    """Render paragraph to PDF flowable.

    Args:
        content: Paragraph content.
        styles: Available styles.

    Returns:
        Paragraph flowable.
    """
    text = content["text"]
    bold = content["bold"]
    italic = content["italic"]

    if bold:
        text = f"<b>{text}</b>"
    if italic:
        text = f"<i>{text}</i>"

    style = styles["Normal"]
    return _create_paragraph(text, style)


def _render_table_pdf(
    content: TableContent,
    styles: dict[str, ParagraphStyleProtocol],
) -> list[FlowableProtocol]:
    """Render table to PDF flowables.

    Args:
        content: Table content.
        styles: Available styles.

    Returns:
        List of flowables (table + optional caption).
    """
    headers = content["headers"]
    rows = content["rows"]

    if not headers:
        return []

    data: list[list[str]] = [headers]
    for row in rows:
        row_data: list[str] = []
        for header in headers:
            value: CellValue = row.get(header)
            row_data.append(str(value) if value is not None else "")
        data.append(row_data)

    commands4: list[TableStyleCommand4] = [
        TableStyleCommand4(cmd="BACKGROUND", start=(0, 0), stop=(-1, 0), value="#CCCCCC"),
        TableStyleCommand4(cmd="TEXTCOLOR", start=(0, 0), stop=(-1, 0), value="#000000"),
        TableStyleCommand4(cmd="ALIGN", start=(0, 0), stop=(-1, -1), value="CENTER"),
        TableStyleCommand4(cmd="FONTNAME", start=(0, 0), stop=(-1, 0), value="Helvetica-Bold"),
        TableStyleCommand4(cmd="FONTSIZE", start=(0, 0), stop=(-1, 0), value=10),
        TableStyleCommand4(cmd="BOTTOMPADDING", start=(0, 0), stop=(-1, 0), value=12),
        TableStyleCommand4(cmd="BACKGROUND", start=(0, 1), stop=(-1, -1), value="#FFFFFF"),
    ]
    commands5: list[TableStyleCommand5] = [
        TableStyleCommand5(cmd="GRID", start=(0, 0), stop=(-1, -1), value1=1, value2="#000000"),
    ]
    style = _create_table_style_mixed(commands4, commands5)

    table = _create_table(data, style=style)
    flowables: list[FlowableProtocol] = [table]

    caption = content["caption"]
    if caption:
        caption_style = styles.get("Caption", styles["Normal"])
        caption_para = _create_paragraph(caption, caption_style)
        spacer = _create_spacer(0, 4)
        flowables.append(spacer)
        flowables.append(caption_para)

    return flowables


def _render_figure_pdf(
    content: FigureContent,
    styles: dict[str, ParagraphStyleProtocol],
) -> list[FlowableProtocol]:
    """Render figure/image to PDF flowables.

    Args:
        content: Figure content.
        styles: Available styles.

    Returns:
        List of flowables (image + optional caption).

    Raises:
        WriterError: If image file not found.
    """
    image_path = content["path"]
    if not image_path.exists():
        raise WriterError(str(image_path), "Image file not found")

    width_inches = content["width_inches"]
    # Use specified width, or default to 5 inches for auto-sizing
    if width_inches > 0.0:
        width_points: float = width_inches * POINTS_PER_INCH
    else:
        width_points = 5.0 * POINTS_PER_INCH  # Default to 5 inches

    image = _create_image(image_path, width=width_points)
    flowables: list[FlowableProtocol] = [image]

    caption = content["caption"]
    if caption:
        caption_style = styles.get("Caption", styles["Normal"])
        caption_para = _create_paragraph(caption, caption_style)
        spacer = _create_spacer(0, 4)
        flowables.append(spacer)
        flowables.append(caption_para)

    return flowables


def _render_list_pdf(
    content: ListContent,
    styles: dict[str, ParagraphStyleProtocol],
) -> FlowableProtocol:
    """Render list to PDF flowable.

    Args:
        content: List content.
        styles: Available styles.

    Returns:
        ListFlowable containing items.
    """
    items = content["items"]
    ordered = content["ordered"]
    style = styles.get("ListItem", styles["Normal"])

    list_items: list[FlowableProtocol] = []
    for item_text in items:
        para = _create_paragraph(item_text, style)
        list_item = _create_list_item(para)
        list_items.append(list_item)

    return _create_list_flowable(list_items, ordered=ordered)


def _render_page_break_pdf() -> FlowableProtocol:
    """Render page break to PDF flowable.

    Returns:
        PageBreak flowable.
    """
    return _create_page_break()


def _section_to_flowables(
    section: DocumentSection,
    styles: dict[str, ParagraphStyleProtocol],
    is_first_heading: bool = False,
) -> list[FlowableProtocol]:
    """Convert section to PDF flowables.

    Args:
        section: Section content.
        styles: Available styles.
        is_first_heading: Whether this is the first heading (for title styling).

    Returns:
        List of flowables for the section.
    """
    if is_heading(section):
        return [_render_heading_pdf(section, styles, is_title=is_first_heading)]
    if is_paragraph(section):
        return [_render_paragraph_pdf(section, styles)]
    if is_table(section):
        return _render_table_pdf(section, styles)
    if is_figure(section):
        return _render_figure_pdf(section, styles)
    if is_list(section):
        return [_render_list_pdf(section, styles)]
    # is_page_break(section) - exhaustive match on DocumentSection union
    return [_render_page_break_pdf()]


class PDFWriter:
    """Writer for PDF documents.

    Provides typed writing of structured document content to PDF format
    via reportlab Platypus with Protocol-based typing.

    All methods raise exceptions on failure.
    """

    def __init__(
        self,
        *,
        page_size: PageSize = "letter",
        margin_inches: float = 1.0,
    ) -> None:
        """Initialize PDF writer.

        Args:
            page_size: Page size (letter, a4, legal).
            margin_inches: Page margins in inches (applied to all sides).
        """
        self._page_size = page_size
        self._margin_points = margin_inches * POINTS_PER_INCH

    def write_document(
        self,
        content: DocumentContent,
        out_path: Path,
    ) -> None:
        """Write document content to PDF file.

        Args:
            content: List of document sections to write.
            out_path: Output file path (.pdf).

        Raises:
            WriterError: If writing fails or content is empty.
        """
        if not content:
            raise WriterError(str(out_path), "No content provided")

        actual_path = out_path
        if actual_path.suffix.lower() != ".pdf":
            actual_path = actual_path.with_suffix(".pdf")

        actual_path.parent.mkdir(parents=True, exist_ok=True)

        page_dims = PAGE_SIZES[self._page_size]

        margins = (
            self._margin_points,
            self._margin_points,
            self._margin_points,
            self._margin_points,
        )
        doc, page_callback = _create_simple_doc_template(actual_path, page_dims, margins)

        # Use research paper styles
        styles = _create_research_styles()

        flowables: list[FlowableProtocol] = []
        first_heading_seen = False

        for section in content:
            # Check if this is the first heading (for title styling)
            is_first_heading = False
            if is_heading(section) and not first_heading_seen:
                is_first_heading = True
                first_heading_seen = True

            section_flowables = _section_to_flowables(
                section, styles, is_first_heading=is_first_heading
            )
            flowables.extend(section_flowables)

        doc.build(flowables, onFirstPage=page_callback, onLaterPages=page_callback)


__all__ = [
    "PDFWriter",
]
