"""Word document (.docx) writer implementation.

Provides typed writing of Word documents via python-docx.
Uses Protocol-based dynamic imports for external libraries.
"""

from __future__ import annotations

from pathlib import Path

from instrument_io._exceptions import WriterError
from instrument_io._protocols.python_docx import (
    DocumentProtocol,
    _create_document,
    _get_inches,
    _get_wd_align_center,
)
from instrument_io.types.common import CellValue
from instrument_io.types.document import (
    DocumentContent,
    DocumentSection,
    FigureContent,
    HeadingContent,
    ListContent,
    ParagraphContent,
    TableContent,
    is_figure,
    is_heading,
    is_list,
    is_paragraph,
    is_table,
)


def _render_heading(doc: DocumentProtocol, content: HeadingContent) -> None:
    """Render heading to document.

    Args:
        doc: Document to write to.
        content: Heading content.
    """
    level = content["level"]
    clamped_level = max(0, min(level, 9))
    doc.add_heading(content["text"], level=clamped_level)


def _render_paragraph(doc: DocumentProtocol, content: ParagraphContent) -> None:
    """Render paragraph to document.

    Args:
        doc: Document to write to.
        content: Paragraph content.
    """
    para = doc.add_paragraph(content["text"])

    bold = content["bold"]
    italic = content["italic"]

    if bold:
        for run in para.runs:
            run.bold = True

    if italic:
        for run in para.runs:
            run.italic = True


def _render_table(doc: DocumentProtocol, content: TableContent) -> None:
    """Render table to document.

    Args:
        doc: Document to write to.
        content: Table content.
    """
    headers = content["headers"]
    rows = content["rows"]

    if not headers:
        return

    num_rows = len(rows) + 1
    num_cols = len(headers)
    table = doc.add_table(rows=num_rows, cols=num_cols)
    table.style = "Table Grid"

    header_cells = table.rows[0].cells
    for idx, header in enumerate(headers):
        header_cells[idx].text = header
        for para in header_cells[idx].paragraphs:
            for run in para.runs:
                run.bold = True

    for row_idx, row_data in enumerate(rows, start=1):
        row_cells = table.rows[row_idx].cells
        for col_idx, header in enumerate(headers):
            value: CellValue = row_data.get(header)
            if value is not None:
                row_cells[col_idx].text = str(value)

    caption = content["caption"]
    if caption:
        caption_para = doc.add_paragraph(caption)
        caption_para.alignment = _get_wd_align_center()


def _render_figure(doc: DocumentProtocol, content: FigureContent) -> None:
    """Render figure/image to document.

    Args:
        doc: Document to write to.
        content: Figure content.

    Raises:
        WriterError: If image file not found.
    """
    image_path = content["path"]
    if not image_path.exists():
        raise WriterError(str(image_path), "Image file not found")

    width_inches = content["width_inches"]
    if width_inches > 0.0:
        width = _get_inches(width_inches)
        doc.add_picture(str(image_path), width=width)
    else:
        doc.add_picture(str(image_path))

    caption = content["caption"]
    if caption:
        caption_para = doc.add_paragraph(caption)
        caption_para.alignment = _get_wd_align_center()


def _render_list(doc: DocumentProtocol, content: ListContent) -> None:
    """Render list to document.

    Args:
        doc: Document to write to.
        content: List content.
    """
    items = content["items"]
    ordered = content["ordered"]

    style = "List Number" if ordered else "List Bullet"

    for item in items:
        doc.add_paragraph(item, style=style)


def _render_page_break(doc: DocumentProtocol) -> None:
    """Render page break to document.

    Args:
        doc: Document to write to.
    """
    doc.add_page_break()


def _render_section(doc: DocumentProtocol, section: DocumentSection) -> None:
    """Render a document section based on its type.

    Args:
        doc: Document to write to.
        section: Section content.
    """
    if is_heading(section):
        _render_heading(doc, section)
    elif is_paragraph(section):
        _render_paragraph(doc, section)
    elif is_table(section):
        _render_table(doc, section)
    elif is_figure(section):
        _render_figure(doc, section)
    elif is_list(section):
        _render_list(doc, section)
    else:
        # Exhaustive: only PageBreakContent remains
        _render_page_break(doc)


class WordWriter:
    """Writer for Word documents (.docx).

    Provides typed writing of structured document content to Word format
    via python-docx with Protocol-based typing.

    All methods raise exceptions on failure.
    """

    def __init__(
        self,
        *,
        title: str = "",
        author: str = "",
    ) -> None:
        """Initialize Word writer.

        Args:
            title: Document title metadata.
            author: Document author metadata.
        """
        self._title = title
        self._author = author

    def write_document(
        self,
        content: DocumentContent,
        out_path: Path,
    ) -> None:
        """Write document content to Word file.

        Args:
            content: List of document sections to write.
            out_path: Output file path (.docx).

        Raises:
            WriterError: If writing fails or content is empty.
        """
        if not content:
            raise WriterError(str(out_path), "No content provided")

        actual_path = out_path
        if actual_path.suffix.lower() != ".docx":
            actual_path = actual_path.with_suffix(".docx")

        doc: DocumentProtocol = _create_document()

        for section in content:
            _render_section(doc, section)

        actual_path.parent.mkdir(parents=True, exist_ok=True)

        doc.save(actual_path)


__all__ = [
    "WordWriter",
]
