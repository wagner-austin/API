"""Tests for writers.pdf module."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import WriterError
from instrument_io.types.document import DocumentContent
from instrument_io.writers.pdf import PDFWriter

# Valid 1x1 pixel PNG (green pixel) - generated with correct CRC checksums
VALID_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"  # PNG signature
    b"\x00\x00\x00\rIHDR"  # IHDR chunk length and type
    b"\x00\x00\x00\x01"  # width = 1
    b"\x00\x00\x00\x01"  # height = 1
    b"\x08\x02"  # bit depth = 8, color type = 2 (RGB)
    b"\x00\x00\x00"  # compression, filter, interlace
    b"\x90wS\xde"  # IHDR CRC
    b"\x00\x00\x00\x0c"  # IDAT chunk length
    b"IDATx\xdac\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00"  # IDAT data
    b"\xf7\x03AC"  # IDAT CRC
    b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND chunk
)


class TestPDFWriter:
    """Tests for PDFWriter class."""

    def test_write_document_creates_file(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {"type": "heading", "text": "Test Document", "level": 1},
            {"type": "paragraph", "text": "This is a test.", "bold": False, "italic": False},
        ]
        out_path = tmp_path / "output.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_write_document_empty_raises(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        out_path = tmp_path / "empty.pdf"
        with pytest.raises(WriterError):
            writer.write_document([], out_path)

    def test_write_document_adds_extension(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {"type": "paragraph", "text": "Test", "bold": False, "italic": False},
        ]
        out_path = tmp_path / "no_extension"
        writer.write_document(content, out_path)
        actual_path = out_path.with_suffix(".pdf")
        assert actual_path.exists()

    def test_write_document_creates_parent_dirs(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {"type": "paragraph", "text": "Test", "bold": False, "italic": False},
        ]
        out_path = tmp_path / "subdir" / "nested" / "output.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_page_size_letter(self, tmp_path: Path) -> None:
        writer = PDFWriter(page_size="letter")
        content: DocumentContent = [
            {"type": "paragraph", "text": "Letter size", "bold": False, "italic": False},
        ]
        out_path = tmp_path / "letter.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_page_size_a4(self, tmp_path: Path) -> None:
        writer = PDFWriter(page_size="a4")
        content: DocumentContent = [
            {"type": "paragraph", "text": "A4 size", "bold": False, "italic": False},
        ]
        out_path = tmp_path / "a4.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_page_size_legal(self, tmp_path: Path) -> None:
        writer = PDFWriter(page_size="legal")
        content: DocumentContent = [
            {"type": "paragraph", "text": "Legal size", "bold": False, "italic": False},
        ]
        out_path = tmp_path / "legal.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_custom_margins(self, tmp_path: Path) -> None:
        writer = PDFWriter(margin_inches=0.5)
        content: DocumentContent = [
            {"type": "paragraph", "text": "Custom margins", "bold": False, "italic": False},
        ]
        out_path = tmp_path / "margins.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_heading_levels(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {"type": "heading", "text": "Heading 1", "level": 1},
            {"type": "heading", "text": "Heading 2", "level": 2},
            {"type": "heading", "text": "Heading 3", "level": 3},
            {"type": "heading", "text": "Heading 4", "level": 4},
            {"type": "heading", "text": "Heading 5", "level": 5},
            {"type": "heading", "text": "Heading 6", "level": 6},
        ]
        out_path = tmp_path / "headings.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_bold_paragraph(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {"type": "paragraph", "text": "Bold text", "bold": True, "italic": False},
        ]
        out_path = tmp_path / "bold.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_italic_paragraph(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {"type": "paragraph", "text": "Italic text", "bold": False, "italic": True},
        ]
        out_path = tmp_path / "italic.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_bold_italic(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {"type": "paragraph", "text": "Bold and italic", "bold": True, "italic": True},
        ]
        out_path = tmp_path / "bold_italic.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_table(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {
                "type": "table",
                "headers": ["Name", "Value"],
                "rows": [
                    {"Name": "Item1", "Value": 100},
                    {"Name": "Item2", "Value": 200},
                ],
                "caption": "Test Table",
            },
        ]
        out_path = tmp_path / "table.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_table_no_caption(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {
                "type": "table",
                "headers": ["A", "B"],
                "rows": [{"A": 1, "B": 2}],
                "caption": "",
            },
        ]
        out_path = tmp_path / "table_no_caption.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_empty_table(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {
                "type": "table",
                "headers": [],
                "rows": [],
                "caption": "",
            },
        ]
        out_path = tmp_path / "empty_table.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_ordered_list(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {
                "type": "list",
                "items": ["First", "Second", "Third"],
                "ordered": True,
            },
        ]
        out_path = tmp_path / "ordered_list.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_unordered_list(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {
                "type": "list",
                "items": ["Bullet 1", "Bullet 2", "Bullet 3"],
                "ordered": False,
            },
        ]
        out_path = tmp_path / "unordered_list.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_page_break(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {"type": "paragraph", "text": "Page 1", "bold": False, "italic": False},
            {"type": "page_break"},
            {"type": "paragraph", "text": "Page 2", "bold": False, "italic": False},
        ]
        out_path = tmp_path / "page_break.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_figure(self, tmp_path: Path) -> None:
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(VALID_PNG_BYTES)

        writer = PDFWriter()
        content: DocumentContent = [
            {
                "type": "figure",
                "path": image_path,
                "caption": "Test Figure",
                "width_inches": 2.0,
            },
        ]
        out_path = tmp_path / "figure.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_figure_no_caption(self, tmp_path: Path) -> None:
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(VALID_PNG_BYTES)

        writer = PDFWriter()
        content: DocumentContent = [
            {
                "type": "figure",
                "path": image_path,
                "caption": "",
                "width_inches": 2.0,
            },
        ]
        out_path = tmp_path / "figure_no_caption.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_with_figure_no_width(self, tmp_path: Path) -> None:
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(VALID_PNG_BYTES)

        writer = PDFWriter()
        content: DocumentContent = [
            {
                "type": "figure",
                "path": image_path,
                "caption": "Figure without explicit width",
                "width_inches": 0.0,
            },
        ]
        out_path = tmp_path / "figure_no_width.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_figure_not_found_raises(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {
                "type": "figure",
                "path": tmp_path / "nonexistent.png",
                "caption": "Missing",
                "width_inches": 2.0,
            },
        ]
        out_path = tmp_path / "figure_missing.pdf"
        with pytest.raises(WriterError):
            writer.write_document(content, out_path)

    def test_write_document_full_example(self, tmp_path: Path) -> None:
        image_path = tmp_path / "chart.png"
        image_path.write_bytes(VALID_PNG_BYTES)

        writer = PDFWriter(page_size="letter", margin_inches=1.0)
        content: DocumentContent = [
            {"type": "heading", "text": "Research Report", "level": 1},
            {
                "type": "paragraph",
                "text": "Introduction paragraph.",
                "bold": False,
                "italic": False,
            },
            {"type": "heading", "text": "Methods", "level": 2},
            {
                "type": "paragraph",
                "text": "Description of methods.",
                "bold": False,
                "italic": False,
            },
            {"type": "list", "items": ["Step 1", "Step 2", "Step 3"], "ordered": True},
            {"type": "heading", "text": "Results", "level": 2},
            {
                "type": "table",
                "headers": ["Sample", "Result"],
                "rows": [{"Sample": "A", "Result": 1.5}, {"Sample": "B", "Result": 2.3}],
                "caption": "Table 1: Results summary",
            },
            {
                "type": "figure",
                "path": image_path,
                "caption": "Figure 1: Data visualization",
                "width_inches": 4.0,
            },
            {"type": "page_break"},
            {"type": "heading", "text": "Discussion", "level": 2},
            {
                "type": "paragraph",
                "text": "Interpretation of results.",
                "bold": False,
                "italic": False,
            },
        ]
        out_path = tmp_path / "full_report.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_write_document_heading_level_clamped(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {"type": "heading", "text": "Level 0 clamped", "level": 0},
            {"type": "heading", "text": "Level 10 clamped", "level": 10},
        ]
        out_path = tmp_path / "clamped_headings.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()

    def test_write_document_table_with_none_values(self, tmp_path: Path) -> None:
        writer = PDFWriter()
        content: DocumentContent = [
            {
                "type": "table",
                "headers": ["A", "B"],
                "rows": [{"A": 1, "B": None}, {"A": None, "B": 2}],
                "caption": "",
            },
        ]
        out_path = tmp_path / "table_none.pdf"
        writer.write_document(content, out_path)
        assert out_path.exists()
