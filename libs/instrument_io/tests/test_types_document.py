"""Tests for types.document module."""

from __future__ import annotations

from pathlib import Path

from instrument_io.types.document import (
    PAGE_SIZES,
    DocumentContent,
    DocumentSection,
    FigureContent,
    HeadingContent,
    ListContent,
    PageBreakContent,
    PageSize,
    ParagraphContent,
    TableContent,
    is_figure,
    is_heading,
    is_list,
    is_page_break,
    is_paragraph,
    is_table,
)


def test_heading_content_creation() -> None:
    heading: HeadingContent = {
        "type": "heading",
        "text": "Test Heading",
        "level": 1,
    }
    assert heading["type"] == "heading"
    assert heading["text"] == "Test Heading"
    assert heading["level"] == 1


def test_paragraph_content_creation() -> None:
    para: ParagraphContent = {
        "type": "paragraph",
        "text": "This is a test paragraph.",
        "bold": False,
        "italic": True,
    }
    assert para["type"] == "paragraph"
    assert para["text"] == "This is a test paragraph."
    assert para["bold"] is False
    assert para["italic"] is True


def test_table_content_creation() -> None:
    table: TableContent = {
        "type": "table",
        "headers": ["Name", "Value"],
        "rows": [{"Name": "Item1", "Value": 100}],
        "caption": "Test Table",
    }
    assert table["type"] == "table"
    assert table["headers"] == ["Name", "Value"]
    assert len(table["rows"]) == 1
    assert table["caption"] == "Test Table"


def test_figure_content_creation() -> None:
    figure: FigureContent = {
        "type": "figure",
        "path": Path("/test/image.png"),
        "caption": "Test Figure",
        "width_inches": 4.0,
    }
    assert figure["type"] == "figure"
    assert figure["path"] == Path("/test/image.png")
    assert figure["caption"] == "Test Figure"
    assert figure["width_inches"] == 4.0


def test_list_content_ordered() -> None:
    list_content: ListContent = {
        "type": "list",
        "items": ["First", "Second", "Third"],
        "ordered": True,
    }
    assert list_content["type"] == "list"
    assert list_content["items"] == ["First", "Second", "Third"]
    assert list_content["ordered"] is True


def test_list_content_unordered() -> None:
    list_content: ListContent = {
        "type": "list",
        "items": ["Bullet 1", "Bullet 2"],
        "ordered": False,
    }
    assert list_content["ordered"] is False


def test_page_break_content() -> None:
    page_break: PageBreakContent = {
        "type": "page_break",
    }
    assert page_break["type"] == "page_break"


def test_document_section_union() -> None:
    heading: DocumentSection = {
        "type": "heading",
        "text": "Heading",
        "level": 1,
    }
    para: DocumentSection = {
        "type": "paragraph",
        "text": "Paragraph",
        "bold": False,
        "italic": False,
    }
    assert heading["type"] == "heading"
    assert para["type"] == "paragraph"


def test_document_content_list() -> None:
    content: DocumentContent = [
        {"type": "heading", "text": "Title", "level": 1},
        {"type": "paragraph", "text": "Body text.", "bold": False, "italic": False},
    ]
    assert len(content) == 2
    assert content[0]["type"] == "heading"
    assert content[1]["type"] == "paragraph"


def test_page_sizes_letter() -> None:
    size: PageSize = "letter"
    dims = PAGE_SIZES[size]
    assert dims == (612.0, 792.0)


def test_page_sizes_a4() -> None:
    size: PageSize = "a4"
    dims = PAGE_SIZES[size]
    assert dims == (595.28, 841.89)


def test_page_sizes_legal() -> None:
    size: PageSize = "legal"
    dims = PAGE_SIZES[size]
    assert dims == (612.0, 1008.0)


def test_is_heading_true() -> None:
    section: DocumentSection = {"type": "heading", "text": "Test", "level": 1}
    assert is_heading(section) is True


def test_is_heading_false() -> None:
    section: DocumentSection = {"type": "paragraph", "text": "Test", "bold": False, "italic": False}
    assert is_heading(section) is False


def test_is_paragraph_true() -> None:
    section: DocumentSection = {"type": "paragraph", "text": "Test", "bold": False, "italic": False}
    assert is_paragraph(section) is True


def test_is_paragraph_false() -> None:
    section: DocumentSection = {"type": "heading", "text": "Test", "level": 1}
    assert is_paragraph(section) is False


def test_is_table_true() -> None:
    section: DocumentSection = {"type": "table", "headers": [], "rows": [], "caption": ""}
    assert is_table(section) is True


def test_is_table_false() -> None:
    section: DocumentSection = {"type": "heading", "text": "Test", "level": 1}
    assert is_table(section) is False


def test_is_figure_true() -> None:
    section: DocumentSection = {
        "type": "figure",
        "path": Path("/test.png"),
        "caption": "",
        "width_inches": 0.0,
    }
    assert is_figure(section) is True


def test_is_figure_false() -> None:
    section: DocumentSection = {"type": "heading", "text": "Test", "level": 1}
    assert is_figure(section) is False


def test_is_list_true() -> None:
    section: DocumentSection = {"type": "list", "items": [], "ordered": False}
    assert is_list(section) is True


def test_is_list_false() -> None:
    section: DocumentSection = {"type": "heading", "text": "Test", "level": 1}
    assert is_list(section) is False


def test_is_page_break_true() -> None:
    section: DocumentSection = {"type": "page_break"}
    assert is_page_break(section) is True


def test_is_page_break_false() -> None:
    section: DocumentSection = {"type": "heading", "text": "Test", "level": 1}
    assert is_page_break(section) is False
