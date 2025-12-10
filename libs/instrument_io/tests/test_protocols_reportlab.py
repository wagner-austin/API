"""Tests for _protocols.reportlab module."""

from __future__ import annotations

from pathlib import Path

from instrument_io._protocols.reportlab import (
    TableStyleCommand4,
    TableStyleCommand5,
    _create_image,
    _create_list_flowable,
    _create_list_item,
    _create_page_break,
    _create_page_number_callback,
    _create_paragraph,
    _create_paragraph_style,
    _create_simple_doc_template,
    _create_spacer,
    _create_table,
    _create_table_style_from_commands4,
    _create_table_style_from_commands5,
    _create_table_style_mixed,
    _get_sample_stylesheet,
)


def test_get_sample_stylesheet() -> None:
    stylesheet = _get_sample_stylesheet()
    normal = stylesheet["Normal"]
    assert normal.name == "Normal"


def test_get_sample_stylesheet_heading_styles() -> None:
    stylesheet = _get_sample_stylesheet()
    h1 = stylesheet["Heading1"]
    h2 = stylesheet["Heading2"]
    h3 = stylesheet["Heading3"]
    assert h1.name == "Heading1"
    assert h2.name == "Heading2"
    assert h3.name == "Heading3"


def test_create_paragraph() -> None:
    stylesheet = _get_sample_stylesheet()
    style = stylesheet["Normal"]
    para = _create_paragraph("Test paragraph text", style)
    # Verify flowable can be wrapped (has geometry)
    width, height = para.wrap(400, 800)
    assert width > 0
    assert height > 0


def test_create_paragraph_with_markup() -> None:
    stylesheet = _get_sample_stylesheet()
    style = stylesheet["Normal"]
    para = _create_paragraph("<b>Bold</b> and <i>italic</i>", style)
    width, height = para.wrap(400, 800)
    assert width > 0
    assert height > 0


def test_create_table() -> None:
    data = [
        ["Header1", "Header2"],
        ["Cell1", "Cell2"],
    ]
    table = _create_table(data)
    # Verify table can be wrapped (has geometry)
    width, height = table.wrap(400, 800)
    assert width > 0
    assert height > 0


def test_create_table_with_style() -> None:
    data = [
        ["A", "B"],
        ["1", "2"],
    ]
    commands4: list[TableStyleCommand4] = [
        TableStyleCommand4(cmd="BACKGROUND", start=(0, 0), stop=(-1, 0), value="#CCCCCC"),
    ]
    commands5: list[TableStyleCommand5] = [
        TableStyleCommand5(cmd="GRID", start=(0, 0), stop=(-1, -1), value1=1, value2="#000000"),
    ]
    style = _create_table_style_mixed(commands4, commands5)
    table = _create_table(data, style=style)
    width, height = table.wrap(400, 800)
    assert width > 0
    assert height > 0


def test_create_table_style_from_commands4() -> None:
    commands: list[TableStyleCommand4] = [
        TableStyleCommand4(cmd="BACKGROUND", start=(0, 0), stop=(-1, 0), value="#CCCCCC"),
        TableStyleCommand4(cmd="TEXTCOLOR", start=(0, 0), stop=(-1, 0), value="#000000"),
        TableStyleCommand4(cmd="ALIGN", start=(0, 0), stop=(-1, -1), value="CENTER"),
        TableStyleCommand4(cmd="FONTNAME", start=(0, 0), stop=(-1, 0), value="Helvetica-Bold"),
        TableStyleCommand4(cmd="FONTSIZE", start=(0, 0), stop=(-1, 0), value=10),
    ]
    style = _create_table_style_from_commands4(commands)
    # TableStyle can be used with tables - verify by creating table with it
    data = [["A"], ["B"]]
    table = _create_table(data, style=style)
    width, _height = table.wrap(200, 400)
    assert width > 0


def test_create_table_style_mixed() -> None:
    commands4: list[TableStyleCommand4] = [
        TableStyleCommand4(cmd="BACKGROUND", start=(0, 0), stop=(-1, 0), value="#CCCCCC"),
        TableStyleCommand4(cmd="ALIGN", start=(0, 0), stop=(-1, -1), value="CENTER"),
    ]
    commands5: list[TableStyleCommand5] = [
        TableStyleCommand5(cmd="GRID", start=(0, 0), stop=(-1, -1), value1=1, value2="#000000"),
    ]
    style = _create_table_style_mixed(commands4, commands5)
    data = [["A", "B"], ["1", "2"]]
    table = _create_table(data, style=style)
    width, _height = table.wrap(200, 400)
    assert width > 0


def test_create_spacer() -> None:
    spacer = _create_spacer(72, 36)
    width, height = spacer.wrap(500, 500)
    assert width == 72
    assert height == 36


def test_create_page_break_in_document(tmp_path: Path) -> None:
    out_path = tmp_path / "page_break_test.pdf"
    pagesize = (612.0, 792.0)
    margins = (72.0, 72.0, 72.0, 72.0)
    doc, page_callback = _create_simple_doc_template(out_path, pagesize, margins)
    stylesheet = _get_sample_stylesheet()
    # Build document with page break between paragraphs
    flowables = [
        _create_paragraph("Page 1 content", stylesheet["Normal"]),
        _create_page_break(),
        _create_paragraph("Page 2 content", stylesheet["Normal"]),
    ]
    doc.build(flowables, onFirstPage=page_callback, onLaterPages=page_callback)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_create_list_item_in_list(tmp_path: Path) -> None:
    out_path = tmp_path / "list_item_test.pdf"
    pagesize = (612.0, 792.0)
    margins = (72.0, 72.0, 72.0, 72.0)
    doc, page_callback = _create_simple_doc_template(out_path, pagesize, margins)
    stylesheet = _get_sample_stylesheet()
    style = stylesheet["Normal"]
    # Create list items and use them in a list flowable
    items = [
        _create_list_item(_create_paragraph("First item", style)),
        _create_list_item(_create_paragraph("Second item", style)),
    ]
    list_flow = _create_list_flowable(items, ordered=False)
    flowables = [list_flow]
    doc.build(flowables, onFirstPage=page_callback, onLaterPages=page_callback)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_create_list_item_with_indent_in_list(tmp_path: Path) -> None:
    out_path = tmp_path / "list_item_indent_test.pdf"
    pagesize = (612.0, 792.0)
    margins = (72.0, 72.0, 72.0, 72.0)
    doc, page_callback = _create_simple_doc_template(out_path, pagesize, margins)
    stylesheet = _get_sample_stylesheet()
    style = stylesheet["Normal"]
    # Create list items with custom indent
    items = [
        _create_list_item(_create_paragraph("Indented item", style), left_indent=36),
    ]
    list_flow = _create_list_flowable(items, ordered=True)
    flowables = [list_flow]
    doc.build(flowables, onFirstPage=page_callback, onLaterPages=page_callback)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_create_list_flowable_bulleted(tmp_path: Path) -> None:
    out_path = tmp_path / "list_bulleted_test.pdf"
    pagesize = (612.0, 792.0)
    margins = (72.0, 72.0, 72.0, 72.0)
    doc, page_callback = _create_simple_doc_template(out_path, pagesize, margins)
    stylesheet = _get_sample_stylesheet()
    style = stylesheet["Normal"]
    items = [
        _create_list_item(_create_paragraph("Item 1", style)),
        _create_list_item(_create_paragraph("Item 2", style)),
    ]
    list_flow = _create_list_flowable(items, ordered=False)
    flowables = [list_flow]
    doc.build(flowables, onFirstPage=page_callback, onLaterPages=page_callback)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_create_list_flowable_numbered(tmp_path: Path) -> None:
    out_path = tmp_path / "list_numbered_test.pdf"
    pagesize = (612.0, 792.0)
    margins = (72.0, 72.0, 72.0, 72.0)
    doc, page_callback = _create_simple_doc_template(out_path, pagesize, margins)
    stylesheet = _get_sample_stylesheet()
    style = stylesheet["Normal"]
    items = [
        _create_list_item(_create_paragraph("First", style)),
        _create_list_item(_create_paragraph("Second", style)),
    ]
    list_flow = _create_list_flowable(items, ordered=True)
    flowables = [list_flow]
    doc.build(flowables, onFirstPage=page_callback, onLaterPages=page_callback)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_create_simple_doc_template(tmp_path: Path) -> None:
    out_path = tmp_path / "test.pdf"
    pagesize = (612.0, 792.0)
    margins = (72.0, 72.0, 72.0, 72.0)
    doc, page_callback = _create_simple_doc_template(out_path, pagesize, margins)
    # Verify doc has build method by building empty flowables
    stylesheet = _get_sample_stylesheet()
    flowables = [_create_paragraph("Test", stylesheet["Normal"])]
    doc.build(flowables, onFirstPage=page_callback, onLaterPages=page_callback)
    assert out_path.exists()


def test_create_simple_doc_template_and_build(tmp_path: Path) -> None:
    out_path = tmp_path / "test_build.pdf"
    pagesize = (612.0, 792.0)
    margins = (72.0, 72.0, 72.0, 72.0)
    doc, page_callback = _create_simple_doc_template(out_path, pagesize, margins)

    stylesheet = _get_sample_stylesheet()
    flowables = [
        _create_paragraph("Test Document", stylesheet["Heading1"]),
        _create_spacer(0, 12),
        _create_paragraph("This is a test paragraph.", stylesheet["Normal"]),
    ]
    doc.build(flowables, onFirstPage=page_callback, onLaterPages=page_callback)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_create_image_with_fixture(tmp_path: Path) -> None:
    # Create a minimal PNG file (1x1 pixel red image)
    png_data = (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        b"\x00\x00\x00\rIHDR"  # IHDR chunk
        b"\x00\x00\x00\x01"  # width = 1
        b"\x00\x00\x00\x01"  # height = 1
        b"\x08\x02"  # bit depth = 8, color type = 2 (RGB)
        b"\x00\x00\x00"  # compression, filter, interlace
        b"\x90wS\xde"  # CRC
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND chunk
    )
    image_path = tmp_path / "test.png"
    image_path.write_bytes(png_data)

    img = _create_image(image_path)
    # Image can be wrapped
    width, height = img.wrap(400, 800)
    assert width > 0
    assert height > 0


def test_create_image_with_width(tmp_path: Path) -> None:
    # Create a minimal PNG file
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01"
        b"\x00\x00\x00\x01"
        b"\x08\x02"
        b"\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    image_path = tmp_path / "test_width.png"
    image_path.write_bytes(png_data)

    img = _create_image(image_path, width=144.0)
    width, height = img.wrap(400, 800)
    assert width == 144.0
    assert height > 0


def test_create_table_style_from_commands5() -> None:
    commands: list[TableStyleCommand5] = [
        TableStyleCommand5(cmd="GRID", start=(0, 0), stop=(-1, -1), value1=1, value2="#000000"),
        TableStyleCommand5(cmd="BOX", start=(0, 0), stop=(-1, -1), value1=2, value2="#333333"),
    ]
    style = _create_table_style_from_commands5(commands)
    # TableStyle can be used with tables - verify by creating table with it
    data = [["A", "B"], ["1", "2"]]
    table = _create_table(data, style=style)
    width, _height = table.wrap(200, 400)
    assert width > 0


def test_create_paragraph_style_name_only() -> None:
    """Test creating a paragraph style with only the name (all defaults)."""
    style = _create_paragraph_style("MinimalStyle")
    assert style.name == "MinimalStyle"
    # Style inherits defaults from reportlab
    assert style.fontSize > 0


def test_create_paragraph_style_basic() -> None:
    """Test creating a basic paragraph style with minimal options."""
    style = _create_paragraph_style(
        "TestStyle",
        font_name="Times-Roman",
        font_size=12.0,
        leading=24.0,
    )
    assert style.name == "TestStyle"
    assert style.fontSize == 12.0
    assert style.leading == 24.0


def test_create_paragraph_style_with_all_options() -> None:
    """Test creating a paragraph style with all options specified."""
    base_style = _create_paragraph_style(
        "BaseStyle",
        font_name="Helvetica",
        font_size=10.0,
        leading=12.0,
    )
    style = _create_paragraph_style(
        "FullStyle",
        parent=base_style,
        font_name="Times-Bold",
        font_size=14.0,
        leading=28.0,
        alignment=1,  # CENTER
        first_line_indent=36.0,
        left_indent=18.0,
        right_indent=18.0,
        space_before=12.0,
        space_after=6.0,
    )
    assert style.name == "FullStyle"
    assert style.fontSize == 14.0
    assert style.leading == 28.0
    assert style.alignment == 1
    assert style.spaceAfter == 6.0
    assert style.spaceBefore == 12.0


def test_create_paragraph_style_in_document(tmp_path: Path) -> None:
    """Test using a custom paragraph style in a document."""
    out_path = tmp_path / "custom_style_test.pdf"
    pagesize = (612.0, 792.0)
    margins = (72.0, 72.0, 72.0, 72.0)
    doc, page_callback = _create_simple_doc_template(out_path, pagesize, margins)

    # Create MLA-style paragraph
    mla_style = _create_paragraph_style(
        "MLAStyle",
        font_name="Times-Roman",
        font_size=12.0,
        leading=24.0,  # Double-spaced
        alignment=0,  # Left
        first_line_indent=36.0,  # 0.5 inch indent
        space_before=0.0,
        space_after=0.0,
    )

    flowables = [
        _create_paragraph("This is a test paragraph in MLA style.", mla_style),
    ]
    doc.build(flowables, onFirstPage=page_callback, onLaterPages=page_callback)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_create_simple_doc_template_no_page_numbers(tmp_path: Path) -> None:
    """Test creating a document without page numbers."""
    out_path = tmp_path / "no_page_numbers.pdf"
    pagesize = (612.0, 792.0)
    margins = (72.0, 72.0, 72.0, 72.0)
    doc, page_callback = _create_simple_doc_template(
        out_path, pagesize, margins, show_page_numbers=False
    )
    assert page_callback is None
    stylesheet = _get_sample_stylesheet()
    flowables = [_create_paragraph("Content without page numbers", stylesheet["Normal"])]
    doc.build(flowables)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_create_page_number_callback(tmp_path: Path) -> None:
    """Test creating and using a page number callback directly."""
    pagesize = (612.0, 792.0)
    callback = _create_page_number_callback(pagesize)
    # Callback is callable
    assert callable(callback)
    # Use it in a document
    out_path = tmp_path / "page_number_callback.pdf"
    margins = (72.0, 72.0, 72.0, 72.0)
    doc, _ = _create_simple_doc_template(out_path, pagesize, margins, show_page_numbers=False)
    stylesheet = _get_sample_stylesheet()
    flowables = [_create_paragraph("Page with number callback", stylesheet["Normal"])]
    doc.build(flowables, onFirstPage=callback, onLaterPages=callback)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
