"""Tests for the minimal deterministic XLSX sheet reader.

Fixtures are hand-built OOXML archives written with ``zipfile``, so
every supported cell shape (shared string, rich-text run, inline
string, formula string, boolean, number, valueless, unreferenced) and
every refusal is exercised against real workbook bytes.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from covenant_ml.datasets.xlsx_reader import read_xlsx_sheet

_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def write_workbook(
    path: Path,
    sheets: dict[str, str],
    shared_strings_xml: str | None = None,
) -> None:
    """Write a minimal ``.xlsx`` archive.

    Args:
        path: Destination file path.
        sheets: Worksheet XML bodies keyed by sheet display name.
        shared_strings_xml: Optional ``sharedStrings.xml`` content.
    """
    names = list(sheets)
    sheet_tags = "".join(
        f'<sheet name="{name}" sheetId="{i + 1}" r:id="rId{i + 1}"/>'
        for i, name in enumerate(names)
    )
    workbook_xml = (
        f'<workbook xmlns="{_MAIN}" xmlns:r="{_REL}"><sheets>{sheet_tags}</sheets></workbook>'
    )
    rel_tags = "".join(
        f'<Relationship Id="rId{i + 1}" Type="{_REL}/worksheet" '
        f'Target="worksheets/sheet{i + 1}.xml"/>'
        for i in range(len(names))
    )
    rels_xml = f'<Relationships xmlns="{_PKG}">{rel_tags}</Relationships>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        if shared_strings_xml is not None:
            archive.writestr("xl/sharedStrings.xml", shared_strings_xml)
        for i, name in enumerate(names):
            archive.writestr(f"xl/worksheets/sheet{i + 1}.xml", sheets[name])


def sheet_of_inline_rows(rows: list[list[str]]) -> str:
    """Render rows of verbatim strings as a worksheet of inline-string cells.

    Cells omit the ``r`` reference attribute, exercising the sequential
    cell semantic on every read.

    Args:
        rows: Cell strings by row.

    Returns:
        The worksheet XML.
    """
    row_tags = []
    for row in rows:
        cells = "".join(f'<c t="inlineStr"><is><t>{cell}</t></is></c>' for cell in row)
        row_tags.append(f"<row>{cells}</row>")
    return f'<worksheet xmlns="{_MAIN}"><sheetData>{"".join(row_tags)}</sheetData></worksheet>'


class TestReadXlsxSheet:
    """Cell-type coverage, padding, and ordering."""

    def test_every_supported_cell_shape(self, tmp_path: Path) -> None:
        """Shared, rich-text, inline, formula, boolean, number, empty."""
        path = tmp_path / "book.xlsx"
        shared = (
            f'<sst xmlns="{_MAIN}"><si><t>plain</t></si>'
            "<si><r><t>rich</t></r><r><t>-text</t></r></si></sst>"
        )
        sheet = (
            f'<worksheet xmlns="{_MAIN}"><sheetData>'
            '<row r="1">'
            '<c r="A1" t="s"><v>0</v></c>'
            '<c r="B1" t="s"><v>1</v></c>'
            '<c r="C1" t="inlineStr"><is><t>inline</t></is></c>'
            '<c r="D1" t="str"><v>=SUM</v></c>'
            "</row>"
            '<row r="2">'
            '<c r="A2"><v>1.697</v></c>'
            '<c r="B2" t="n"><v>90</v></c>'
            '<c r="C2" t="b"><v>1</v></c>'
            '<c r="D2"/>'
            "</row>"
            "</sheetData></worksheet>"
        )
        write_workbook(path, {"Data": sheet}, shared)
        assert read_xlsx_sheet(path, "Data") == [
            ["plain", "rich-text", "inline", "=SUM"],
            ["1.697", "90", "1", ""],
        ]

    def test_sparse_rows_pad_and_place_by_reference(self, tmp_path: Path) -> None:
        """A row naming only C3 still lands in column index 2."""
        path = tmp_path / "book.xlsx"
        sheet = (
            f'<worksheet xmlns="{_MAIN}"><sheetData>'
            '<row r="1"><c r="A1"><v>1</v></c><c r="C1"><v>3</v></c></row>'
            '<row r="2"><c r="B2"><v>2</v></c></row>'
            "</sheetData></worksheet>"
        )
        write_workbook(path, {"Data": sheet})
        assert read_xlsx_sheet(path, "Data") == [["1", "", "3"], ["", "2", ""]]

    def test_all_letter_reference_and_cellless_row(self, tmp_path: Path) -> None:
        """A digitless ref still names its column; an empty row pads out."""
        path = tmp_path / "book.xlsx"
        sheet = (
            f'<worksheet xmlns="{_MAIN}"><sheetData>'
            '<row r="1"><c r="AB"><v>9</v></c></row>'
            '<row r="2"/>'
            "</sheetData></worksheet>"
        )
        write_workbook(path, {"Data": sheet})
        assert read_xlsx_sheet(path, "Data") == [[""] * 27 + ["9"], [""] * 28]

    def test_unreferenced_cells_fill_sequentially(self, tmp_path: Path) -> None:
        """Cells without an r attribute take the next column position."""
        path = tmp_path / "book.xlsx"
        write_workbook(path, {"Data": sheet_of_inline_rows([["a", "b"], ["c", ""]])})
        assert read_xlsx_sheet(path, "Data") == [["a", "b"], ["c", ""]]

    def test_second_sheet_resolves_through_relationships(self, tmp_path: Path) -> None:
        """Sheet names map to parts via rIds, not declaration order guesses."""
        path = tmp_path / "book.xlsx"
        write_workbook(
            path,
            {
                "First": sheet_of_inline_rows([["one"]]),
                "Second": sheet_of_inline_rows([["two"]]),
            },
        )
        assert read_xlsx_sheet(path, "Second") == [["two"]]

    def test_workbook_without_shared_strings(self, tmp_path: Path) -> None:
        """A numbers-only workbook has no sharedStrings part."""
        path = tmp_path / "book.xlsx"
        sheet = (
            f'<worksheet xmlns="{_MAIN}"><sheetData>'
            '<row r="1"><c r="A1"><v>7</v></c></row>'
            "</sheetData></worksheet>"
        )
        write_workbook(path, {"Data": sheet})
        assert read_xlsx_sheet(path, "Data") == [["7"]]

    def test_empty_sheet_returns_no_rows(self, tmp_path: Path) -> None:
        """A sheet with no row elements reads as an empty list."""
        path = tmp_path / "book.xlsx"
        write_workbook(path, {"Data": f'<worksheet xmlns="{_MAIN}"><sheetData/></worksheet>'})
        assert read_xlsx_sheet(path, "Data") == []


class TestRefusals:
    """Structural defects refuse by name."""

    def test_unknown_sheet_lists_available(self, tmp_path: Path) -> None:
        """The refusal names every sheet the workbook does have."""
        path = tmp_path / "book.xlsx"
        write_workbook(path, {"Alpha": sheet_of_inline_rows([["x"]])})
        with pytest.raises(ValueError, match="sheet 'Beta' not found; workbook has: Alpha"):
            read_xlsx_sheet(path, "Beta")

    def test_missing_relationship_target(self, tmp_path: Path) -> None:
        """A sheet whose rId has no relationship entry is a defect."""
        path = tmp_path / "book.xlsx"
        workbook_xml = (
            f'<workbook xmlns="{_MAIN}" xmlns:r="{_REL}"><sheets>'
            f'<sheet name="Data" sheetId="1" r:id="rId9"/></sheets></workbook>'
        )
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("xl/workbook.xml", workbook_xml)
            archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{_PKG}"/>')
        with pytest.raises(ValueError, match="relationship 'rId9' has no target"):
            read_xlsx_sheet(path, "Data")

    def test_shared_string_index_out_of_range(self, tmp_path: Path) -> None:
        """A dangling shared-string index is a defect, not empty text."""
        path = tmp_path / "book.xlsx"
        shared = f'<sst xmlns="{_MAIN}"><si><t>only</t></si></sst>'
        sheet = (
            f'<worksheet xmlns="{_MAIN}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>5</v></c></row>'
            "</sheetData></worksheet>"
        )
        write_workbook(path, {"Data": sheet}, shared)
        with pytest.raises(ValueError, match=r"shared string index 5 out of range \(1 strings\)"):
            read_xlsx_sheet(path, "Data")

    def test_unsupported_cell_type(self, tmp_path: Path) -> None:
        """An error-typed cell is refused, never silently blanked."""
        path = tmp_path / "book.xlsx"
        sheet = (
            f'<worksheet xmlns="{_MAIN}"><sheetData>'
            '<row r="1"><c r="A1" t="e"><v>#DIV/0!</v></c></row>'
            "</sheetData></worksheet>"
        )
        write_workbook(path, {"Data": sheet})
        with pytest.raises(ValueError, match="unsupported cell type 'e'"):
            read_xlsx_sheet(path, "Data")

    def test_reference_without_column_letters(self, tmp_path: Path) -> None:
        """A cell reference of bare digits cannot name a column."""
        path = tmp_path / "book.xlsx"
        sheet = (
            f'<worksheet xmlns="{_MAIN}"><sheetData>'
            '<row r="1"><c r="11"><v>7</v></c></row>'
            "</sheetData></worksheet>"
        )
        write_workbook(path, {"Data": sheet})
        with pytest.raises(ValueError, match="cell reference '11' has no column letters"):
            read_xlsx_sheet(path, "Data")
