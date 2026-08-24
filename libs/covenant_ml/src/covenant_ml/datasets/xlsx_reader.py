"""Minimal deterministic XLSX sheet reader on the standard library.

Reads one worksheet of an ``.xlsx`` workbook into rows of verbatim cell
strings — numbers exactly as the file stores them ("1.697" stays
"1.697"), shared and inline strings resolved, empty cells as ``""``.
Built for corpus builders that must reproduce byte-identical output from
a sha256-pinned workbook: no float round-trips, no date coercion, no
third-party dependency.

Supported cell types: shared strings (``t="s"``), inline strings
(``t="inlineStr"``), formula string results (``t="str"``), and untyped
value cells (numbers/booleans, returned verbatim). Anything the reader
does not understand is a refusal, not a guess.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_index(cell_ref: str) -> int:
    """Convert a cell reference's column letters to a zero-based index.

    Args:
        cell_ref: A1-style cell reference (``"D5"``).

    Returns:
        Zero-based column index (``"A"`` is 0).

    Raises:
        ValueError: If the reference has no leading column letters.
    """
    index = 0
    letters = 0
    for char in cell_ref:
        if not char.isalpha():
            break
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
        letters += 1
    if letters == 0:
        raise ValueError(f"cell reference '{cell_ref}' has no column letters")
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Read the workbook's shared-string table.

    Each ``<si>`` entry's text is the concatenation of every ``<t>``
    beneath it, which flattens rich-text runs.

    Args:
        archive: The open workbook archive.

    Returns:
        Shared strings by index; empty when the part is absent.
    """
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for entry in root.findall(f"{{{_MAIN_NS}}}si"):
        strings.append("".join(t.text or "" for t in entry.iter(f"{{{_MAIN_NS}}}t")))
    return strings


def _sheet_part_name(archive: zipfile.ZipFile, sheet_name: str) -> str:
    """Resolve a worksheet's archive part through the workbook relationships.

    Args:
        archive: The open workbook archive.
        sheet_name: The worksheet's display name.

    Returns:
        The archive part path of the worksheet XML.

    Raises:
        ValueError: If the sheet or its relationship target is missing.
    """
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    rel_id = ""
    available: list[str] = []
    for sheet in workbook.iter(f"{{{_MAIN_NS}}}sheet"):
        name = sheet.get("name", "")
        available.append(name)
        if name == sheet_name:
            rel_id = sheet.get(_REL_ATTR, "")
    if rel_id == "":
        raise ValueError(f"sheet '{sheet_name}' not found; workbook has: {', '.join(available)}")

    rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for rel in rels.findall(f"{{{_PKG_REL_NS}}}Relationship"):
        if rel.get("Id", "") == rel_id:
            return "xl/" + rel.get("Target", "")
    raise ValueError(f"sheet '{sheet_name}' relationship '{rel_id}' has no target")


def _cell_text(cell: ElementTree.Element, strings: list[str]) -> str:
    """Extract one cell's value as a verbatim string.

    Args:
        cell: The ``<c>`` element.
        strings: The shared-string table.

    Returns:
        The cell text; ``""`` for a valueless cell.

    Raises:
        ValueError: If the cell type is unsupported or a shared-string
            index is out of range.
    """
    cell_type = cell.get("t", "")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(f"{{{_MAIN_NS}}}t"))
    value = cell.find(f"{{{_MAIN_NS}}}v")
    if value is None:
        return ""
    text = value.text or ""
    if cell_type == "s":
        index = int(text)
        if not 0 <= index < len(strings):
            raise ValueError(f"shared string index {index} out of range ({len(strings)} strings)")
        return strings[index]
    if cell_type in ("", "n", "str", "b"):
        return text
    raise ValueError(f"unsupported cell type '{cell_type}'")


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[list[str]]:
    """Read one worksheet into dense rows of verbatim cell strings.

    Rows are padded to a common width; cells without a stored value are
    ``""``. Cells that omit the ``r`` reference attribute fill the next
    column position, per the OOXML sequential-cell semantic.

    Args:
        path: Path to the ``.xlsx`` workbook.
        sheet_name: The worksheet's display name.

    Returns:
        The sheet's rows, in file order.

    Raises:
        ValueError: If the sheet is missing or a cell cannot be read.
    """
    with zipfile.ZipFile(path) as archive:
        strings = _shared_strings(archive)
        part = _sheet_part_name(archive, sheet_name)
        root = ElementTree.fromstring(archive.read(part))

    rows: list[dict[int, str]] = []
    width = 0
    for row in root.iter(f"{{{_MAIN_NS}}}row"):
        cells: dict[int, str] = {}
        cursor = 0
        for cell in row.findall(f"{{{_MAIN_NS}}}c"):
            ref = cell.get("r")
            column = _column_index(ref) if ref is not None else cursor
            cells[column] = _cell_text(cell, strings)
            cursor = column + 1
        rows.append(cells)
        if cells:
            width = max(width, max(cells) + 1)

    return [[cells.get(i, "") for i in range(width)] for cells in rows]


__all__ = ["read_xlsx_sheet"]
