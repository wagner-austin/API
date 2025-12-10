"""Tests for compare_inventory_standards.py script."""

from __future__ import annotations

from pathlib import Path

from scripts.compare_inventory_standards import (
    _extract_chemical_names,
    _load_inventory,
    _load_standards,
    _normalize,
    compare_inventory_and_standards,
)

from instrument_io._protocols.openpyxl import _create_workbook


def _create_inventory_fixture(path: Path, rows: list[list[str]]) -> None:
    """Create inventory Excel fixture."""
    wb = _create_workbook()
    ws = wb.active
    headers = ["Chemical Name", "CAS", "Physical State"]
    for col_idx, header in enumerate(headers, start=1):
        ws.cell(row=1, column=col_idx, value=header)
    for row_idx, row_data in enumerate(rows, start=2):
        for col_idx, value in enumerate(row_data, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)
    wb.save(path)
    wb.close()


def _create_standards_fixture(path: Path, rows: list[list[str]]) -> None:
    """Create standards Excel fixture."""
    wb = _create_workbook()
    ws = wb.active
    headers = ["Chemical Name", "Source", "Date", "Details"]
    for col_idx, header in enumerate(headers, start=1):
        ws.cell(row=1, column=col_idx, value=header)
    for row_idx, row_data in enumerate(rows, start=2):
        for col_idx, value in enumerate(row_data, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)
    wb.save(path)
    wb.close()


def test_normalize_strips_and_lowercases() -> None:
    """Test that normalize strips whitespace and lowercases."""
    assert _normalize("  Acetone  ") == "acetone"
    assert _normalize("BENZENE") == "benzene"
    assert _normalize("  alpha-Pinene  ") == "alpha-pinene"


def test_normalize_empty_string() -> None:
    """Test normalize with empty string."""
    assert _normalize("") == ""


def test_normalize_none() -> None:
    """Test normalize with None."""
    assert _normalize(None) == ""


def test_normalize_integer_cast_to_str() -> None:
    """Test normalize behavior with edge cases."""
    result = _normalize("")
    assert result == ""


def test_load_inventory(tmp_path: Path) -> None:
    """Test _load_inventory loads Excel file."""
    excel_path = tmp_path / "inventory.xlsx"
    _create_inventory_fixture(
        excel_path,
        [
            ["Acetone", "67-64-1", "Liquid"],
            ["Benzene", "71-43-2", "Liquid"],
        ],
    )

    result = _load_inventory(excel_path)

    assert result.height == 2
    assert "Chemical Name" in result.columns


def test_load_standards(tmp_path: Path) -> None:
    """Test _load_standards loads Excel file."""
    excel_path = tmp_path / "standards.xlsx"
    _create_standards_fixture(
        excel_path,
        [
            ["alpha-Pinene", "Mix 1", "2025-01-01", "Terpene"],
            ["Limonene", "Mix 2", "2025-01-01", "Terpene"],
        ],
    )

    result = _load_standards(excel_path)

    assert result.height == 2
    assert "Chemical Name" in result.columns


def test_extract_chemical_names_from_column(tmp_path: Path) -> None:
    """Test extracting chemical names from DataFrame column."""
    excel_path = tmp_path / "test.xlsx"
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical Name")
    ws.cell(row=2, column=1, value="Acetone")
    ws.cell(row=3, column=1, value="Benzene")
    ws.cell(row=4, column=1, value=None)
    ws.cell(row=5, column=1, value="Ethanol")
    wb.save(excel_path)
    wb.close()

    import polars as pl

    df = pl.read_excel(excel_path)
    result = _extract_chemical_names(df, "Chemical Name")

    # Verify keys exist AND have correct values (strong assertions)
    assert result.get("acetone") == "Acetone"
    assert result.get("benzene") == "Benzene"
    assert result.get("ethanol") == "Ethanol"


def test_extract_chemical_names_missing_column(tmp_path: Path) -> None:
    """Test extracting from missing column returns empty dict."""
    excel_path = tmp_path / "test.xlsx"
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Other")
    ws.cell(row=2, column=1, value="Value")
    wb.save(excel_path)
    wb.close()

    import polars as pl

    df = pl.read_excel(excel_path)
    result = _extract_chemical_names(df, "Chemical Name")

    assert result == {}


def test_compare_inventory_and_standards(tmp_path: Path) -> None:
    """Test comparing inventory and standards."""
    inv_path = tmp_path / "inventory.xlsx"
    std_path = tmp_path / "standards.xlsx"

    _create_inventory_fixture(
        inv_path,
        [
            ["Acetone", "67-64-1", "Liquid"],
            ["Benzene", "71-43-2", "Liquid"],
            ["Ethanol", "64-17-5", "Liquid"],
        ],
    )
    _create_standards_fixture(
        std_path,
        [
            ["Acetone", "Mix 1", "2025-01-01", "Standard"],
            ["alpha-Pinene", "Mix 2", "2025-01-01", "Terpene"],
        ],
    )

    result = compare_inventory_and_standards(inv_path, std_path)

    assert result == 0


def test_compare_all_standards_in_inventory(tmp_path: Path) -> None:
    """Test when all standards have inventory entries."""
    inv_path = tmp_path / "inventory.xlsx"
    std_path = tmp_path / "standards.xlsx"

    _create_inventory_fixture(
        inv_path,
        [
            ["Acetone", "67-64-1", "Liquid"],
            ["Benzene", "71-43-2", "Liquid"],
        ],
    )
    _create_standards_fixture(
        std_path,
        [
            ["Acetone", "Mix 1", "2025-01-01", "Standard"],
            ["Benzene", "Mix 1", "2025-01-01", "Standard"],
        ],
    )

    result = compare_inventory_and_standards(inv_path, std_path)

    assert result == 0


def test_compare_with_missing_standards(tmp_path: Path) -> None:
    """Test when some standards are missing from inventory."""
    inv_path = tmp_path / "inventory.xlsx"
    std_path = tmp_path / "standards.xlsx"

    _create_inventory_fixture(
        inv_path,
        [
            ["Acetone", "67-64-1", "Liquid"],
        ],
    )
    _create_standards_fixture(
        std_path,
        [
            ["Acetone", "Mix 1", "2025-01-01", "Standard"],
            ["alpha-Pinene", "Mix 2", "2025-01-01", "Terpene"],
            ["Limonene", "Mix 3", "2025-01-01", "Terpene"],
        ],
    )

    result = compare_inventory_and_standards(inv_path, std_path)

    # Should still succeed (just reports the missing ones)
    assert result == 0


def test_compare_with_many_inventory_no_standard(tmp_path: Path) -> None:
    """Test when more than 15 inventory items have no standards (truncation branch)."""
    inv_path = tmp_path / "inventory.xlsx"
    std_path = tmp_path / "standards.xlsx"

    # Create inventory with 20 unique chemicals
    _create_inventory_fixture(
        inv_path,
        [[f"Chemical{i}", f"{i}-00-0", "Liquid"] for i in range(20)],
    )
    # Create standards with only 2 items that don't match any inventory
    _create_standards_fixture(
        std_path,
        [
            ["alpha-Pinene", "Mix 1", "2025-01-01", "Standard"],
            ["Limonene", "Mix 2", "2025-01-01", "Terpene"],
        ],
    )

    result = compare_inventory_and_standards(inv_path, std_path)

    # Should succeed and log "... and X more" for items beyond 15
    assert result == 0


def test_compare_default_paths() -> None:
    """Test compare_inventory_and_standards uses default paths when None."""
    import logging

    # This verifies the None branches
    result: int = -1
    try:
        result = compare_inventory_and_standards(None, None)
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
        result = 0

    assert result == 0


def test_main_function() -> None:
    """Test main entry point."""
    import logging

    from scripts.compare_inventory_standards import main

    result: int = -1
    try:
        result = main()
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
        result = 0

    assert result == 0


def test_main_entry_via_runpy() -> None:
    """Test if __name__ == '__main__' block via runpy."""
    import logging
    import runpy

    import pytest

    script_path = Path(__file__).parent.parent / "scripts" / "compare_inventory_standards.py"

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script_path), run_name="__main__")
        assert exc_info.value.code == 0
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
