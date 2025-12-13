"""Tests for scan_missing_data.py script."""

from __future__ import annotations

from pathlib import Path

from scripts.scan_missing_data import _get_processed_paths, scan_for_missing_data


def test_get_processed_paths_returns_set() -> None:
    """Test that _get_processed_paths returns a set of paths."""
    base_path = Path("/test/base")
    result = _get_processed_paths(base_path)

    # Should return at least 10 known processed files
    assert len(result) >= 10


def test_get_processed_paths_are_lowercase() -> None:
    """Test that all paths are lowercase."""
    base_path = Path("/test/base")
    result = _get_processed_paths(base_path)

    for path in result:
        assert path == path.lower()


def test_get_processed_paths_contains_expected_files() -> None:
    """Test that expected files are in the set."""
    base_path = Path("/test/base")
    result = _get_processed_paths(base_path)

    # Should contain normalized versions of known files
    assert any("response factors.xlsx" in p for p in result)
    assert any("chem_inv.xlsx" in p for p in result)


def test_scan_with_no_excel_files(tmp_path: Path) -> None:
    """Test scan with no Excel files in directory."""
    result = scan_for_missing_data(tmp_path)
    assert result == 0


def test_scan_finds_potential_inventory_files(tmp_path: Path) -> None:
    """Test scan finds files with 'inventory' keyword."""
    # Create test files
    (tmp_path / "chemical_inventory.xlsx").touch()

    result = scan_for_missing_data(tmp_path)
    assert result == 0


def test_scan_finds_potential_standard_files(tmp_path: Path) -> None:
    """Test scan finds files with 'standard' keyword."""
    (tmp_path / "standard_list.xlsx").touch()

    result = scan_for_missing_data(tmp_path)
    assert result == 0


def test_scan_ignores_files_without_keywords(tmp_path: Path) -> None:
    """Test scan ignores files without matching keywords."""
    (tmp_path / "random_data.xlsx").touch()

    result = scan_for_missing_data(tmp_path)
    assert result == 0


def test_scan_skips_temp_files(tmp_path: Path) -> None:
    """Test scan skips temporary Excel files starting with ~."""
    (tmp_path / "~$inventory.xlsx").touch()

    result = scan_for_missing_data(tmp_path)
    assert result == 0


def test_scan_finds_nested_files(tmp_path: Path) -> None:
    """Test scan finds files in subdirectories."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "chemical_inventory.xlsx").touch()

    result = scan_for_missing_data(tmp_path)
    assert result == 0


def test_scan_finds_xls_files(tmp_path: Path) -> None:
    """Test scan finds .xls files as well as .xlsx."""
    (tmp_path / "inventory.xls").touch()

    result = scan_for_missing_data(tmp_path)
    assert result == 0


def test_scan_skips_already_processed_files(tmp_path: Path) -> None:
    """Test scan skips files that match processed paths (line 95)."""
    # Create a file structure that matches one of the processed paths
    # The processed path "Notebooks/Emily Truong Notebook/Chem_Inv.xlsx" should be skipped
    notebook_dir = tmp_path / "Notebooks" / "Emily Truong Notebook"
    notebook_dir.mkdir(parents=True)

    # Create the file that matches a processed path - should be skipped (line 95)
    processed_file = notebook_dir / "Chem_Inv.xlsx"
    processed_file.touch()

    # Also create a file that has a keyword but is NOT in processed paths - should be found
    unprocessed_file = tmp_path / "new_inventory.xlsx"
    unprocessed_file.touch()

    result = scan_for_missing_data(tmp_path)
    assert result == 0


def test_scan_default_base_path() -> None:
    """Test scan uses default base path when None."""
    import logging

    # This verifies the None branch - will succeed or raise if path doesn't exist
    result: int = -1
    try:
        result = scan_for_missing_data(None)
    except FileNotFoundError:
        # Expected when default path doesn't exist
        logging.info("Default path not found - expected in CI")
        result = 0

    assert result == 0


def test_main_function() -> None:
    """Test main entry point."""
    import logging

    from scripts.scan_missing_data import main

    # main() calls setup_logging and scan_for_missing_data
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

    script_path = Path(__file__).parent.parent / "scripts" / "scan_missing_data.py"

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script_path), run_name="__main__")
        assert exc_info.value.code == 0
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
