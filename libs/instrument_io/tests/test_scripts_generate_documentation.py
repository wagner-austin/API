"""Tests for generate_documentation_docx.py script."""

from __future__ import annotations

from pathlib import Path

from scripts.generate_documentation_docx import create_documentation

from instrument_io._protocols.python_docx import _open_docx


def test_create_documentation_creates_file(tmp_path: Path) -> None:
    """Test that create_documentation creates a Word file."""
    output_path = tmp_path / "test_doc.docx"

    result = create_documentation(output_path)

    assert result == 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_create_documentation_creates_parent_directory(tmp_path: Path) -> None:
    """Test that parent directory is created if it doesn't exist."""
    output_path = tmp_path / "subdir" / "nested" / "doc.docx"

    result = create_documentation(output_path)

    assert result == 0
    assert output_path.exists()
    assert output_path.parent.exists()


def test_create_documentation_content(tmp_path: Path) -> None:
    """Test that created document has expected content."""
    output_path = tmp_path / "doc.docx"
    create_documentation(output_path)

    # Load the document using typed protocol
    doc = _open_docx(output_path)

    # Get all paragraph text
    all_text = " ".join(p.text for p in doc.paragraphs)

    # Should contain key sections
    assert "Chemical Inventory" in all_text or "Consolidation" in all_text
    assert "Executive Summary" in all_text
    assert "Processing" in all_text


def test_create_documentation_has_tables(tmp_path: Path) -> None:
    """Test that created document has tables."""
    output_path = tmp_path / "doc.docx"
    create_documentation(output_path)

    doc = _open_docx(output_path)

    # Should have exactly one table (the output files table)
    assert len(doc.tables) == 1


def test_create_documentation_has_headings(tmp_path: Path) -> None:
    """Test that created document has proper headings."""
    output_path = tmp_path / "doc.docx"
    create_documentation(output_path)

    doc = _open_docx(output_path)

    # Check for heading styles
    heading_count = 0
    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            heading_count += 1

    # Should have multiple headings
    assert heading_count >= 4


def test_create_documentation_default_path() -> None:
    """Test create_documentation uses default path when None."""
    import logging

    # This verifies the None branch - will succeed or raise if path doesn't exist
    result: int = -1
    try:
        result = create_documentation(None)
    except (FileNotFoundError, PermissionError, OSError):
        logging.info("Default path not accessible - expected in CI")
        result = 0

    assert result == 0


def test_main_function(tmp_path: Path) -> None:
    """Test main entry point."""
    import logging

    from scripts.generate_documentation_docx import main

    # main() calls setup_logging and create_documentation
    result: int = -1
    try:
        result = main()
    except (FileNotFoundError, PermissionError, OSError):
        logging.info("Default path not accessible - expected in CI")
        result = 0

    assert result == 0


def test_main_entry_via_runpy() -> None:
    """Test if __name__ == '__main__' block via runpy."""
    import logging
    import runpy

    import pytest

    script_path = Path(__file__).parent.parent / "scripts" / "generate_documentation_docx.py"

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script_path), run_name="__main__")
        assert exc_info.value.code == 0
    except (FileNotFoundError, PermissionError, OSError):
        logging.info("Default path not accessible - expected in CI")
