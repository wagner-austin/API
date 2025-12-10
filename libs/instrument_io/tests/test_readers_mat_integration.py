"""Integration tests for MAT reader using actual files."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import MATReadError
from instrument_io.readers.mat import MATReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_MAT = FIXTURES_DIR / "sample.mat"


def test_list_variables() -> None:
    """Test listing variables from actual MAT file."""
    reader = MATReader()
    variables = reader.list_variables(SAMPLE_MAT)

    assert "scalar_value" in variables
    assert "vector" in variables
    assert "matrix" in variables
    assert "text" in variables
    # Metadata variables should be excluded
    assert not any(v.startswith("__") for v in variables)


def test_read_variable_scalar() -> None:
    """Test reading scalar variable from actual MAT file."""
    reader = MATReader()
    value = reader.read_variable(SAMPLE_MAT, "scalar_value")

    # MATLAB stores scalars as 1x1 arrays
    assert value == [[42.0]]


def test_read_variable_vector() -> None:
    """Test reading vector variable from actual MAT file."""
    reader = MATReader()
    value = reader.read_variable(SAMPLE_MAT, "vector")

    # MATLAB stores vectors as row vectors (1xN arrays)
    assert value == [[1, 2, 3, 4, 5]]


def test_read_variable_matrix() -> None:
    """Test reading matrix variable from actual MAT file."""
    reader = MATReader()
    value = reader.read_variable(SAMPLE_MAT, "matrix")

    # Verify matrix structure
    assert value == [[1.1, 2.2], [3.3, 4.4]]


def test_read_variable_text() -> None:
    """Test reading text variable from actual MAT file."""
    reader = MATReader()
    value = reader.read_variable(SAMPLE_MAT, "text")

    # MATLAB stores strings as character arrays
    assert value == ["hello"]


def test_read_all() -> None:
    """Test reading all variables from actual MAT file."""
    reader = MATReader()
    data = reader.read_all(SAMPLE_MAT)

    # MATLAB stores scalars as 1x1 arrays, vectors as row vectors
    # Verify all expected keys exist by checking their actual values
    assert data["scalar_value"] == [[42.0]]
    assert data["vector"] == [[1, 2, 3, 4, 5]]
    assert data["text"] == ["hello"]
    # Verify matrix has expected value (2x2 matrix)
    assert data["matrix"] == [[1.1, 2.2], [3.3, 4.4]]


def test_get_metadata() -> None:
    """Test getting metadata from actual MAT file."""
    reader = MATReader()
    metadata = reader.get_metadata(SAMPLE_MAT)

    # Should have MATLAB metadata keys (may or may not be present)
    # Just verify it returns a dict-like structure
    assert "__header__" in metadata or metadata == {}


def test_supports_format_mat() -> None:
    """Test format detection for .mat files."""
    reader = MATReader()
    assert reader.supports_format(SAMPLE_MAT) is True


def test_supports_format_non_mat() -> None:
    """Test format detection for non-.mat files."""
    reader = MATReader()
    non_mat = FIXTURES_DIR / "sample.txt"
    assert reader.supports_format(non_mat) is False


def test_read_variable_file_not_exists() -> None:
    """Test error when file doesn't exist."""
    reader = MATReader()
    nonexistent = FIXTURES_DIR / "nonexistent.mat"

    with pytest.raises(MATReadError) as exc_info:
        reader.read_variable(nonexistent, "some_var")

    assert "File does not exist" in str(exc_info.value)


def test_read_variable_not_found() -> None:
    """Test error when variable doesn't exist."""
    reader = MATReader()

    with pytest.raises(MATReadError) as exc_info:
        reader.read_variable(SAMPLE_MAT, "nonexistent_variable")

    assert "Variable 'nonexistent_variable' not found" in str(exc_info.value)


def test_read_variable_not_mat_file() -> None:
    """Test error when file is not a .mat file."""
    reader = MATReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(MATReadError) as exc_info:
        reader.read_variable(txt_file, "some_var")

    assert "Not a MATLAB data file" in str(exc_info.value)


def test_read_all_file_not_exists() -> None:
    """Test error when file doesn't exist for read_all."""
    reader = MATReader()
    nonexistent = FIXTURES_DIR / "nonexistent.mat"

    with pytest.raises(MATReadError) as exc_info:
        reader.read_all(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_all_not_mat_file() -> None:
    """Test error when file is not a .mat file for read_all."""
    reader = MATReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(MATReadError) as exc_info:
        reader.read_all(txt_file)

    assert "Not a MATLAB data file" in str(exc_info.value)


def test_get_metadata_file_not_exists() -> None:
    """Test error when file doesn't exist for get_metadata."""
    reader = MATReader()
    nonexistent = FIXTURES_DIR / "nonexistent.mat"

    with pytest.raises(MATReadError) as exc_info:
        reader.get_metadata(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_get_metadata_not_mat_file() -> None:
    """Test error when file is not a .mat file for get_metadata."""
    reader = MATReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(MATReadError) as exc_info:
        reader.get_metadata(txt_file)

    assert "Not a MATLAB data file" in str(exc_info.value)


def test_list_variables_file_not_exists() -> None:
    """Test error when file doesn't exist for list_variables."""
    reader = MATReader()
    nonexistent = FIXTURES_DIR / "nonexistent.mat"

    with pytest.raises(MATReadError) as exc_info:
        reader.list_variables(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_list_variables_not_mat_file() -> None:
    """Test error when file is not a .mat file for list_variables."""
    reader = MATReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(MATReadError) as exc_info:
        reader.list_variables(txt_file)

    assert "Not a MATLAB data file" in str(exc_info.value)
