from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.toml_reader import (
    check_banned_api,
    extract_mypy_bool,
    extract_mypy_files,
    extract_ruff_src,
    read_pyproject,
)


def test_extract_mypy_files_parses_array() -> None:
    """Test extracting files list from [tool.mypy]."""
    toml_content = """
[tool.mypy]
strict = true
files = ["src", "tests", "scripts"]
python_version = "3.11"
"""
    result = extract_mypy_files(toml_content)
    assert result == ["src", "tests", "scripts"]


def test_extract_mypy_files_returns_none_when_missing() -> None:
    """Test that missing files returns None."""
    toml_content = """
[tool.mypy]
strict = true
python_version = "3.11"
"""
    result = extract_mypy_files(toml_content)
    assert result is None


def test_extract_mypy_files_returns_none_when_empty() -> None:
    """Test that empty files array returns None."""
    toml_content = """
[tool.mypy]
files = []
"""
    result = extract_mypy_files(toml_content)
    assert result is None


def test_extract_mypy_bool_returns_true() -> None:
    """Test extracting true boolean value."""
    toml_content = """
[tool.mypy]
strict = true
disallow_any_expr = true
"""
    assert extract_mypy_bool(toml_content, "strict") is True
    assert extract_mypy_bool(toml_content, "disallow_any_expr") is True


def test_extract_mypy_bool_returns_false() -> None:
    """Test extracting false boolean value."""
    toml_content = """
[tool.mypy]
strict = false
"""
    assert extract_mypy_bool(toml_content, "strict") is False


def test_extract_mypy_bool_returns_none_when_missing() -> None:
    """Test that missing key returns None."""
    toml_content = """
[tool.mypy]
strict = true
"""
    assert extract_mypy_bool(toml_content, "nonexistent") is None


def test_extract_ruff_src_parses_array() -> None:
    """Test extracting src list from [tool.ruff]."""
    toml_content = """
[tool.ruff]
line-length = 100
src = ["src", "tests", "scripts"]
"""
    result = extract_ruff_src(toml_content)
    assert result == ["src", "tests", "scripts"]


def test_extract_ruff_src_returns_none_when_missing() -> None:
    """Test that missing src returns None."""
    toml_content = """
[tool.ruff]
line-length = 100
"""
    result = extract_ruff_src(toml_content)
    assert result is None


def test_extract_ruff_src_returns_none_when_empty() -> None:
    """Test that empty src array returns None."""
    toml_content = """
[tool.ruff]
src = []
"""
    result = extract_ruff_src(toml_content)
    assert result is None


def test_check_banned_api_finds_typing_any() -> None:
    """Test that banned typing.Any is detected."""
    toml_content = """
[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.Any" = { msg = "Do not use Any" }
"""
    assert check_banned_api(toml_content, "typing.Any") is True


def test_check_banned_api_finds_typing_cast() -> None:
    """Test that banned typing.cast is detected."""
    toml_content = """
[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.cast" = { msg = "Do not use cast" }
"""
    assert check_banned_api(toml_content, "typing.cast") is True


def test_check_banned_api_returns_false_when_missing() -> None:
    """Test that missing banned API returns False."""
    toml_content = """
[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.Any" = { msg = "Do not use Any" }
"""
    assert check_banned_api(toml_content, "typing.cast") is False


def test_read_pyproject_reads_file(tmp_path: Path) -> None:
    """Test reading pyproject.toml file."""
    pyproject = tmp_path / "pyproject.toml"
    content = "[tool.mypy]\nstrict = true\n"
    pyproject.write_text(content, encoding="utf-8")

    result = read_pyproject(pyproject)
    assert result == content


def test_read_pyproject_raises_on_missing_file(tmp_path: Path) -> None:
    """Test that missing file raises RuntimeError."""
    pyproject = tmp_path / "nonexistent.toml"

    with pytest.raises(RuntimeError, match=r"Failed to read pyproject\.toml"):
        read_pyproject(pyproject)


__all__ = [
    "test_check_banned_api_finds_typing_any",
    "test_check_banned_api_finds_typing_cast",
    "test_check_banned_api_returns_false_when_missing",
    "test_extract_mypy_bool_returns_false",
    "test_extract_mypy_bool_returns_none_when_missing",
    "test_extract_mypy_bool_returns_true",
    "test_extract_mypy_files_parses_array",
    "test_extract_mypy_files_returns_none_when_empty",
    "test_extract_mypy_files_returns_none_when_missing",
    "test_extract_ruff_src_parses_array",
    "test_extract_ruff_src_returns_none_when_empty",
    "test_extract_ruff_src_returns_none_when_missing",
    "test_read_pyproject_raises_on_missing_file",
    "test_read_pyproject_reads_file",
]
