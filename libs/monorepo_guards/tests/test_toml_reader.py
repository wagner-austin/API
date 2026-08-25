from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.toml_reader import (
    PackageInclude,
    check_banned_api,
    extract_mypy_bool,
    extract_mypy_files,
    extract_package_includes,
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


def test_extract_package_includes_reports_from_per_entry() -> None:
    """The shape every package in this monorepo used until 2026-08-24."""
    toml_content = """
[tool.poetry]
name = "platform-core"
packages = [
  { include = "platform_core", from = "src" },
  { include = "scripts" },
]
"""
    assert extract_package_includes(toml_content) == [
        PackageInclude(include="platform_core", has_from=True, line_no=5),
        PackageInclude(include="scripts", has_from=False, line_no=6),
    ]


def test_extract_package_includes_reads_a_single_line_array() -> None:
    """Poetry accepts the whole array on one line; both entries are found."""
    toml_content = """
[tool.poetry]
packages = [{ include = "a", from = "src" }, { include = "b" }]
"""
    assert extract_package_includes(toml_content) == [
        PackageInclude(include="a", has_from=True, line_no=3),
        PackageInclude(include="b", has_from=False, line_no=3),
    ]


def test_extract_package_includes_ignores_other_sections() -> None:
    """A `packages` key outside [tool.poetry] is not a poetry declaration.

    Without the section check this reads a foreign tool's config as if it
    governed what ships in the wheel.
    """
    toml_content = """
[tool.poetry]
name = "thing"

[tool.other]
packages = [
  { include = "not_poetrys_business" },
]
"""
    assert extract_package_includes(toml_content) == []


def test_extract_package_includes_returns_empty_without_a_packages_key() -> None:
    toml_content = """
[tool.poetry]
name = "thing"
version = "0.1.0"
"""
    assert extract_package_includes(toml_content) == []


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
    "test_extract_package_includes_ignores_other_sections",
    "test_extract_package_includes_reads_a_single_line_array",
    "test_extract_package_includes_reports_from_per_entry",
    "test_extract_package_includes_returns_empty_without_a_packages_key",
    "test_extract_ruff_src_parses_array",
    "test_extract_ruff_src_returns_none_when_empty",
    "test_extract_ruff_src_returns_none_when_missing",
    "test_read_pyproject_raises_on_missing_file",
    "test_read_pyproject_reads_file",
]
