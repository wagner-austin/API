"""Tests for platform_codebase.toml module."""

from __future__ import annotations

from pathlib import Path

from platform_codebase.toml import (
    extract_poetry_dependencies,
    extract_poetry_name,
    parse_pyproject,
)


class TestExtractPoetryName:
    """Tests for extract_poetry_name function."""

    def test_extracts_name(self) -> None:
        """Test extracting name from poetry section."""
        content = """
[tool.poetry]
name = "my-package"
version = "1.0.0"
"""
        result = extract_poetry_name(content)
        assert result == "my-package"

    def test_extracts_name_single_quotes(self) -> None:
        """Test extracting name with single quotes."""
        content = """
[tool.poetry]
name = 'another-package'
version = "1.0.0"
"""
        result = extract_poetry_name(content)
        assert result == "another-package"

    def test_no_poetry_section(self) -> None:
        """Test with no [tool.poetry] section."""
        content = """
[project]
name = "other-package"
"""
        result = extract_poetry_name(content)
        assert result == ""

    def test_no_name_field(self) -> None:
        """Test with [tool.poetry] but no name field."""
        content = """
[tool.poetry]
version = "1.0.0"
"""
        result = extract_poetry_name(content)
        assert result == ""


class TestExtractPoetryDependencies:
    """Tests for extract_poetry_dependencies function."""

    def test_extracts_dependencies(self) -> None:
        """Test extracting dependencies from section."""
        content = """
[tool.poetry.dependencies]
python = "^3.11"
requests = "^2.31.0"
httpx = "^0.27.0"

[tool.poetry.group.dev.dependencies]
pytest = "^9.0.0"
"""
        result = extract_poetry_dependencies(content)
        assert "requests" in result
        assert "httpx" in result
        assert "python" not in result  # Python is excluded

    def test_dependencies_with_extras(self) -> None:
        """Test dependencies with extras and complex versions."""
        content = """
[tool.poetry.dependencies]
python = "^3.11"
fastapi = { version = "^0.100", extras = ["all"] }
"""
        result = extract_poetry_dependencies(content)
        assert "fastapi" in result

    def test_no_dependencies_section(self) -> None:
        """Test with no dependencies section."""
        content = """
[tool.poetry]
name = "my-package"
"""
        result = extract_poetry_dependencies(content)
        assert result == []

    def test_empty_dependencies_section(self) -> None:
        """Test with empty dependencies section."""
        content = """
[tool.poetry.dependencies]

[tool.other]
foo = "bar"
"""
        result = extract_poetry_dependencies(content)
        assert result == []

    def test_dependencies_at_end_of_file(self) -> None:
        """Test dependencies section at end of file without trailing section."""
        content = """
[tool.poetry]
name = "pkg"

[tool.poetry.dependencies]
python = "^3.11"
xgboost = "^2.0.0"
"""
        result = extract_poetry_dependencies(content)
        assert "xgboost" in result


class TestParsePyproject:
    """Tests for parse_pyproject function."""

    def test_parses_file(self, tmp_path: Path) -> None:
        """Test parsing a pyproject.toml file."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.poetry]
name = "test-package"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
httpx = "^0.27.0"
pandas = "^2.0.0"
""")

        name, deps = parse_pyproject(pyproject)
        assert name == "test-package"
        assert "httpx" in deps
        assert "pandas" in deps
        assert "python" not in deps
