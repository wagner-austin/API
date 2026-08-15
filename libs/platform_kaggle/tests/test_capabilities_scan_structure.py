"""Tests for capabilities: scanning trees with missing or malformed packages."""

from __future__ import annotations

from pathlib import Path

from platform_kaggle.capabilities import scan_codebase


class TestScanCodebaseTreeStructure:
    """Scanning trees with missing or malformed packages."""

    def test_scan_codebase_empty(self, tmp_path: Path) -> None:
        """Test scanning empty codebase."""
        # Create empty libs and services directories
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()
        assert profile.ml_backends == ()
        assert profile.data_formats == ()
        assert profile.task_types == ()

    def test_scan_codebase_no_libs_dir(self, tmp_path: Path) -> None:
        """Test scanning codebase without libs directory."""
        # Only create services directory
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()

    def test_scan_codebase_no_services_dir(self, tmp_path: Path) -> None:
        """Test scanning codebase without services directory."""
        # Only create libs directory
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()

    def test_scan_codebase_skips_non_dirs(self, tmp_path: Path) -> None:
        """Test scanning codebase skips files in libs/services directories."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        # Create a file instead of a directory
        not_a_dir = libs_dir / "not_a_lib.txt"
        not_a_dir.write_text("This is not a library")

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()

    def test_scan_codebase_skips_missing_pyproject(self, tmp_path: Path) -> None:
        """Test scanning codebase skips directories without pyproject.toml."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        # Create a directory without pyproject.toml
        lib_dir = libs_dir / "incomplete_lib"
        lib_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()

    def test_scan_codebase_with_tensorflow(self, tmp_path: Path) -> None:
        """Test scanning codebase with TensorFlow dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "tf_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "tf-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
tensorflow = "^2.0.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert "tensorflow" in profile.ml_backends

    def test_scan_codebase_pyproject_no_name(self, tmp_path: Path) -> None:
        """Test scanning pyproject.toml without name field."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "noname_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        # Missing [tool.poetry] name field
        pyproject.write_text(
            """
[tool.poetry]
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
pandas = "^2.0.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        # Should still work, just have empty name
        profile = scan_codebase(tmp_path)
        # pandas was detected
        assert "csv" in profile.data_formats

    def test_scan_codebase_pyproject_no_dependencies(self, tmp_path: Path) -> None:
        """Test scanning pyproject.toml without dependencies section."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "nodeps_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        # No [tool.poetry.dependencies] section
        pyproject.write_text(
            """
[tool.poetry]
name = "nodeps-lib"
version = "0.1.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        # Should still work, just have no dependencies
        profile = scan_codebase(tmp_path)
        assert profile.capabilities == ()

    def test_scan_codebase_service_skips_non_dirs(self, tmp_path: Path) -> None:
        """Test scanning skips files in services directory."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        # Create a file instead of a directory in services
        not_a_dir = services_dir / "not_a_service.txt"
        not_a_dir.write_text("This is not a service")

        profile = scan_codebase(tmp_path)
        assert profile.capabilities == ()

    def test_scan_codebase_service_skips_missing_pyproject(self, tmp_path: Path) -> None:
        """Test scanning skips service directories without pyproject.toml."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        # Create a directory without pyproject.toml
        service_dir = services_dir / "incomplete_service"
        service_dir.mkdir()

        profile = scan_codebase(tmp_path)
        assert profile.capabilities == ()
