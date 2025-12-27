"""Tests for platform_codebase.scanner module."""

from __future__ import annotations

from pathlib import Path

from platform_codebase.scanner import (
    collect_all_dependencies,
    has_dependency,
    scan_libs,
    scan_services,
)
from platform_codebase.types import LibInfo, ServiceInfo


class TestScanLibs:
    """Tests for scan_libs function."""

    def test_scans_libs_directory(self, tmp_path: Path) -> None:
        """Test scanning libs/ directory."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib1 = libs_dir / "lib1"
        lib1.mkdir()
        (lib1 / "pyproject.toml").write_text("""
[tool.poetry]
name = "lib-one"

[tool.poetry.dependencies]
python = "^3.11"
httpx = "^0.27.0"
""")

        lib2 = libs_dir / "lib2"
        lib2.mkdir()
        (lib2 / "pyproject.toml").write_text("""
[tool.poetry]
name = "lib-two"

[tool.poetry.dependencies]
python = "^3.11"
requests = "^2.31.0"
""")

        result = scan_libs(tmp_path)

        assert len(result) == 2
        names = {lib.name for lib in result}
        assert "lib-one" in names
        assert "lib-two" in names

    def test_skips_non_directories(self, tmp_path: Path) -> None:
        """Test that non-directories are skipped."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        # Create a file (not directory) in libs/
        (libs_dir / "not_a_dir.txt").write_text("content")

        lib1 = libs_dir / "lib1"
        lib1.mkdir()
        (lib1 / "pyproject.toml").write_text("""
[tool.poetry]
name = "lib-one"

[tool.poetry.dependencies]
python = "^3.11"
""")

        result = scan_libs(tmp_path)
        assert len(result) == 1

    def test_skips_dirs_without_pyproject(self, tmp_path: Path) -> None:
        """Test that directories without pyproject.toml are skipped."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib1 = libs_dir / "lib1"
        lib1.mkdir()
        # No pyproject.toml

        lib2 = libs_dir / "lib2"
        lib2.mkdir()
        (lib2 / "pyproject.toml").write_text("""
[tool.poetry]
name = "lib-two"

[tool.poetry.dependencies]
python = "^3.11"
""")

        result = scan_libs(tmp_path)
        assert len(result) == 1
        assert result[0].name == "lib-two"

    def test_no_libs_directory(self, tmp_path: Path) -> None:
        """Test with no libs/ directory."""
        result = scan_libs(tmp_path)
        assert result == ()


class TestScanServices:
    """Tests for scan_services function."""

    def test_scans_services_directory(self, tmp_path: Path) -> None:
        """Test scanning services/ directory."""
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        svc1 = services_dir / "svc1"
        svc1.mkdir()
        (svc1 / "pyproject.toml").write_text("""
[tool.poetry]
name = "service-one"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.100.0"
""")

        result = scan_services(tmp_path)

        assert len(result) == 1
        assert result[0].name == "service-one"
        assert result[0].has_rules_files is False

    def test_detects_rules_files(self, tmp_path: Path) -> None:
        """Test detection of .rules files."""
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        svc1 = services_dir / "svc1"
        svc1.mkdir()
        (svc1 / "pyproject.toml").write_text("""
[tool.poetry]
name = "service-one"

[tool.poetry.dependencies]
python = "^3.11"
""")

        # Create a .rules file in a subdirectory
        rules_dir = svc1 / "rules"
        rules_dir.mkdir()
        (rules_dir / "cyrillic.rules").write_text("rule content")

        result = scan_services(tmp_path)

        assert len(result) == 1
        assert result[0].has_rules_files is True

    def test_skips_non_directories(self, tmp_path: Path) -> None:
        """Test that non-directories are skipped."""
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        # Create a file (not directory) in services/
        (services_dir / "not_a_dir.txt").write_text("content")

        result = scan_services(tmp_path)
        assert result == ()

    def test_skips_dirs_without_pyproject(self, tmp_path: Path) -> None:
        """Test that directories without pyproject.toml are skipped."""
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        svc1 = services_dir / "svc1"
        svc1.mkdir()
        # No pyproject.toml

        result = scan_services(tmp_path)
        assert result == ()

    def test_no_services_directory(self, tmp_path: Path) -> None:
        """Test with no services/ directory."""
        result = scan_services(tmp_path)
        assert result == ()


class TestHasDependency:
    """Tests for has_dependency function."""

    def test_exact_match(self) -> None:
        """Test exact dependency match."""
        deps = ("httpx", "requests", "pandas")
        assert has_dependency(deps, "httpx") is True
        assert has_dependency(deps, "requests") is True
        assert has_dependency(deps, "missing") is False

    def test_with_extras(self) -> None:
        """Test dependency with extras notation."""
        deps = ("fastapi[all]", "httpx", "pandas[excel]")
        assert has_dependency(deps, "fastapi") is True
        assert has_dependency(deps, "pandas") is True
        assert has_dependency(deps, "httpx") is True

    def test_no_partial_match(self) -> None:
        """Test that partial matches don't count."""
        deps = ("httpx-oauth", "pandas-stubs")
        assert has_dependency(deps, "httpx") is False
        assert has_dependency(deps, "pandas") is False


class TestCollectAllDependencies:
    """Tests for collect_all_dependencies function."""

    def test_collects_from_libs_and_services(self) -> None:
        """Test collecting dependencies from both libs and services."""
        libs: tuple[LibInfo, ...] = (
            LibInfo(
                name="lib1",
                path=Path("libs/lib1"),
                dependencies=("httpx", "pandas"),
            ),
            LibInfo(
                name="lib2",
                path=Path("libs/lib2"),
                dependencies=("requests",),
            ),
        )
        services: tuple[ServiceInfo, ...] = (
            ServiceInfo(
                name="svc1",
                path=Path("services/svc1"),
                dependencies=("fastapi", "pandas"),
                has_rules_files=False,
            ),
        )

        result = collect_all_dependencies(libs, services)

        assert "httpx" in result
        assert "pandas" in result
        assert "requests" in result
        assert "fastapi" in result
        # Should be deduplicated and sorted
        assert result == tuple(sorted(set(result)))

    def test_empty_inputs(self) -> None:
        """Test with empty libs and services."""
        result = collect_all_dependencies((), ())
        assert result == ()
