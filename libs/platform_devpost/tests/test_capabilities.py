"""Tests for platform_devpost.capabilities module."""

from __future__ import annotations

from pathlib import Path

from platform_devpost.capabilities import scan_codebase


class TestScanCodebase:
    """Tests for scan_codebase function."""

    def test_scan_codebase_returns_profile(self, tmp_path: Path) -> None:
        """Test scan_codebase returns a CodebaseProfile."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()
        assert profile.technologies == ()
        assert profile.frameworks == ()

    def test_scan_codebase_detects_polars(self, tmp_path: Path) -> None:
        """Test scan_codebase detects polars dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        lib_dir = libs_dir / "test_lib"
        lib_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text("""
[tool.poetry]
name = "test-lib"

[tool.poetry.dependencies]
python = "^3.11"
polars = "^0.18.0"
""")

        profile = scan_codebase(tmp_path)

        assert "python" in profile.technologies
        assert any(cap.name == "data_analysis" for cap in profile.capabilities)

    def test_scan_codebase_detects_flask(self, tmp_path: Path) -> None:
        """Test scan_codebase detects Flask framework."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        lib_dir = libs_dir / "test_lib"
        lib_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text("""
[tool.poetry]
name = "test-lib"

[tool.poetry.dependencies]
python = "^3.11"
flask = "^2.0.0"
""")

        profile = scan_codebase(tmp_path)

        assert "flask" in profile.frameworks
        assert any(cap.name == "web_development" for cap in profile.capabilities)

    def test_scan_codebase_detects_ml_libs(self, tmp_path: Path) -> None:
        """Test scan_codebase detects ML libraries."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        lib_dir = libs_dir / "ml_lib"
        lib_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text("""
[tool.poetry]
name = "ml-lib"

[tool.poetry.dependencies]
python = "^3.11"
pytorch = "^2.0.0"
xgboost = "^1.7.0"
""")

        profile = scan_codebase(tmp_path)

        assert "pytorch" in profile.frameworks
        assert any(cap.name == "machine_learning" for cap in profile.capabilities)

    def test_scan_codebase_from_services(self, tmp_path: Path) -> None:
        """Test scan_codebase scans services directory."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()
        service_dir = services_dir / "api-service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text("""
[tool.poetry]
name = "api-service"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.100.0"
""")

        profile = scan_codebase(tmp_path)

        assert "fastapi" in profile.frameworks
        assert any(cap.name == "web_development" for cap in profile.capabilities)

    def test_scan_codebase_deduplicates_capabilities(self, tmp_path: Path) -> None:
        """Test scan_codebase does not duplicate capabilities."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        lib1 = libs_dir / "lib1"
        lib1.mkdir()
        (lib1 / "pyproject.toml").write_text("""
[tool.poetry]
name = "lib1"

[tool.poetry.dependencies]
flask = "^2.0.0"
""")

        lib2 = libs_dir / "lib2"
        lib2.mkdir()
        (lib2 / "pyproject.toml").write_text("""
[tool.poetry]
name = "lib2"

[tool.poetry.dependencies]
flask = "^2.0.0"
""")

        profile = scan_codebase(tmp_path)

        web_caps = [c for c in profile.capabilities if c.name == "web_development"]
        assert len(web_caps) == 1

    def test_scan_codebase_detects_data_libs(self, tmp_path: Path) -> None:
        """Test scan_codebase detects data analysis libraries."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        lib_dir = libs_dir / "data_lib"
        lib_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text("""
[tool.poetry]
name = "data-lib"

[tool.poetry.dependencies]
python = "^3.11"
pandas = "^2.0.0"
polars = "^0.18.0"
numpy = "^1.25.0"
""")

        profile = scan_codebase(tmp_path)

        assert "python" in profile.technologies
        assert any(cap.name == "data_analysis" for cap in profile.capabilities)

    def test_scan_codebase_detects_ai_integration(self, tmp_path: Path) -> None:
        """Test scan_codebase detects AI integration libraries."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        lib_dir = libs_dir / "ai_lib"
        lib_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text("""
[tool.poetry]
name = "ai-lib"

[tool.poetry.dependencies]
python = "^3.11"
openai = "^1.0.0"
langchain = "^0.1.0"
""")

        profile = scan_codebase(tmp_path)

        assert "langchain" in profile.frameworks
        assert any(cap.name == "ai_integration" for cap in profile.capabilities)

    def test_scan_codebase_no_dirs(self, tmp_path: Path) -> None:
        """Test scan_codebase handles missing libs/services dirs."""
        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()
        assert profile.technologies == ()
        assert profile.frameworks == ()
