"""Tests for capabilities: scanning for web frameworks."""

from __future__ import annotations

from pathlib import Path

from platform_kaggle.capabilities import scan_codebase


class TestScanCodebaseWeb:
    """Scanning for web frameworks."""

    def test_scan_codebase_with_fastapi(self, tmp_path: Path) -> None:
        """Test scanning codebase detects FastAPI framework."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "api_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "api-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.109.0"
"""
        )

        profile = scan_codebase(tmp_path)

        assert "fastapi" in profile.frameworks

    def test_scan_codebase_with_flask(self, tmp_path: Path) -> None:
        """Test scanning codebase detects Flask framework."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "flask_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "flask-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
flask = "^3.0.0"
"""
        )

        profile = scan_codebase(tmp_path)

        assert "flask" in profile.frameworks

    def test_scan_codebase_with_django(self, tmp_path: Path) -> None:
        """Test scanning codebase detects Django framework."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "django_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "django-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
django = "^5.0.0"
"""
        )

        profile = scan_codebase(tmp_path)

        assert "django" in profile.frameworks

    def test_scan_codebase_with_starlette(self, tmp_path: Path) -> None:
        """Test scanning codebase detects Starlette framework."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "starlette_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "starlette-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
starlette = "^0.35.0"
"""
        )

        profile = scan_codebase(tmp_path)

        assert "starlette" in profile.frameworks

    def test_scan_codebase_with_aiohttp(self, tmp_path: Path) -> None:
        """Test scanning codebase detects aiohttp framework."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "aiohttp_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "aiohttp-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
aiohttp = "^3.9.0"
"""
        )

        profile = scan_codebase(tmp_path)

        assert "aiohttp" in profile.frameworks

    def test_scan_codebase_detects_python_technology(self, tmp_path: Path) -> None:
        """Test scanning codebase detects Python as a technology."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "any_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "any-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert "python" in profile.technologies

    def test_scan_codebase_with_openai_detects_llm_api(self, tmp_path: Path) -> None:
        """Test scanning codebase with OpenAI also detects LLM API capability."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "gpt_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "gpt-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
openai = "^1.12.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        # Should have both the NLP capability and the LLM API capability
        assert "speech_to_text" in cap_names
        assert "openai_api" in cap_names
