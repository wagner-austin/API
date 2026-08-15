"""Tests for capabilities: scanning for LLM API and cloud libraries."""

from __future__ import annotations

from pathlib import Path

from platform_kaggle.capabilities import scan_codebase


class TestScanCodebaseCloud:
    """Scanning for LLM API and cloud libraries."""

    def test_scan_codebase_with_google_generativeai(self, tmp_path: Path) -> None:
        """Test scanning codebase with google-generativeai (Gemini) dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "llm_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "llm-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
google-generativeai = "^0.3.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "gemini_api" in cap_names

    def test_scan_codebase_with_vertex_ai(self, tmp_path: Path) -> None:
        """Test scanning codebase with google-cloud-aiplatform dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "vertex_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "vertex-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
google-cloud-aiplatform = "^1.38.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "vertex_ai" in cap_names

    def test_scan_codebase_with_anthropic(self, tmp_path: Path) -> None:
        """Test scanning codebase with anthropic dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "claude_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "claude-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
anthropic = "^0.18.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "anthropic_claude" in cap_names

    def test_scan_codebase_with_langchain(self, tmp_path: Path) -> None:
        """Test scanning codebase with langchain dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "rag_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "rag-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
langchain = "^0.1.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "langchain" in cap_names

    def test_scan_codebase_with_gcs(self, tmp_path: Path) -> None:
        """Test scanning codebase with google-cloud-storage dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "storage_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "storage-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
google-cloud-storage = "^2.14.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "google_cloud_storage" in cap_names

    def test_scan_codebase_with_bigquery(self, tmp_path: Path) -> None:
        """Test scanning codebase with google-cloud-bigquery dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "bq_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "bq-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
google-cloud-bigquery = "^3.14.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "bigquery" in cap_names

    def test_scan_codebase_with_boto3(self, tmp_path: Path) -> None:
        """Test scanning codebase with boto3 (AWS SDK) dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "aws_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "aws-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
boto3 = "^1.34.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "aws_sdk" in cap_names

    def test_scan_codebase_with_azure(self, tmp_path: Path) -> None:
        """Test scanning codebase with azure-core dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "azure_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "azure-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
azure-core = "^1.29.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "azure_sdk" in cap_names
