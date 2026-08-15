"""Tests for github scanner: ParseGitHubRepo."""

from __future__ import annotations

from pathlib import Path

import pytest

from platform_codebase.github_scanner import (
    parse_github_repo,
    scan_libs_from_github,
    scan_services_from_github,
)
from platform_codebase.testing import (
    FakeGitHubClient,
)


class TestParseGitHubRepo:
    """Tests for parse_github_repo function."""

    def test_parses_valid_repo(self) -> None:
        """Test parsing valid owner/repo format."""
        owner, repo = parse_github_repo("wagner-austin/API")
        assert owner == "wagner-austin"
        assert repo == "API"

    def test_parses_with_hyphens(self) -> None:
        """Test parsing repo with hyphens."""
        owner, repo = parse_github_repo("my-org/my-repo")
        assert owner == "my-org"
        assert repo == "my-repo"

    def test_raises_on_no_slash(self) -> None:
        """Test error when no slash in input."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            parse_github_repo("noslash")

    def test_raises_on_multiple_slashes(self) -> None:
        """Test error when multiple slashes in input."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            parse_github_repo("too/many/slashes")

    def test_raises_on_empty_string(self) -> None:
        """Test error on empty string."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            parse_github_repo("")


class TestScanLibsFromGitHub:
    """Tests for scan_libs_from_github function."""

    def test_scans_libs_directory(self) -> None:
        """Test scanning libs directory."""
        pyproject_content = """
[tool.poetry]
name = "platform-core"

[tool.poetry.dependencies]
python = "^3.11"
httpx = "^0.27.0"
"""
        client = FakeGitHubClient(
            directories={"libs": ["platform_core", "platform_ml"]},
            files={
                "libs/platform_core/pyproject.toml": pyproject_content,
                "libs/platform_ml/pyproject.toml": """
[tool.poetry]
name = "platform-ml"

[tool.poetry.dependencies]
python = "^3.11"
torch = "^2.0.0"
""",
            },
        )

        result = scan_libs_from_github(client, "owner", "repo")

        assert len(result) == 2
        names = {lib.name for lib in result}
        assert "platform-core" in names
        assert "platform-ml" in names

    def test_returns_lib_info_with_path(self) -> None:
        """Test that LibInfo includes correct path."""
        client = FakeGitHubClient(
            directories={"libs": ["mylib"]},
            files={
                "libs/mylib/pyproject.toml": """
[tool.poetry]
name = "my-lib"

[tool.poetry.dependencies]
python = "^3.11"
"""
            },
        )

        result = scan_libs_from_github(client, "owner", "repo")

        assert len(result) == 1
        assert result[0].path == Path("libs/mylib")

    def test_returns_lib_info_with_dependencies(self) -> None:
        """Test that LibInfo includes dependencies."""
        client = FakeGitHubClient(
            directories={"libs": ["mylib"]},
            files={
                "libs/mylib/pyproject.toml": """
[tool.poetry]
name = "my-lib"

[tool.poetry.dependencies]
python = "^3.11"
httpx = "^0.27.0"
pandas = "^2.0.0"
"""
            },
        )

        result = scan_libs_from_github(client, "owner", "repo")

        assert len(result) == 1
        assert "httpx" in result[0].dependencies
        assert "pandas" in result[0].dependencies

    def test_skips_libs_without_pyproject(self) -> None:
        """Test that libs without pyproject.toml are skipped."""
        client = FakeGitHubClient(
            directories={"libs": ["valid_lib", "missing_pyproject"]},
            files={
                "libs/valid_lib/pyproject.toml": """
[tool.poetry]
name = "valid-lib"

[tool.poetry.dependencies]
python = "^3.11"
"""
            },
        )

        result = scan_libs_from_github(client, "owner", "repo")

        assert len(result) == 1
        assert result[0].name == "valid-lib"

    def test_empty_libs_directory(self) -> None:
        """Test with empty libs directory."""
        client = FakeGitHubClient(directories={"libs": []})

        result = scan_libs_from_github(client, "owner", "repo")

        assert result == ()

    def test_no_libs_directory(self) -> None:
        """Test when libs directory doesn't exist."""
        client = FakeGitHubClient()

        result = scan_libs_from_github(client, "owner", "repo")

        assert result == ()


class TestScanServicesFromGitHub:
    """Tests for scan_services_from_github function."""

    def test_scans_services_directory(self) -> None:
        """Test scanning services directory."""
        client = FakeGitHubClient(
            directories={"services": ["api-service", "worker-service"]},
            files={
                "services/api-service/pyproject.toml": """
[tool.poetry]
name = "api-service"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.100.0"
""",
                "services/worker-service/pyproject.toml": """
[tool.poetry]
name = "worker-service"

[tool.poetry.dependencies]
python = "^3.11"
rq = "^1.15.0"
""",
            },
        )

        result = scan_services_from_github(client, "owner", "repo")

        assert len(result) == 2
        names = {svc.name for svc in result}
        assert "api-service" in names
        assert "worker-service" in names

    def test_returns_service_info_with_path(self) -> None:
        """Test that ServiceInfo includes correct path."""
        client = FakeGitHubClient(
            directories={"services": ["my-api"]},
            files={
                "services/my-api/pyproject.toml": """
[tool.poetry]
name = "my-api"

[tool.poetry.dependencies]
python = "^3.11"
"""
            },
        )

        result = scan_services_from_github(client, "owner", "repo")

        assert len(result) == 1
        assert result[0].path == Path("services/my-api")

    def test_returns_service_info_with_dependencies(self) -> None:
        """Test that ServiceInfo includes dependencies."""
        client = FakeGitHubClient(
            directories={"services": ["my-api"]},
            files={
                "services/my-api/pyproject.toml": """
[tool.poetry]
name = "my-api"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.100.0"
uvicorn = "^0.23.0"
"""
            },
        )

        result = scan_services_from_github(client, "owner", "repo")

        assert len(result) == 1
        assert "fastapi" in result[0].dependencies
        assert "uvicorn" in result[0].dependencies

    def test_detects_rules_files(self) -> None:
        """Test detection of .rules files."""
        client = FakeGitHubClient(
            directories={"services": ["turkic-api"]},
            files={
                "services/turkic-api/pyproject.toml": """
[tool.poetry]
name = "turkic-api"

[tool.poetry.dependencies]
python = "^3.11"
"""
            },
            path_patterns={("services/turkic-api", ".rules"): True},
        )

        result = scan_services_from_github(client, "owner", "repo")

        assert len(result) == 1
        assert result[0].has_rules_files is True

    def test_no_rules_files(self) -> None:
        """Test service without .rules files."""
        client = FakeGitHubClient(
            directories={"services": ["my-api"]},
            files={
                "services/my-api/pyproject.toml": """
[tool.poetry]
name = "my-api"

[tool.poetry.dependencies]
python = "^3.11"
"""
            },
            path_patterns={("services/my-api", ".rules"): False},
        )

        result = scan_services_from_github(client, "owner", "repo")

        assert len(result) == 1
        assert result[0].has_rules_files is False

    def test_skips_services_without_pyproject(self) -> None:
        """Test that services without pyproject.toml are skipped."""
        client = FakeGitHubClient(
            directories={"services": ["valid_svc", "no_pyproject"]},
            files={
                "services/valid_svc/pyproject.toml": """
[tool.poetry]
name = "valid-svc"

[tool.poetry.dependencies]
python = "^3.11"
"""
            },
        )

        result = scan_services_from_github(client, "owner", "repo")

        assert len(result) == 1
        assert result[0].name == "valid-svc"

    def test_empty_services_directory(self) -> None:
        """Test with empty services directory."""
        client = FakeGitHubClient(directories={"services": []})

        result = scan_services_from_github(client, "owner", "repo")

        assert result == ()

    def test_no_services_directory(self) -> None:
        """Test when services directory doesn't exist."""
        client = FakeGitHubClient()

        result = scan_services_from_github(client, "owner", "repo")

        assert result == ()
