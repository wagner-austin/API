"""Tests for platform_codebase.github_scanner module."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue

from platform_codebase.github_scanner import (
    GitHubClient,
    parse_github_repo,
    scan_libs_from_github,
    scan_services_from_github,
)
from platform_codebase.testing import (
    FakeGitHubClient,
    FakeHttpxClient,
    FakeHttpxResponse,
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


class TestGitHubClient:
    """Tests for GitHubClient class."""

    def test_headers_include_token(self) -> None:
        """Test that headers include authorization token."""
        http_client = FakeHttpxClient()
        client = GitHubClient("test-token", client=http_client)

        headers = client._headers()

        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["X-GitHub-Api-Version"] == "2022-11-28"

    def test_list_directory_returns_dirs(self) -> None:
        """Test listing directories from GitHub API."""
        response_data: JSONValue = [
            {"type": "dir", "name": "lib1"},
            {"type": "dir", "name": "lib2"},
            {"type": "file", "name": "README.md"},
        ]
        http_client = FakeHttpxClient(
            responses={
                "contents/libs": FakeHttpxResponse(data=response_data),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.list_directory("owner", "repo", "libs")

        assert result == ["lib1", "lib2"]

    def test_list_directory_returns_empty_on_404(self) -> None:
        """Test that 404 returns empty list."""
        http_client = FakeHttpxClient(
            responses={
                "contents/libs": FakeHttpxResponse(status_code=404),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.list_directory("owner", "repo", "libs")

        assert result == []

    def test_list_directory_returns_empty_on_non_list(self) -> None:
        """Test that non-list response returns empty."""
        http_client = FakeHttpxClient(
            responses={
                "contents/libs": FakeHttpxResponse(data={"type": "file"}),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.list_directory("owner", "repo", "libs")

        assert result == []

    def test_list_directory_skips_non_dict_items(self) -> None:
        """Test that non-dict items in list are skipped."""
        response_data: JSONValue = [
            {"type": "dir", "name": "lib1"},
            "invalid",
            {"type": "dir", "name": "lib2"},
        ]
        http_client = FakeHttpxClient(
            responses={
                "contents/libs": FakeHttpxResponse(data=response_data),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.list_directory("owner", "repo", "libs")

        assert result == ["lib1", "lib2"]

    def test_list_directory_skips_non_string_names(self) -> None:
        """Test that items with non-string names are skipped."""
        response_data: JSONValue = [
            {"type": "dir", "name": "lib1"},
            {"type": "dir", "name": 123},
        ]
        http_client = FakeHttpxClient(
            responses={
                "contents/libs": FakeHttpxResponse(data=response_data),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.list_directory("owner", "repo", "libs")

        assert result == ["lib1"]

    def test_get_file_content_decodes_base64(self) -> None:
        """Test decoding base64 file content."""
        file_content = '[tool.poetry]\nname = "test"\n'
        encoded = base64.b64encode(file_content.encode()).decode()
        # Add newlines like GitHub does
        encoded_with_newlines = "\n".join(encoded[i : i + 76] for i in range(0, len(encoded), 76))
        response_data: JSONValue = {
            "encoding": "base64",
            "content": encoded_with_newlines,
        }
        http_client = FakeHttpxClient(
            responses={
                "pyproject.toml": FakeHttpxResponse(data=response_data),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.get_file_content("owner", "repo", "pyproject.toml")

        assert result == file_content

    def test_get_file_content_returns_none_on_404(self) -> None:
        """Test that 404 returns None."""
        http_client = FakeHttpxClient(
            responses={
                "pyproject.toml": FakeHttpxResponse(status_code=404),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.get_file_content("owner", "repo", "pyproject.toml")

        assert result is None

    def test_get_file_content_returns_none_on_non_dict(self) -> None:
        """Test that non-dict response returns None."""
        http_client = FakeHttpxClient(
            responses={
                "pyproject.toml": FakeHttpxResponse(data=[]),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.get_file_content("owner", "repo", "pyproject.toml")

        assert result is None

    def test_get_file_content_returns_none_on_non_base64(self) -> None:
        """Test that non-base64 encoding returns None."""
        response_data: JSONValue = {
            "encoding": "utf-8",
            "content": "raw content",
        }
        http_client = FakeHttpxClient(
            responses={
                "pyproject.toml": FakeHttpxResponse(data=response_data),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.get_file_content("owner", "repo", "pyproject.toml")

        assert result is None

    def test_get_file_content_returns_none_on_non_string_content(self) -> None:
        """Test that non-string content returns None."""
        response_data: JSONValue = {
            "encoding": "base64",
            "content": 12345,
        }
        http_client = FakeHttpxClient(
            responses={
                "pyproject.toml": FakeHttpxResponse(data=response_data),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.get_file_content("owner", "repo", "pyproject.toml")

        assert result is None

    def test_check_path_exists_returns_true_on_match(self) -> None:
        """Test pattern matching returns true when file exists."""
        response_data: JSONValue = [
            {"name": "cyrillic.rules"},
            {"name": "README.md"},
        ]
        http_client = FakeHttpxClient(
            responses={
                "services/turkic": FakeHttpxResponse(data=response_data),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.check_path_exists("owner", "repo", "services/turkic", ".rules")

        assert result is True

    def test_check_path_exists_returns_false_on_no_match(self) -> None:
        """Test pattern matching returns false when no file matches."""
        response_data: JSONValue = [
            {"name": "README.md"},
            {"name": "main.py"},
        ]
        http_client = FakeHttpxClient(
            responses={
                "services/api": FakeHttpxResponse(data=response_data),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.check_path_exists("owner", "repo", "services/api", ".rules")

        assert result is False

    def test_check_path_exists_returns_false_on_404(self) -> None:
        """Test that 404 returns false."""
        http_client = FakeHttpxClient(
            responses={
                "services/missing": FakeHttpxResponse(status_code=404),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.check_path_exists("owner", "repo", "services/missing", ".rules")

        assert result is False

    def test_check_path_exists_returns_false_on_non_list(self) -> None:
        """Test that non-list response returns false."""
        http_client = FakeHttpxClient(
            responses={
                "services/file": FakeHttpxResponse(data={"type": "file"}),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.check_path_exists("owner", "repo", "services/file", ".rules")

        assert result is False

    def test_check_path_exists_skips_non_dict_items(self) -> None:
        """Test that non-dict items are skipped."""
        response_data: JSONValue = [
            "invalid",
            {"name": "file.rules"},
        ]
        http_client = FakeHttpxClient(
            responses={
                "services/mixed": FakeHttpxResponse(data=response_data),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.check_path_exists("owner", "repo", "services/mixed", ".rules")

        assert result is True

    def test_check_path_exists_skips_non_string_names(self) -> None:
        """Test that items with non-string names are skipped."""
        response_data: JSONValue = [
            {"name": 123},
            {"name": "file.rules"},
        ]
        http_client = FakeHttpxClient(
            responses={
                "services/typed": FakeHttpxResponse(data=response_data),
            }
        )
        client = GitHubClient("token", client=http_client)

        result = client.check_path_exists("owner", "repo", "services/typed", ".rules")

        assert result is True

    def test_close_calls_client_close(self) -> None:
        """Test that close() calls underlying client close()."""
        http_client = FakeHttpxClient()
        client = GitHubClient("token", client=http_client)

        # Should not raise
        client.close()


class TestFakeGitHubClient:
    """Tests for FakeGitHubClient."""

    def test_list_directory_returns_configured_dirs(self) -> None:
        """Test returning configured directories."""
        client = FakeGitHubClient(directories={"libs": ["lib1", "lib2"]})

        result = client.list_directory("owner", "repo", "libs")

        assert result == ["lib1", "lib2"]

    def test_list_directory_returns_empty_for_unknown_path(self) -> None:
        """Test returning empty list for unknown path."""
        client = FakeGitHubClient()

        result = client.list_directory("owner", "repo", "libs")

        assert result == []

    def test_get_file_content_returns_configured_content(self) -> None:
        """Test returning configured file content."""
        client = FakeGitHubClient(files={"path/file.txt": "content"})

        result = client.get_file_content("owner", "repo", "path/file.txt")

        assert result == "content"

    def test_get_file_content_returns_none_for_unknown_path(self) -> None:
        """Test returning None for unknown file."""
        client = FakeGitHubClient()

        result = client.get_file_content("owner", "repo", "missing.txt")

        assert result is None

    def test_check_path_exists_returns_configured_result(self) -> None:
        """Test returning configured pattern check result."""
        client = FakeGitHubClient(path_patterns={("services/api", ".rules"): True})

        result = client.check_path_exists("owner", "repo", "services/api", ".rules")

        assert result is True

    def test_check_path_exists_returns_false_for_unknown_pattern(self) -> None:
        """Test returning False for unknown pattern."""
        client = FakeGitHubClient()

        result = client.check_path_exists("owner", "repo", "services/api", ".rules")

        assert result is False


class TestFakeHttpxResponse:
    """Tests for FakeHttpxResponse."""

    def test_default_values(self) -> None:
        """Test default response values."""
        response = FakeHttpxResponse()

        assert response.status_code == 200
        assert response.text == ""
        assert response.headers == {}
        assert response.content == b""
        assert response.json() is None

    def test_custom_status_code(self) -> None:
        """Test custom status code."""
        response = FakeHttpxResponse(status_code=404)

        assert response.status_code == 404

    def test_custom_data(self) -> None:
        """Test custom JSON data."""
        response = FakeHttpxResponse(data={"key": "value"})

        assert response.json() == {"key": "value"}

    def test_custom_text(self) -> None:
        """Test custom text content."""
        response = FakeHttpxResponse(text="Hello")

        assert response.text == "Hello"
        assert response.content == b"Hello"

    def test_custom_headers(self) -> None:
        """Test custom headers."""
        response = FakeHttpxResponse(headers={"X-Custom": "value"})

        assert response.headers["X-Custom"] == "value"

    def test_raise_for_status_does_nothing_on_success(self) -> None:
        """Test raise_for_status is no-op on success."""
        response = FakeHttpxResponse(status_code=200)

        # Should not raise
        response.raise_for_status()

    def test_raise_for_status_raises_on_error(self) -> None:
        """Test raise_for_status raises on error status."""
        response = FakeHttpxResponse(status_code=500)

        with pytest.raises(Exception, match="HTTP 500"):
            response.raise_for_status()


class TestFakeHttpxClient:
    """Tests for FakeHttpxClient."""

    def test_get_returns_configured_response(self) -> None:
        """Test GET returns configured response."""
        response = FakeHttpxResponse(data={"result": "ok"})
        client = FakeHttpxClient(responses={"test-url": response})

        result = client.get("https://example.com/test-url")

        assert result.json() == {"result": "ok"}

    def test_get_returns_404_for_unmatched_url(self) -> None:
        """Test GET returns 404 for unmatched URL."""
        client = FakeHttpxClient()

        result = client.get("https://example.com/unknown")

        assert result.status_code == 404

    def test_get_returns_404_when_no_pattern_matches(self) -> None:
        """Test GET returns 404 when URL matches none of configured patterns."""
        response = FakeHttpxResponse(data={"found": True})
        client = FakeHttpxClient(responses={"different-url": response})

        result = client.get("https://example.com/unknown")

        assert result.status_code == 404

    def test_get_matches_url_substring(self) -> None:
        """Test GET matches URL substring."""
        response = FakeHttpxResponse(data={"found": True})
        client = FakeHttpxClient(responses={"contents/libs": response})

        result = client.get("https://api.github.com/repos/owner/repo/contents/libs")

        assert result.json() == {"found": True}

    def test_post_returns_404(self) -> None:
        """Test POST returns 404 (not used by GitHub scanner)."""
        client = FakeHttpxClient()

        result = client.post("https://example.com", headers={})

        assert result.status_code == 404

    def test_close_is_noop(self) -> None:
        """Test close() is a no-op."""
        client = FakeHttpxClient()

        # Should not raise
        client.close()
