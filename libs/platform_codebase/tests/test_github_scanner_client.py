"""Tests for github scanner: GitHubClient."""

from __future__ import annotations

import base64

from platform_core.json_utils import JSONValue

from platform_codebase.github_scanner import (
    GitHubClient,
)
from platform_codebase.testing import (
    FakeHttpxClient,
    FakeHttpxResponse,
)


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
