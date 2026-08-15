"""Tests for github scanner: FakeGitHubClient."""

from __future__ import annotations

import pytest

from platform_codebase.testing import (
    FakeGitHubClient,
    FakeHttpxClient,
    FakeHttpxResponse,
)


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
