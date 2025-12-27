"""Tests for _test_hooks module."""

from __future__ import annotations

from platform_codebase import FakeGitHubClient, GitHubClientProtocol
from platform_core.http_client import HttpxAsyncClient
from platform_core.testing import FakeHttpxAsyncClient, FakeHttpxResponse

from github_stats_api._test_hooks import (
    _default_build_client,
    _default_build_github_client,
    get_client_hook,
    get_github_client_hook,
    reset_client_hook,
    reset_github_client_hook,
    set_client_hook,
    set_github_client_hook,
)


class TestClientHooks:
    """Tests for client hook functions."""

    async def test_default_build_client_returns_async_client(self) -> None:
        """Test _default_build_client returns an async client."""
        client = _default_build_client(10.0)

        # Verify we can call aclose on the returned client
        # This proves it's a valid async client
        await client.aclose()

    def test_get_client_hook_returns_default(self) -> None:
        """Test get_client_hook returns default implementation."""
        reset_client_hook()
        hook = get_client_hook()
        assert hook == _default_build_client

    def test_set_and_reset_client_hook(self) -> None:
        """Test setting and resetting client hook."""
        fake_response = FakeHttpxResponse(200, {"data": {}})
        fake_client = FakeHttpxAsyncClient(fake_response)

        def fake_builder(timeout: float) -> HttpxAsyncClient:
            return fake_client

        set_client_hook(fake_builder)
        assert get_client_hook() == fake_builder

        reset_client_hook()
        assert get_client_hook() == _default_build_client


class TestGitHubClientHooks:
    """Tests for GitHub client hook functions."""

    def test_default_build_github_client_returns_client(self) -> None:
        """Test _default_build_github_client returns a GitHubClient."""
        client = _default_build_github_client("test_token")

        # Verify it's the right type by checking it matches GitHubClientProtocol
        # The callable methods exist and are callable
        assert callable(client.list_directory)
        assert callable(client.get_file_content)

    def test_get_github_client_hook_returns_default(self) -> None:
        """Test get_github_client_hook returns default implementation."""
        reset_github_client_hook()
        hook = get_github_client_hook()
        assert hook == _default_build_github_client

    def test_set_and_reset_github_client_hook(self) -> None:
        """Test setting and resetting GitHub client hook."""
        fake_client = FakeGitHubClient(
            directories={},
            files={},
            path_patterns={},
        )

        def fake_builder(token: str) -> GitHubClientProtocol:
            return fake_client

        set_github_client_hook(fake_builder)
        assert get_github_client_hook() == fake_builder

        reset_github_client_hook()
        assert get_github_client_hook() == _default_build_github_client
