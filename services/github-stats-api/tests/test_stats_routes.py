from __future__ import annotations

from collections.abc import Generator

import httpx
import pytest
from platform_core.http_client import HttpxAsyncClient
from platform_core.json_utils import JSONValue
from platform_core.testing import FakeHttpxAsyncClient, FakeHttpxResponse

from github_stats_api._test_hooks import reset_client_hook, set_client_hook
from github_stats_api.api.main import create_app
from github_stats_api.settings import Settings


def _make_test_settings() -> Settings:
    """Test settings fixture."""
    return {
        "github_token": "test-token",
        "cache_ttl_seconds": 60,
        "port": 8000,
    }


def _make_fake_user_response() -> dict[str, JSONValue]:
    """Create a fake GitHub GraphQL response for user stats."""
    return {
        "data": {
            "user": {
                "login": "testuser",
                "name": "Test User",
                "contributionsCollection": {
                    "totalCommitContributions": 100,
                    "restrictedContributionsCount": 10,
                },
                "pullRequests": {"totalCount": 20},
                "openIssues": {"totalCount": 5},
                "closedIssues": {"totalCount": 5},
                "repositories": {
                    "nodes": [
                        {"stargazerCount": 50},
                        {"stargazerCount": 30},
                    ]
                },
                "repositoriesContributedTo": {"totalCount": 15},
            }
        }
    }


def _make_fake_langs_response() -> dict[str, JSONValue]:
    """Create a fake GitHub GraphQL response for languages."""
    return {
        "data": {
            "user": {
                "repositories": {
                    "nodes": [
                        {
                            "languages": {
                                "edges": [
                                    {
                                        "size": 50000,
                                        "node": {"name": "Python", "color": "#3572A5"},
                                    },
                                    {
                                        "size": 30000,
                                        "node": {"name": "TypeScript", "color": "#3178c6"},
                                    },
                                ]
                            }
                        }
                    ]
                }
            }
        }
    }


class TestStatsEndpoint:
    """Tests for /api stats endpoint."""

    @pytest.fixture(autouse=True)
    def _reset_hooks(self) -> Generator[None, None, None]:
        """Reset hooks after each test."""
        yield
        reset_client_hook()

    async def test_get_stats_returns_svg(self) -> None:
        """Test /api endpoint returns SVG card."""
        fake_response = FakeHttpxResponse(200, _make_fake_user_response())

        def build_fake_client(timeout: float) -> HttpxAsyncClient:
            return FakeHttpxAsyncClient(fake_response)

        set_client_hook(build_fake_client)

        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api?username=testuser")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/svg+xml"
        assert "<svg" in response.text
        assert "testuser" in response.text or "Test User" in response.text

    async def test_get_stats_with_theme(self) -> None:
        """Test /api endpoint with theme parameter."""
        fake_response = FakeHttpxResponse(200, _make_fake_user_response())

        def build_fake_client(timeout: float) -> HttpxAsyncClient:
            return FakeHttpxAsyncClient(fake_response)

        set_client_hook(build_fake_client)

        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api?username=testuser&theme=dark")

        assert response.status_code == 200
        assert "<svg" in response.text

    async def test_get_stats_with_hide_border(self) -> None:
        """Test /api endpoint with hide_border parameter."""
        fake_response = FakeHttpxResponse(200, _make_fake_user_response())

        def build_fake_client(timeout: float) -> HttpxAsyncClient:
            return FakeHttpxAsyncClient(fake_response)

        set_client_hook(build_fake_client)

        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api?username=testuser&hide_border=true")

        assert response.status_code == 200
        assert "<svg" in response.text

    async def test_get_stats_with_hide(self) -> None:
        """Test /api endpoint with hide parameter."""
        fake_response = FakeHttpxResponse(200, _make_fake_user_response())

        def build_fake_client(timeout: float) -> HttpxAsyncClient:
            return FakeHttpxAsyncClient(fake_response)

        set_client_hook(build_fake_client)

        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api?username=testuser&hide=stars,prs")

        assert response.status_code == 200
        assert "<svg" in response.text

    async def test_get_stats_missing_username_returns_error(self) -> None:
        """Test /api endpoint without username returns 400."""
        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api")

        assert response.status_code == 400

    async def test_get_stats_cache_control_header(self) -> None:
        """Test /api endpoint sets cache-control header."""
        fake_response = FakeHttpxResponse(200, _make_fake_user_response())

        def build_fake_client(timeout: float) -> HttpxAsyncClient:
            return FakeHttpxAsyncClient(fake_response)

        set_client_hook(build_fake_client)

        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api?username=testuser")

        assert "Cache-Control" in response.headers
        assert "max-age=60" in response.headers["Cache-Control"]


class TestTopLangsEndpoint:
    """Tests for /api/top-langs endpoint."""

    @pytest.fixture(autouse=True)
    def _reset_hooks(self) -> Generator[None, None, None]:
        """Reset hooks after each test."""
        yield
        reset_client_hook()

    async def test_get_top_langs_returns_svg(self) -> None:
        """Test /api/top-langs endpoint returns SVG card."""
        fake_response = FakeHttpxResponse(200, _make_fake_langs_response())

        def build_fake_client(timeout: float) -> HttpxAsyncClient:
            return FakeHttpxAsyncClient(fake_response)

        set_client_hook(build_fake_client)

        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/top-langs?username=testuser")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/svg+xml"
        assert "<svg" in response.text

    async def test_get_top_langs_with_compact_layout(self) -> None:
        """Test /api/top-langs endpoint with compact layout."""
        fake_response = FakeHttpxResponse(200, _make_fake_langs_response())

        def build_fake_client(timeout: float) -> HttpxAsyncClient:
            return FakeHttpxAsyncClient(fake_response)

        set_client_hook(build_fake_client)

        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/top-langs?username=testuser&layout=compact")

        assert response.status_code == 200
        assert "<svg" in response.text

    async def test_get_top_langs_with_hide(self) -> None:
        """Test /api/top-langs endpoint filters hidden languages."""
        fake_response = FakeHttpxResponse(200, _make_fake_langs_response())

        def build_fake_client(timeout: float) -> HttpxAsyncClient:
            return FakeHttpxAsyncClient(fake_response)

        set_client_hook(build_fake_client)

        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/top-langs?username=testuser&hide=python")

        assert response.status_code == 200
        assert "<svg" in response.text
        assert "Python" not in response.text

    async def test_get_top_langs_with_langs_count(self) -> None:
        """Test /api/top-langs endpoint with langs_count."""
        fake_response = FakeHttpxResponse(200, _make_fake_langs_response())

        def build_fake_client(timeout: float) -> HttpxAsyncClient:
            return FakeHttpxAsyncClient(fake_response)

        set_client_hook(build_fake_client)

        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/top-langs?username=testuser&langs_count=1")

        assert response.status_code == 200
        assert "<svg" in response.text

    async def test_get_top_langs_missing_username_returns_error(self) -> None:
        """Test /api/top-langs endpoint without username returns 400."""
        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/top-langs")

        assert response.status_code == 400

    async def test_get_top_langs_cache_control_header(self) -> None:
        """Test /api/top-langs endpoint sets cache-control header."""
        fake_response = FakeHttpxResponse(200, _make_fake_langs_response())

        def build_fake_client(timeout: float) -> HttpxAsyncClient:
            return FakeHttpxAsyncClient(fake_response)

        set_client_hook(build_fake_client)

        test_settings = _make_test_settings()
        app = create_app(test_settings)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/top-langs?username=testuser")

        assert "Cache-Control" in response.headers
        assert "max-age=60" in response.headers["Cache-Control"]
