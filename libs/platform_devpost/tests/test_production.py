"""Tests for platform_devpost._production module."""

from __future__ import annotations

import httpx
from platform_core.http_client import HttpxClient, SyncTransport, build_client

from platform_devpost._production import (
    _HttpDevpostApi,
    _reset_http_client_builder,
    _set_http_client_builder,
    create_devpost_api,
    make_devpost_client,
)
from platform_devpost.testing import FakeDevpostApi, hooks, make_fake_hackathon


class FakeHttpTransport(httpx.BaseTransport):
    """Fake HTTP transport for testing.

    Extends httpx.BaseTransport to properly implement the transport interface.
    """

    def __init__(self, response_text: str, status_code: int = 200) -> None:
        """Initialize fake transport.

        Args:
            response_text: Text to return in response.
            status_code: HTTP status code to return.
        """
        self._response_text = response_text
        self._status_code = status_code
        self._requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Handle HTTP request.

        Args:
            request: The httpx request object.

        Returns:
            httpx.Response with configured response.
        """
        self._requests.append(request)
        return httpx.Response(
            status_code=self._status_code,
            content=self._response_text.encode(),
            request=request,
        )


class TestHttpDevpostApi:
    """Tests for _HttpDevpostApi class."""

    def test_fetch_hackathons_returns_response(self) -> None:
        """Test fetch_hackathons parses and returns response."""
        response_json = """{
            "hackathons": [
                {
                    "id": 1,
                    "title": "Test Hackathon",
                    "url": "https://test.devpost.com/",
                    "thumbnail_url": "https://example.com/thumb.jpg",
                    "organization_name": "Test Org",
                    "displayed_location": {"icon": "globe", "location": "Online"},
                    "open_state": "open",
                    "time_left_to_submission": "5 days",
                    "submission_period_dates": "Jan 1 - Feb 1",
                    "themes": [],
                    "prize_amount": "$1,000",
                    "registrations_count": 100,
                    "featured": false,
                    "winners_announced": false,
                    "invite_only": false
                }
            ],
            "meta": {
                "total_count": 1,
                "per_page": 10
            }
        }"""

        fake_transport = FakeHttpTransport(response_json)

        def test_builder(timeout: float, transport: SyncTransport | None) -> HttpxClient:
            return build_client(timeout, transport=fake_transport)

        _set_http_client_builder(test_builder)

        try:
            api = _HttpDevpostApi()
            result = api.fetch_hackathons()

            assert len(result.hackathons) == 1
            assert result.hackathons[0].id == 1
            assert result.hackathons[0].title == "Test Hackathon"
            assert result.meta.total_count == 1
            assert result.meta.per_page == 10
        finally:
            _reset_http_client_builder()

    def test_fetch_hackathons_with_search(self) -> None:
        """Test fetch_hackathons with search parameter."""
        response_json = """{
            "hackathons": [],
            "meta": {"total_count": 0, "per_page": 10}
        }"""

        fake_transport = FakeHttpTransport(response_json)

        def test_builder(timeout: float, transport: SyncTransport | None) -> HttpxClient:
            return build_client(timeout, transport=fake_transport)

        _set_http_client_builder(test_builder)

        try:
            api = _HttpDevpostApi()
            result = api.fetch_hackathons(search="AI", page=2)

            assert result.meta.total_count == 0
            assert result.hackathons == ()
        finally:
            _reset_http_client_builder()


class TestCreateDevpostApi:
    """Tests for create_devpost_api function."""

    def test_returns_callable_api(self) -> None:
        """Test create_devpost_api returns API that can fetch hackathons."""
        response_json = """{
            "hackathons": [],
            "meta": {"total_count": 0, "per_page": 10}
        }"""

        fake_transport = FakeHttpTransport(response_json)

        def test_builder(timeout: float, transport: SyncTransport | None) -> HttpxClient:
            return build_client(timeout, transport=fake_transport)

        _set_http_client_builder(test_builder)

        try:
            api = create_devpost_api()
            result = api.fetch_hackathons()
            assert result.meta.total_count == 0
        finally:
            _reset_http_client_builder()


class TestMakeDevpostClient:
    """Tests for make_devpost_client function."""

    def test_returns_callable_client(self) -> None:
        """Test make_devpost_client returns client that can list hackathons."""
        h = make_fake_hackathon(id=99, title="Test")
        fake_api = FakeDevpostApi(hackathons=(h,))
        hooks.devpost_api_factory = lambda: fake_api

        client = make_devpost_client()
        result = client.list_hackathons()

        assert len(result) == 1
        assert result[0].id == 99
