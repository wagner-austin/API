"""Tests for production HTTP implementation functions.

These tests exercise the http_get_impl and http_post_impl functions directly
by setting up fake HTTP request/response objects that satisfy the Protocol.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from io import BytesIO
from types import ModuleType

from platform_music.services.apple import http_get_impl as apple_http_get_impl
from platform_music.services.spotify import http_get_impl as spotify_http_get_impl
from platform_music.services.youtube import http_post_impl as youtube_http_post_impl
from platform_music.services.youtube import sapisidhash


class FakeRequest:
    """Fake urllib.request.Request for testing."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.headers: dict[str, str] = {}
        self.data: bytes | None = None

    def add_header(self, name: str, value: str) -> None:
        self.headers[name] = value


class FakeResponse:
    """Fake HTTP response context manager."""

    def __init__(self, content: bytes) -> None:
        self._content = content
        self._stream = BytesIO(content)

    def read(self) -> bytes:
        return self._content

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        pass


class FakeUrlOpen:
    """Fake urllib.request.urlopen for testing."""

    def __init__(self, response_content: bytes) -> None:
        self._response_content = response_content
        self.seen_requests: list[FakeRequest] = []

    def __call__(self, req: FakeRequest, timeout: float) -> FakeResponse:
        self.seen_requests.append(req)
        return FakeResponse(self._response_content)


class FakeUrllibRequestModule(ModuleType):
    """Fake urllib.request module for testing HTTP implementations."""

    Request: type[FakeRequest]
    urlopen: Callable[[FakeRequest, float], FakeResponse]

    def __init__(self, name: str, fake_opener: FakeUrlOpen) -> None:
        super().__init__(name)
        self.Request = FakeRequest
        self.urlopen = fake_opener


def test_sapisidhash_format() -> None:
    """Verify SAPISIDHASH header format."""
    result = sapisidhash("test_sapisid", "https://music.youtube.com", 1700000000)
    assert result.startswith("SAPISIDHASH 1700000000_")
    # SHA-1 hash is 40 hex characters
    hash_part = result.split("_")[1]
    assert len(hash_part) == 40


def test_sapisidhash_deterministic() -> None:
    """Verify SAPISIDHASH produces consistent results."""
    r1 = sapisidhash("sid", "https://example.com", 12345)
    r2 = sapisidhash("sid", "https://example.com", 12345)
    assert r1 == r2


def test_sapisidhash_different_inputs() -> None:
    """Verify SAPISIDHASH produces different results for different inputs."""
    r1 = sapisidhash("sid1", "https://example.com", 12345)
    r2 = sapisidhash("sid2", "https://example.com", 12345)
    assert r1 != r2


def test_spotify_http_get_impl_with_fake() -> None:
    """Test spotify http_get_impl with injected fake urlopen."""
    response_json = b'{"items": []}'
    fake_opener = FakeUrlOpen(response_json)

    # Replace urllib.request module temporarily
    import sys

    fake_request_mod = FakeUrllibRequestModule("urllib.request", fake_opener)
    sys.modules["urllib.request"] = fake_request_mod

    result = spotify_http_get_impl(
        "https://api.spotify.com/v1/test",
        access_token="test_token",
        timeout=10.0,
    )

    assert result == '{"items": []}'
    assert len(fake_opener.seen_requests) == 1
    req = fake_opener.seen_requests[0]
    assert req.headers["Authorization"] == "Bearer test_token"

    # Restore real module
    import importlib

    importlib.invalidate_caches()
    sys.modules.pop("urllib.request", None)


def test_apple_http_get_impl_with_fake() -> None:
    """Test apple http_get_impl with injected fake urlopen."""
    response_json = b'{"data": []}'
    fake_opener = FakeUrlOpen(response_json)

    import sys

    fake_request_mod = FakeUrllibRequestModule("urllib.request", fake_opener)
    sys.modules["urllib.request"] = fake_request_mod

    result = apple_http_get_impl(
        "https://api.music.apple.com/v1/test",
        developer_token="dev_token",
        user_token="user_token",
        timeout=10.0,
    )

    assert result == '{"data": []}'
    assert len(fake_opener.seen_requests) == 1
    req = fake_opener.seen_requests[0]
    assert req.headers["Authorization"] == "Bearer dev_token"
    assert req.headers["Music-User-Token"] == "user_token"

    import importlib

    importlib.invalidate_caches()
    sys.modules.pop("urllib.request", None)


def test_youtube_http_post_impl_with_fake() -> None:
    """Test youtube http_post_impl with injected fake urlopen."""
    response_json = b'{"items": []}'
    fake_opener = FakeUrlOpen(response_json)

    import sys

    fake_request_mod = FakeUrllibRequestModule("urllib.request", fake_opener)
    sys.modules["urllib.request"] = fake_request_mod

    result = youtube_http_post_impl(
        "https://music.youtube.com/youtubei/v1/test",
        sapisid="test_sapisid",
        cookies="SID=abc",
        origin="https://music.youtube.com",
        timeout=10.0,
        body='{"context": {}}',
    )

    assert result == '{"items": []}'
    assert len(fake_opener.seen_requests) == 1
    req = fake_opener.seen_requests[0]
    assert req.headers["Cookie"] == "SID=abc"
    assert req.headers["Origin"] == "https://music.youtube.com"
    assert req.headers["Content-Type"] == "application/json"
    assert req.headers["X-YouTube-Client-Name"] == "67"
    assert "SAPISIDHASH" in req.headers["Authorization"]
    assert req.data == b'{"context": {}}'

    import importlib

    importlib.invalidate_caches()
    sys.modules.pop("urllib.request", None)
