"""Integration tests for production hooks.

These tests verify that the production hooks correctly delegate to the
underlying service implementations when called through the hooks layer.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from types import ModuleType

from platform_music.testing import (
    _prod_apple_http_get,
    _prod_redis_client,
    _prod_spotify_http_get,
    _prod_youtube_http_post,
)


class FakeRequest:
    """Fake urllib.request.Request."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.headers: dict[str, str] = {}
        self.data: bytes | None = None

    def add_header(self, name: str, value: str) -> None:
        self.headers[name] = value


class FakeResponse:
    """Fake HTTP response."""

    def __init__(self, content: bytes) -> None:
        self._content = content

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
    """Fake urlopen for HTTP tests."""

    def __init__(self, response: bytes) -> None:
        self._response = response
        self.seen_requests: list[FakeRequest] = []

    def __call__(self, req: FakeRequest, timeout: float) -> FakeResponse:
        self.seen_requests.append(req)
        return FakeResponse(self._response)


class FakeUrllibRequestModule(ModuleType):
    """Fake urllib.request module for testing."""

    Request: type[FakeRequest]
    urlopen: Callable[[FakeRequest, float], FakeResponse]

    def __init__(self, name: str, fake_opener: FakeUrlOpen) -> None:
        super().__init__(name)
        self.Request = FakeRequest
        self.urlopen = fake_opener


def _setup_fake_urllib(response: bytes) -> FakeUrlOpen:
    """Set up fake urllib.request module."""
    import sys

    fake_opener = FakeUrlOpen(response)
    fake_mod = FakeUrllibRequestModule("urllib.request", fake_opener)
    sys.modules["urllib.request"] = fake_mod
    return fake_opener


def _cleanup_fake_urllib() -> None:
    """Clean up fake urllib.request module."""
    import importlib
    import sys

    importlib.invalidate_caches()
    sys.modules.pop("urllib.request", None)


def test_prod_spotify_http_get_delegates() -> None:
    """Verify _prod_spotify_http_get delegates to service impl."""
    fake = _setup_fake_urllib(b'{"test": true}')

    result = _prod_spotify_http_get("https://api.spotify.com/test", "access_tok", 5.0)

    assert result == '{"test": true}'
    assert len(fake.seen_requests) == 1
    assert fake.seen_requests[0].headers["Authorization"] == "Bearer access_tok"

    _cleanup_fake_urllib()


def test_prod_apple_http_get_delegates() -> None:
    """Verify _prod_apple_http_get delegates to service impl."""
    fake = _setup_fake_urllib(b'{"data": []}')

    result = _prod_apple_http_get("https://api.music.apple.com/test", "dev_tok", "user_tok", 5.0)

    assert result == '{"data": []}'
    assert len(fake.seen_requests) == 1
    req = fake.seen_requests[0]
    assert req.headers["Authorization"] == "Bearer dev_tok"
    assert req.headers["Music-User-Token"] == "user_tok"

    _cleanup_fake_urllib()


def test_prod_youtube_http_post_delegates() -> None:
    """Verify _prod_youtube_http_post delegates to service impl."""
    fake = _setup_fake_urllib(b'{"items": []}')

    result = _prod_youtube_http_post(
        "https://music.youtube.com/test",
        "sapisid_val",
        "cookie=1",
        "https://music.youtube.com",
        5.0,
        '{"body": true}',
    )

    assert result == '{"items": []}'
    assert len(fake.seen_requests) == 1
    req = fake.seen_requests[0]
    assert req.headers["Cookie"] == "cookie=1"
    assert "SAPISIDHASH" in req.headers["Authorization"]

    _cleanup_fake_urllib()


def test_prod_redis_client_calls_redis_for_kv() -> None:
    """Verify _prod_redis_client delegates to redis_for_kv.

    We use a valid URL format but an unreachable host. The redis client
    will be created successfully but fail on actual connection.
    """
    # Use a valid redis URL format pointing to localhost
    # The client is created lazily, so this won't actually connect
    client = _prod_redis_client("redis://localhost:6379/0")
    # Verify we got a Redis client back - check the module path
    assert "redis" in client.__class__.__module__.lower()
