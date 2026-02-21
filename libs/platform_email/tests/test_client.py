"""Tests for platform_email.client module."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_email.client import gmail_email_client, outlook_email_client
from platform_email.testing import hooks, reset_hooks
from platform_email.types import OAuthTokens


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


def _make_tokens() -> OAuthTokens:
    """Create test tokens."""
    return OAuthTokens(
        access_token="test_access_token",
        refresh_token="test_refresh_token",
        expires_at=9999999999,
        token_type="Bearer",
    )


class TestOutlookEmailClient:
    """Tests for outlook_email_client factory."""

    def test_client_uses_correct_auth_header(self) -> None:
        """Test that client uses correct authorization header."""
        captured_headers: dict[str, str] = {}

        def capture_get(url: str, headers: dict[str, str]) -> str:
            captured_headers.update(headers)
            return '{"value": []}'

        hooks.http_get = capture_get

        tokens = _make_tokens()
        client = outlook_email_client(tokens=tokens)
        client.list_folders()

        assert captured_headers["Authorization"] == "Bearer test_access_token"
        assert captured_headers["Content-Type"] == "application/json"

    def test_factory_returns_protocol_compatible_client(self) -> None:
        """Test that factory returns protocol-compatible client."""

        def fake_get(url: str, headers: dict[str, str]) -> str:
            return '{"value": []}'

        hooks.http_get = fake_get

        tokens = _make_tokens()
        client = outlook_email_client(tokens=tokens)
        folders = client.list_folders()
        assert folders == ()


class TestGmailEmailClient:
    """Tests for gmail_email_client factory."""

    def test_client_uses_correct_auth_header(self) -> None:
        """Test that client uses correct authorization header."""
        captured_headers: dict[str, str] = {}

        def capture_get(url: str, headers: dict[str, str]) -> str:
            captured_headers.update(headers)
            return '{"labels": []}'

        hooks.http_get = capture_get

        tokens = _make_tokens()
        client = gmail_email_client(tokens=tokens)
        client.list_folders()

        assert captured_headers["Authorization"] == "Bearer test_access_token"
        assert captured_headers["Content-Type"] == "application/json"

    def test_factory_returns_protocol_compatible_client(self) -> None:
        """Test that factory returns protocol-compatible client."""

        def fake_get(url: str, headers: dict[str, str]) -> str:
            return '{"labels": []}'

        hooks.http_get = fake_get

        tokens = _make_tokens()
        client = gmail_email_client(tokens=tokens)
        folders = client.list_folders()
        assert folders == ()
