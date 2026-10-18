"""Tests for platform_email.auth.common module."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_email.auth.common import is_token_expired
from platform_email.fake_hooks import (
    make_fake_current_time,
)
from platform_email.testing import (
    hooks,
    reset_hooks,
)
from platform_email.types import OAuthTokens, TokenType


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


def _make_tokens(expires_at: int) -> OAuthTokens:
    """Create tokens with specified expiry."""
    return OAuthTokens(
        access_token="access",
        refresh_token="refresh",
        expires_at=expires_at,
        token_type=TokenType.BEARER,
    )


class TestIsTokenExpired:
    """Tests for is_token_expired function."""

    def test_returns_false_for_future_expiry(self) -> None:
        """Test that token expiring in the future is not expired."""
        tokens = _make_tokens(expires_at=2000)
        hooks.current_time = make_fake_current_time(1000)

        result = is_token_expired(tokens)

        assert result is False

    def test_returns_true_for_past_expiry(self) -> None:
        """Test that token expiring in the past is expired."""
        tokens = _make_tokens(expires_at=500)
        hooks.current_time = make_fake_current_time(1000)

        result = is_token_expired(tokens)

        assert result is True

    def test_returns_true_within_buffer(self) -> None:
        """Test that token expiring within buffer is considered expired."""
        # Token expires at 1050, current time 1000, buffer 60 -> expired
        tokens = _make_tokens(expires_at=1050)
        hooks.current_time = make_fake_current_time(1000)

        result = is_token_expired(tokens, buffer_seconds=60)

        assert result is True

    def test_returns_false_outside_buffer(self) -> None:
        """Test that token expiring outside buffer is not expired."""
        # Token expires at 1100, current time 1000, buffer 60 -> not expired
        tokens = _make_tokens(expires_at=1100)
        hooks.current_time = make_fake_current_time(1000)

        result = is_token_expired(tokens, buffer_seconds=60)

        assert result is False
