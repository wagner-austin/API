"""Tests for platform_email.auth.common module."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_email.auth.common import (
    generate_code_challenge,
    generate_code_verifier,
    generate_state,
    is_token_expired,
)
from platform_email.fake_hooks import (
    make_fake_current_time,
)
from platform_email.testing import (
    hooks,
    reset_hooks,
)
from platform_email.types import OAuthTokens


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
        token_type="Bearer",
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


class TestGenerateCodeVerifier:
    """Tests for generate_code_verifier function."""

    def test_generates_valid_verifier(self) -> None:
        """Test that verifier meets length requirements."""
        verifier = generate_code_verifier()

        # PKCE verifiers must be 43-128 chars
        assert len(verifier) >= 43
        assert len(verifier) <= 128

    def test_generates_unique_verifiers(self) -> None:
        """Test that different calls generate different verifiers."""
        verifier1 = generate_code_verifier()
        verifier2 = generate_code_verifier()

        assert verifier1 != verifier2


class TestGenerateCodeChallenge:
    """Tests for generate_code_challenge function."""

    def test_generates_challenge_from_verifier(self) -> None:
        """Test that challenge is generated from verifier."""
        verifier = "test_verifier_string_at_least_43_characters_long_here"
        challenge = generate_code_challenge(verifier)

        # Challenge is base64url encoded SHA256 hash (43 chars without padding)
        assert len(challenge) >= 32

    def test_same_verifier_same_challenge(self) -> None:
        """Test that same verifier produces same challenge."""
        verifier = generate_code_verifier()
        challenge1 = generate_code_challenge(verifier)
        challenge2 = generate_code_challenge(verifier)

        assert challenge1 == challenge2


class TestGenerateState:
    """Tests for generate_state function."""

    def test_generates_valid_state(self) -> None:
        """Test that state is generated with sufficient entropy."""
        state = generate_state()

        # State should be reasonably long for security
        assert len(state) >= 16

    def test_generates_unique_states(self) -> None:
        """Test that different calls generate different states."""
        state1 = generate_state()
        state2 = generate_state()

        assert state1 != state2
