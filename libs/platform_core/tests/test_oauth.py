"""Tests for platform_core.oauth module."""

from __future__ import annotations

import pytest

from platform_core.errors import AppError, OAuthErrorCode
from platform_core.oauth import (
    build_authorization_url,
    exchange_authorization_code,
    generate_code_challenge,
    generate_code_verifier,
    generate_state,
    is_token_expired,
    refresh_access_token,
)
from platform_core.oauth_testing import (
    make_error_response_json,
    make_fake_current_time,
    make_fake_http_post,
    make_raising_http_post,
    make_test_credentials,
    make_test_tokens,
    make_token_response_json,
)


class TestGenerateCodeVerifier:
    def test_returns_url_safe_string_with_expected_length(self) -> None:
        verifier = generate_code_verifier()
        # Default length=64 bytes produces ~86 character base64url string
        assert len(verifier) >= 80
        # URL-safe base64 only contains: letters, digits, hyphen, underscore
        for char in verifier:
            assert char.isalnum() or char in "-_"

    def test_generates_unique_values(self) -> None:
        verifiers = [generate_code_verifier() for _ in range(10)]
        assert len(set(verifiers)) == 10

    def test_custom_length(self) -> None:
        short_verifier = generate_code_verifier(length=16)
        long_verifier = generate_code_verifier(length=128)
        # Length after base64 encoding varies but longer input = longer output
        assert len(long_verifier) > len(short_verifier)


class TestGenerateCodeChallenge:
    def test_deterministic(self) -> None:
        verifier = "test_verifier_12345"
        c1 = generate_code_challenge(verifier)
        c2 = generate_code_challenge(verifier)
        assert c1 == c2

    def test_is_url_safe(self) -> None:
        verifier = "test_verifier"
        challenge = generate_code_challenge(verifier)
        # URL-safe base64 without padding
        assert "=" not in challenge
        for char in challenge:
            assert char.isalnum() or char in "-_"

    def test_different_verifiers_produce_different_challenges(self) -> None:
        c1 = generate_code_challenge("verifier_1")
        c2 = generate_code_challenge("verifier_2")
        assert c1 != c2

    def test_produces_expected_length(self) -> None:
        # SHA-256 produces 32 bytes, base64url encoding produces 43 chars (without padding)
        verifier = "any_verifier"
        challenge = generate_code_challenge(verifier)
        assert len(challenge) == 43


class TestGenerateState:
    def test_returns_url_safe_string_with_expected_length(self) -> None:
        state = generate_state()
        # 32 bytes produces ~43 character base64url string
        assert len(state) >= 40
        # URL-safe base64 only contains: letters, digits, hyphen, underscore
        for char in state:
            assert char.isalnum() or char in "-_"

    def test_generates_unique_values(self) -> None:
        states = [generate_state() for _ in range(10)]
        assert len(set(states)) == 10


class TestIsTokenExpired:
    def test_not_expired(self) -> None:
        current_time = 1735200000
        tokens = make_test_tokens(expires_at=current_time + 3600)

        assert is_token_expired(tokens, current_time) is False

    def test_expired(self) -> None:
        current_time = 1735200000
        tokens = make_test_tokens(expires_at=current_time - 100)

        assert is_token_expired(tokens, current_time) is True

    def test_expires_within_default_buffer(self) -> None:
        current_time = 1735200000
        # Expires in 30 seconds, but default buffer is 60
        tokens = make_test_tokens(expires_at=current_time + 30)

        assert is_token_expired(tokens, current_time) is True

    def test_not_expired_outside_buffer(self) -> None:
        current_time = 1735200000
        # Expires in 61 seconds, default buffer is 60
        tokens = make_test_tokens(expires_at=current_time + 61)

        assert is_token_expired(tokens, current_time) is False

    def test_custom_buffer(self) -> None:
        current_time = 1735200000
        # Expires in 30 seconds
        tokens = make_test_tokens(expires_at=current_time + 30)

        # With smaller buffer, not expired
        assert is_token_expired(tokens, current_time, buffer_seconds=10) is False
        # With larger buffer, expired
        assert is_token_expired(tokens, current_time, buffer_seconds=60) is True

    def test_exactly_at_expiry(self) -> None:
        current_time = 1735200000
        tokens = make_test_tokens(expires_at=current_time)

        assert is_token_expired(tokens, current_time, buffer_seconds=0) is True


class TestBuildAuthorizationUrl:
    def test_builds_url_with_all_params(self) -> None:
        url = build_authorization_url(
            "https://auth.example.com/authorize",
            "client_123",
            "http://localhost:8080/callback",
            code_challenge="challenge_abc",
            state="state_xyz",
            scopes=("scope1", "scope2"),
        )

        assert "auth.example.com/authorize" in url
        assert "client_id=client_123" in url
        assert "redirect_uri=http" in url
        assert "code_challenge=challenge_abc" in url
        assert "state=state_xyz" in url
        assert "response_type=code" in url
        assert "access_type=offline" in url
        assert "prompt=consent" in url
        assert "code_challenge_method=S256" in url
        assert "scope1" in url
        assert "scope2" in url

    def test_custom_access_type(self) -> None:
        url = build_authorization_url(
            "https://auth.example.com/authorize",
            "client_123",
            "http://localhost",
            code_challenge="challenge",
            state="state",
            scopes=("scope1",),
            access_type="online",
        )

        assert "access_type=online" in url

    def test_custom_prompt(self) -> None:
        url = build_authorization_url(
            "https://auth.example.com/authorize",
            "client_123",
            "http://localhost",
            code_challenge="challenge",
            state="state",
            scopes=("scope1",),
            prompt="select_account",
        )

        assert "prompt=select_account" in url

    def test_scopes_joined_with_space(self) -> None:
        url = build_authorization_url(
            "https://auth.example.com/authorize",
            "client_123",
            "http://localhost",
            code_challenge="challenge",
            state="state",
            scopes=("read", "write", "admin"),
        )

        # Spaces are URL-encoded as %20 or +
        assert "read" in url
        assert "write" in url
        assert "admin" in url


class TestExchangeAuthorizationCode:
    def test_successful_exchange(self) -> None:
        response_json = make_token_response_json(
            access_token="new_access",
            refresh_token="new_refresh",
            expires_in=3600,
        )
        http_post = make_fake_http_post(response_json)
        current_time = 1735200000
        credentials = make_test_credentials()

        tokens = exchange_authorization_code(
            "https://token.example.com/token",
            credentials,
            "auth_code",
            "code_verifier",
            http_post=http_post,
            current_time=current_time,
        )

        assert tokens["access_token"] == "new_access"
        assert tokens["refresh_token"] == "new_refresh"
        assert tokens["expires_at"] == current_time + 3600
        assert tokens["token_type"] == "Bearer"

    def test_connection_error(self) -> None:
        http_post = make_raising_http_post(ConnectionError("Network error"))
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            exchange_authorization_code(
                "https://token.example.com/token",
                credentials,
                "auth_code",
                "code_verifier",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.TOKEN_EXCHANGE_FAILED
        assert "Failed to exchange" in error.message
        assert error.http_status == 401

    def test_os_error(self) -> None:
        http_post = make_raising_http_post(OSError("Socket error"))
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            exchange_authorization_code(
                "https://token.example.com/token",
                credentials,
                "auth_code",
                "code_verifier",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.TOKEN_EXCHANGE_FAILED

    def test_invalid_json_response(self) -> None:
        http_post = make_fake_http_post("not valid json")
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            exchange_authorization_code(
                "https://token.example.com/token",
                credentials,
                "auth_code",
                "code_verifier",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.TOKEN_EXCHANGE_FAILED
        assert "Invalid JSON" in error.message

    def test_response_not_object(self) -> None:
        http_post = make_fake_http_post("[]")
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            exchange_authorization_code(
                "https://token.example.com/token",
                credentials,
                "auth_code",
                "code_verifier",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.TOKEN_EXCHANGE_FAILED
        assert "Expected JSON object" in error.message

    def test_oauth_error_response(self) -> None:
        response_json = make_error_response_json(
            error="invalid_grant",
            error_description="Authorization code expired",
        )
        http_post = make_fake_http_post(response_json)
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            exchange_authorization_code(
                "https://token.example.com/token",
                credentials,
                "auth_code",
                "code_verifier",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.TOKEN_EXCHANGE_FAILED
        assert "Authorization code expired" in error.message

    def test_oauth_error_without_description(self) -> None:
        response_json = make_error_response_json(error="invalid_request")
        http_post = make_fake_http_post(response_json)
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            exchange_authorization_code(
                "https://token.example.com/token",
                credentials,
                "auth_code",
                "code_verifier",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.TOKEN_EXCHANGE_FAILED
        assert "invalid_request" in error.message

    def test_no_refresh_token_in_response(self) -> None:
        response_json = make_token_response_json(refresh_token=None)
        http_post = make_fake_http_post(response_json)
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            exchange_authorization_code(
                "https://token.example.com/token",
                credentials,
                "auth_code",
                "code_verifier",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.MISSING_REFRESH_TOKEN
        assert "No refresh token" in error.message


class TestRefreshAccessToken:
    def test_successful_refresh(self) -> None:
        response_json = make_token_response_json(
            access_token="new_access",
            refresh_token=None,  # Refresh responses don't include refresh token
            expires_in=3600,
        )
        http_post = make_fake_http_post(response_json)
        current_time = 1735200000
        credentials = make_test_credentials()

        tokens = refresh_access_token(
            "https://token.example.com/token",
            credentials,
            "original_refresh_token",
            http_post=http_post,
            current_time=current_time,
        )

        assert tokens["access_token"] == "new_access"
        assert tokens["refresh_token"] == "original_refresh_token"  # Preserved
        assert tokens["expires_at"] == current_time + 3600
        assert tokens["token_type"] == "Bearer"

    def test_connection_error(self) -> None:
        http_post = make_raising_http_post(ConnectionError("Network error"))
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            refresh_access_token(
                "https://token.example.com/token",
                credentials,
                "refresh_token",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.TOKEN_REFRESH_FAILED
        assert "Failed to refresh" in error.message

    def test_os_error(self) -> None:
        http_post = make_raising_http_post(OSError("Socket error"))
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            refresh_access_token(
                "https://token.example.com/token",
                credentials,
                "refresh_token",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.TOKEN_REFRESH_FAILED

    def test_invalid_json_response(self) -> None:
        http_post = make_fake_http_post("{not json")
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            refresh_access_token(
                "https://token.example.com/token",
                credentials,
                "refresh_token",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.TOKEN_REFRESH_FAILED
        assert "Invalid JSON" in error.message

    def test_response_not_object(self) -> None:
        http_post = make_fake_http_post('"just a string"')
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            refresh_access_token(
                "https://token.example.com/token",
                credentials,
                "refresh_token",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.TOKEN_REFRESH_FAILED
        assert "Expected JSON object" in error.message

    def test_oauth_error_response(self) -> None:
        response_json = make_error_response_json(
            error="invalid_grant",
            error_description="Token has been revoked",
        )
        http_post = make_fake_http_post(response_json)
        credentials = make_test_credentials()

        with pytest.raises(AppError) as exc_info:
            refresh_access_token(
                "https://token.example.com/token",
                credentials,
                "refresh_token",
                http_post=http_post,
                current_time=1735200000,
            )

        error: AppError[OAuthErrorCode] = exc_info.value
        assert error.code == OAuthErrorCode.TOKEN_REFRESH_FAILED
        assert "Token has been revoked" in error.message


class TestOAuthTestingHelpers:
    """Test that the oauth_testing helpers work correctly."""

    def test_make_test_tokens_defaults(self) -> None:
        tokens = make_test_tokens()
        assert tokens["access_token"] == "test_access_token"
        assert tokens["refresh_token"] == "test_refresh_token"
        assert tokens["token_type"] == "Bearer"

    def test_make_test_tokens_expired(self) -> None:
        current_time = 1735200000
        tokens = make_test_tokens(expired=True, current_time=current_time)
        assert tokens["expires_at"] < current_time

    def test_make_fake_current_time(self) -> None:
        time_hook = make_fake_current_time(1735200000)
        assert time_hook() == 1735200000
        assert time_hook() == 1735200000

    def test_make_fake_http_post(self) -> None:
        http_post = make_fake_http_post('{"key": "value"}')
        result = http_post("http://example.com", {}, "body")
        assert result == '{"key": "value"}'

    def test_make_raising_http_post(self) -> None:
        http_post = make_raising_http_post(ConnectionError("Test error"))
        with pytest.raises(ConnectionError):
            http_post("http://example.com", {}, "body")
