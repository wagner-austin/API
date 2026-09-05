"""Token validity, PKCE, authorize flows."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, CalendarErrorCode
from platform_core.json_utils import dump_json_str
from platform_core.oauth import generate_code_challenge, generate_code_verifier
from platform_core.oauth_types import OAuthCredentials, OAuthTokens

from platform_calendar.auth import (
    authorize,
    exchange_code_for_tokens,
    get_valid_tokens,
    is_token_expired,
    load_or_authorize,
    refresh_access_token,
)
from platform_calendar.fakes import (
    make_fake_console,
    make_fake_credentials,
    make_fake_current_time,
    make_fake_http_send,
    make_fake_no_tokens,
    make_fake_tokens,
    make_raising_http_send,
)
from platform_calendar.testing import (
    hooks,
)


class TestIsTokenExpired:
    def test_not_expired(self) -> None:
        hooks.current_time = make_fake_current_time(1735200000)

        tokens = OAuthTokens(
            access_token="access",
            refresh_token="refresh",
            expires_at=1735200000 + 3600,  # Expires in 1 hour
            token_type="Bearer",
        )

        assert is_token_expired(tokens) is False

    def test_expired(self) -> None:
        hooks.current_time = make_fake_current_time(1735200000)

        tokens = OAuthTokens(
            access_token="access",
            refresh_token="refresh",
            expires_at=1735200000 - 100,  # Expired 100 seconds ago
            token_type="Bearer",
        )

        assert is_token_expired(tokens) is True

    def test_expires_within_buffer(self) -> None:
        hooks.current_time = make_fake_current_time(1735200000)

        tokens = OAuthTokens(
            access_token="access",
            refresh_token="refresh",
            expires_at=1735200000 + 30,  # Expires in 30 seconds
            token_type="Bearer",
        )

        # Default buffer is 60 seconds
        assert is_token_expired(tokens) is True

    def test_custom_buffer(self) -> None:
        hooks.current_time = make_fake_current_time(1735200000)

        tokens = OAuthTokens(
            access_token="access",
            refresh_token="refresh",
            expires_at=1735200000 + 30,
            token_type="Bearer",
        )

        # With smaller buffer, not expired
        assert is_token_expired(tokens, buffer_seconds=10) is False


class TestGetValidTokens:
    def test_returns_valid_tokens(self) -> None:
        hooks.current_time = make_fake_current_time(1735200000)

        tokens = OAuthTokens(
            access_token="access",
            refresh_token="refresh",
            expires_at=1735200000 + 3600,
            token_type="Bearer",
        )

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        result = get_valid_tokens(creds, tokens)
        assert result == tokens

    def test_refreshes_expired_tokens(self) -> None:
        hooks.current_time = make_fake_current_time(1735200000)

        expired_tokens = OAuthTokens(
            access_token="old_access",
            refresh_token="refresh",
            expires_at=1735200000 - 100,
            token_type="Bearer",
        )

        token_response = {
            "access_token": "new_access",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        hooks.http_post = make_fake_http_send(dump_json_str(token_response))

        saved_tokens: list[OAuthTokens] = []

        def save_hook(tokens: OAuthTokens) -> None:
            saved_tokens.append(tokens)

        hooks.save_tokens = save_hook

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        result = get_valid_tokens(creds, expired_tokens)
        assert result["access_token"] == "new_access"
        assert len(saved_tokens) == 1


class TestPKCEFunctions:
    def test_generate_code_verifier_is_url_safe(self) -> None:
        verifier = generate_code_verifier()
        # Should be URL-safe base64 (letters, digits, hyphen, underscore)
        # Verifier should be ~86 chars (64 bytes base64url encoded)
        assert len(verifier) >= 43
        for char in verifier:
            assert char.isalnum() or char in "-_"

    def test_generate_code_verifier_unique(self) -> None:
        v1 = generate_code_verifier()
        v2 = generate_code_verifier()
        assert v1 != v2

    def test_generate_code_challenge_deterministic(self) -> None:
        verifier = "test_verifier_12345"
        c1 = generate_code_challenge(verifier)
        c2 = generate_code_challenge(verifier)
        assert c1 == c2

    def test_generate_code_challenge_format(self) -> None:
        verifier = "test_verifier"
        challenge = generate_code_challenge(verifier)
        # Should be URL-safe base64 without padding
        assert "=" not in challenge
        for char in challenge:
            assert char.isalnum() or char in "-_"


class TestExchangeCodeForTokensOSError:
    def test_exchange_os_error(self) -> None:
        hooks.http_post = make_raising_http_send(OSError("Socket error"))

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        with pytest.raises(AppError) as exc_info:
            exchange_code_for_tokens(creds, code="code", code_verifier="verifier")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.AUTH_FAILED
        assert "Failed to exchange" in error.message


class TestRefreshAccessTokenOSError:
    def test_refresh_os_error(self) -> None:
        hooks.http_post = make_raising_http_send(OSError("Socket error"))

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        with pytest.raises(AppError) as exc_info:
            refresh_access_token(creds, "refresh_token")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.TOKEN_EXPIRED
        assert "Failed to refresh" in error.message


class TestAuthorize:
    def test_authorize_success(self) -> None:
        hooks.current_time = make_fake_current_time(1735200000)

        token_response = {
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        hooks.http_post = make_fake_http_send(dump_json_str(token_response))

        # Fake browser opener (does nothing)
        opened_urls: list[str] = []

        def fake_open_browser(url: str) -> None:
            opened_urls.append(url)

        hooks.open_browser = fake_open_browser

        # Fake console I/O - user enters authorization code
        console_output, console_input = make_fake_console(["test_auth_code"])
        hooks.console_output = console_output
        hooks.console_input = console_input

        # Fake token saver
        saved_tokens: list[OAuthTokens] = []

        def fake_save_tokens(tokens: OAuthTokens) -> None:
            saved_tokens.append(tokens)

        hooks.save_tokens = fake_save_tokens

        creds = OAuthCredentials(
            client_id="test_client_id",
            client_secret="test_secret",
            redirect_uri="http://localhost:8080",
        )

        result = authorize(creds)

        assert result["access_token"] == "new_access"
        assert result["refresh_token"] == "new_refresh"
        assert len(opened_urls) == 1
        assert "accounts.google.com" in opened_urls[0]
        assert len(saved_tokens) == 1

    def test_authorize_empty_code(self) -> None:
        # Fake browser opener
        hooks.open_browser = lambda url: None

        # Fake console I/O - user enters empty code
        console_output, console_input = make_fake_console([""])
        hooks.console_output = console_output
        hooks.console_input = console_input

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        with pytest.raises(AppError) as exc_info:
            authorize(creds)
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.AUTH_FAILED
        assert "No authorization code" in error.message


class TestLoadOrAuthorize:
    def test_returns_valid_cached_tokens(self) -> None:
        hooks.current_time = make_fake_current_time(1735200000)

        cached_tokens = OAuthTokens(
            access_token="cached_access",
            refresh_token="cached_refresh",
            expires_at=1735200000 + 3600,  # Valid for 1 hour
            token_type="Bearer",
        )

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        hooks.load_credentials = make_fake_credentials(creds)
        hooks.load_tokens = make_fake_tokens(cached_tokens)

        result = load_or_authorize()
        assert result["access_token"] == "cached_access"

    def test_refreshes_expired_cached_tokens(self) -> None:
        hooks.current_time = make_fake_current_time(1735200000)

        expired_tokens = OAuthTokens(
            access_token="old_access",
            refresh_token="old_refresh",
            expires_at=1735200000 - 100,  # Expired
            token_type="Bearer",
        )

        token_response = {
            "access_token": "new_access",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        hooks.http_post = make_fake_http_send(dump_json_str(token_response))

        saved_tokens: list[OAuthTokens] = []

        def fake_save_tokens(tokens: OAuthTokens) -> None:
            saved_tokens.append(tokens)

        hooks.save_tokens = fake_save_tokens

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        hooks.load_credentials = make_fake_credentials(creds)
        hooks.load_tokens = make_fake_tokens(expired_tokens)

        result = load_or_authorize()
        assert result["access_token"] == "new_access"
        assert len(saved_tokens) == 1

    def test_authorizes_when_no_cached_tokens(self) -> None:
        hooks.current_time = make_fake_current_time(1735200000)

        token_response = {
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        hooks.http_post = make_fake_http_send(dump_json_str(token_response))
        hooks.open_browser = lambda url: None

        console_output, console_input = make_fake_console(["auth_code"])
        hooks.console_output = console_output
        hooks.console_input = console_input

        saved_tokens: list[OAuthTokens] = []

        def fake_save_tokens(tokens: OAuthTokens) -> None:
            saved_tokens.append(tokens)

        hooks.save_tokens = fake_save_tokens

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        hooks.load_credentials = make_fake_credentials(creds)
        hooks.load_tokens = make_fake_no_tokens()

        result = load_or_authorize()
        assert result["access_token"] == "new_access"
        assert result["refresh_token"] == "new_refresh"

    def test_authorizes_when_refresh_fails(self) -> None:
        hooks.current_time = make_fake_current_time(1735200000)

        expired_tokens = OAuthTokens(
            access_token="old_access",
            refresh_token="old_refresh",
            expires_at=1735200000 - 100,  # Expired
            token_type="Bearer",
        )

        # First call will fail (refresh), second will succeed (authorize)
        call_count = [0]
        token_response_success = {
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        def mock_http_post(url: str, headers: dict[str, str], body: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                # Refresh fails
                raise ConnectionError("Network error")
            # Authorize succeeds
            return dump_json_str(token_response_success)

        hooks.http_post = mock_http_post
        hooks.open_browser = lambda url: None

        console_output, console_input = make_fake_console(["auth_code"])
        hooks.console_output = console_output
        hooks.console_input = console_input

        saved_tokens: list[OAuthTokens] = []

        def fake_save_tokens(tokens: OAuthTokens) -> None:
            saved_tokens.append(tokens)

        hooks.save_tokens = fake_save_tokens

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        hooks.load_credentials = make_fake_credentials(creds)
        hooks.load_tokens = make_fake_tokens(expired_tokens)

        result = load_or_authorize()
        assert result["access_token"] == "new_access"
        assert result["refresh_token"] == "new_refresh"
