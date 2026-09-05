"""Tests for platform_email.auth.outlook module."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import dump_json_str

from platform_email.auth.outlook import (
    authorize_outlook,
    build_outlook_auth_url,
    exchange_outlook_code_for_tokens,
    get_valid_outlook_tokens,
    outlook_load_or_authorize,
    refresh_outlook_access_token,
)
from platform_email.fake_hooks import (
    make_fake_console,
    make_fake_current_time,
    make_fake_http_send,
    make_fake_no_tokens,
    make_fake_outlook_config,
    make_fake_tokens,
    make_raising_http_send,
)
from platform_email.testing import (
    hooks,
    reset_hooks,
)
from platform_email.types import OAuthTokens, OutlookOAuthConfig


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


def _make_config() -> OutlookOAuthConfig:
    """Create a test config."""
    return OutlookOAuthConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",
        redirect_uri="http://localhost/callback",
        tenant_id="common",
    )


def _make_valid_tokens(expires_at: int = 9999999999) -> OAuthTokens:
    """Create valid tokens."""
    return OAuthTokens(
        access_token="test_access_token",
        refresh_token="test_refresh_token",
        expires_at=expires_at,
        token_type="Bearer",
    )


class TestBuildOutlookAuthUrl:
    """Tests for build_outlook_auth_url function."""

    def test_builds_correct_url(self) -> None:
        """Test that the URL contains all required parameters."""
        config = _make_config()
        url = build_outlook_auth_url(
            config,
            code_challenge="test_challenge",
            state="test_state",
        )

        assert "login.microsoftonline.com" in url
        assert "client_id=test_client_id" in url
        assert "response_type=code" in url
        assert "code_challenge=test_challenge" in url
        assert "state=test_state" in url
        assert "scope=" in url


class TestExchangeOutlookCodeForTokens:
    """Tests for exchange_outlook_code_for_tokens function."""

    def test_exchanges_code_successfully(self) -> None:
        """Test successful token exchange."""
        config = _make_config()
        response = dump_json_str(
            {
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "expires_in": 3600,
                "token_type": "Bearer",
            }
        )
        hooks.http_post = make_fake_http_send(response)
        hooks.current_time = make_fake_current_time(1000)

        tokens = exchange_outlook_code_for_tokens(
            config,
            code="auth_code",
            code_verifier="verifier",
        )

        assert tokens["access_token"] == "new_access"
        assert tokens["refresh_token"] == "new_refresh"
        assert tokens["expires_at"] == 4600  # 1000 + 3600

    def test_raises_on_connection_error(self) -> None:
        """Test that ConnectionError is wrapped in AppError."""
        config = _make_config()
        hooks.http_post = make_raising_http_send(ConnectionError("Network down"))

        with pytest.raises(AppError) as exc_info:
            exchange_outlook_code_for_tokens(config, code="code", code_verifier="verifier")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.AUTH_FAILED

    def test_raises_on_os_error(self) -> None:
        """Test that OSError is wrapped in AppError."""
        config = _make_config()
        hooks.http_post = make_raising_http_send(OSError("Socket error"))

        with pytest.raises(AppError) as exc_info:
            exchange_outlook_code_for_tokens(config, code="code", code_verifier="verifier")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.AUTH_FAILED

    def test_raises_on_invalid_json(self) -> None:
        """Test that invalid JSON raises AppError."""
        config = _make_config()
        hooks.http_post = make_fake_http_send("not json")

        with pytest.raises(AppError) as exc_info:
            exchange_outlook_code_for_tokens(config, code="code", code_verifier="verifier")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.AUTH_FAILED

    def test_raises_on_error_response(self) -> None:
        """Test that error in response raises AppError."""
        config = _make_config()
        response = dump_json_str(
            {
                "error": "invalid_grant",
                "error_description": "Code expired",
            }
        )
        hooks.http_post = make_fake_http_send(response)

        with pytest.raises(AppError) as exc_info:
            exchange_outlook_code_for_tokens(config, code="code", code_verifier="verifier")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.AUTH_FAILED
        assert "Code expired" in err.message

    def test_raises_when_no_refresh_token(self) -> None:
        """Test that missing refresh token raises AppError."""
        config = _make_config()
        response = dump_json_str(
            {
                "access_token": "access",
                "expires_in": 3600,
                "token_type": "Bearer",
            }
        )
        hooks.http_post = make_fake_http_send(response)
        hooks.current_time = make_fake_current_time(1000)

        with pytest.raises(AppError) as exc_info:
            exchange_outlook_code_for_tokens(config, code="code", code_verifier="verifier")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.AUTH_FAILED
        assert "No refresh token" in err.message


class TestRefreshOutlookAccessToken:
    """Tests for refresh_outlook_access_token function."""

    def test_refreshes_successfully(self) -> None:
        """Test successful token refresh."""
        config = _make_config()
        response = dump_json_str(
            {
                "access_token": "refreshed_access",
                "refresh_token": "refreshed_refresh",
                "expires_in": 3600,
            }
        )
        hooks.http_post = make_fake_http_send(response)
        hooks.current_time = make_fake_current_time(2000)

        tokens = refresh_outlook_access_token(config, "old_refresh")

        assert tokens["access_token"] == "refreshed_access"
        assert tokens["refresh_token"] == "refreshed_refresh"
        assert tokens["expires_at"] == 5600

    def test_keeps_old_refresh_token_if_not_returned(self) -> None:
        """Test that original refresh token is kept if new one not returned."""
        config = _make_config()
        response = dump_json_str(
            {
                "access_token": "refreshed_access",
                "expires_in": 3600,
            }
        )
        hooks.http_post = make_fake_http_send(response)
        hooks.current_time = make_fake_current_time(2000)

        tokens = refresh_outlook_access_token(config, "old_refresh")

        assert tokens["refresh_token"] == "old_refresh"

    def test_raises_on_connection_error(self) -> None:
        """Test that ConnectionError is wrapped in AppError."""
        config = _make_config()
        hooks.http_post = make_raising_http_send(ConnectionError("Down"))

        with pytest.raises(AppError) as exc_info:
            refresh_outlook_access_token(config, "refresh")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.TOKEN_EXPIRED

    def test_raises_on_os_error(self) -> None:
        """Test that OSError is wrapped in AppError."""
        config = _make_config()
        hooks.http_post = make_raising_http_send(OSError("Socket"))

        with pytest.raises(AppError) as exc_info:
            refresh_outlook_access_token(config, "refresh")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.TOKEN_EXPIRED

    def test_raises_on_invalid_json(self) -> None:
        """Test that invalid JSON raises AppError."""
        config = _make_config()
        hooks.http_post = make_fake_http_send("{invalid")

        with pytest.raises(AppError) as exc_info:
            refresh_outlook_access_token(config, "refresh")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.TOKEN_EXPIRED

    def test_raises_on_error_response(self) -> None:
        """Test that error in response raises AppError."""
        config = _make_config()
        response = dump_json_str({"error": "invalid_grant"})
        hooks.http_post = make_fake_http_send(response)

        with pytest.raises(AppError) as exc_info:
            refresh_outlook_access_token(config, "refresh")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.TOKEN_EXPIRED


class TestGetValidOutlookTokens:
    """Tests for get_valid_outlook_tokens function."""

    def test_returns_tokens_if_not_expired(self) -> None:
        """Test that non-expired tokens are returned as-is."""
        config = _make_config()
        tokens = _make_valid_tokens(expires_at=9999999999)
        hooks.current_time = make_fake_current_time(1000)

        result = get_valid_outlook_tokens(config, tokens)

        assert result == tokens

    def test_refreshes_expired_tokens(self) -> None:
        """Test that expired tokens are refreshed."""
        config = _make_config()
        expired_tokens = _make_valid_tokens(expires_at=500)
        response = dump_json_str(
            {
                "access_token": "refreshed",
                "refresh_token": "new_refresh",
                "expires_in": 3600,
            }
        )
        hooks.http_post = make_fake_http_send(response)
        hooks.current_time = make_fake_current_time(1000)

        saved_tokens: list[OAuthTokens] = []

        def _save(t: OAuthTokens) -> None:
            saved_tokens.append(t)

        hooks.save_outlook_tokens = _save

        result = get_valid_outlook_tokens(config, expired_tokens)

        assert result["access_token"] == "refreshed"
        assert len(saved_tokens) == 1


class TestAuthorizeOutlook:
    """Tests for authorize_outlook function."""

    def test_runs_full_auth_flow(self) -> None:
        """Test the complete authorization flow."""
        config = _make_config()

        opened_urls: list[str] = []

        def _open_browser(url: str) -> None:
            opened_urls.append(url)

        output_hook, input_hook = make_fake_console(["auth_code_123"])

        response = dump_json_str(
            {
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "expires_in": 3600,
            }
        )
        hooks.http_post = make_fake_http_send(response)
        hooks.current_time = make_fake_current_time(1000)
        hooks.open_browser = _open_browser
        hooks.console_output = output_hook
        hooks.console_input = input_hook

        saved_tokens: list[OAuthTokens] = []

        def _save(t: OAuthTokens) -> None:
            saved_tokens.append(t)

        hooks.save_outlook_tokens = _save

        tokens = authorize_outlook(config)

        assert len(opened_urls) == 1
        assert "login.microsoftonline.com" in opened_urls[0]
        assert tokens["access_token"] == "new_access"
        assert len(saved_tokens) == 1

    def test_raises_if_no_code_provided(self) -> None:
        """Test that empty code raises AppError."""
        config = _make_config()

        def _open_browser(_url: str) -> None:
            pass

        output_hook, input_hook = make_fake_console([""])

        hooks.open_browser = _open_browser
        hooks.console_output = output_hook
        hooks.console_input = input_hook

        with pytest.raises(AppError) as exc_info:
            authorize_outlook(config)
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.AUTH_FAILED
        assert "No authorization code" in err.message


class TestOutlookLoadOrAuthorize:
    """Tests for outlook_load_or_authorize function."""

    def test_returns_cached_valid_tokens(self) -> None:
        """Test that valid cached tokens are returned."""
        config = _make_config()
        tokens = _make_valid_tokens()

        hooks.load_outlook_config = make_fake_outlook_config(config)
        hooks.load_outlook_tokens = make_fake_tokens(tokens)
        hooks.current_time = make_fake_current_time(1000)

        result = outlook_load_or_authorize()

        assert result == tokens

    def test_refreshes_expired_cached_tokens(self) -> None:
        """Test that expired cached tokens are refreshed."""
        config = _make_config()
        expired_tokens = _make_valid_tokens(expires_at=500)
        response = dump_json_str(
            {
                "access_token": "refreshed",
                "refresh_token": "new_refresh",
                "expires_in": 3600,
            }
        )

        hooks.load_outlook_config = make_fake_outlook_config(config)
        hooks.load_outlook_tokens = make_fake_tokens(expired_tokens)
        hooks.http_post = make_fake_http_send(response)
        hooks.current_time = make_fake_current_time(1000)

        saved_tokens: list[OAuthTokens] = []

        def _save(t: OAuthTokens) -> None:
            saved_tokens.append(t)

        hooks.save_outlook_tokens = _save

        result = outlook_load_or_authorize()

        assert result["access_token"] == "refreshed"
        assert len(saved_tokens) == 1

    def test_runs_auth_flow_on_refresh_failure(self) -> None:
        """Test that auth flow runs if refresh fails."""
        config = _make_config()
        expired_tokens = _make_valid_tokens(expires_at=500)

        hooks.load_outlook_config = make_fake_outlook_config(config)
        hooks.load_outlook_tokens = make_fake_tokens(expired_tokens)
        hooks.current_time = make_fake_current_time(1000)

        # First call fails (refresh), second succeeds (auth)
        call_count = [0]

        def _fake_http_post(_url: str, _headers: dict[str, str], _body: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return dump_json_str({"error": "invalid_grant"})
            return dump_json_str(
                {
                    "access_token": "new_access",
                    "refresh_token": "new_refresh",
                    "expires_in": 3600,
                }
            )

        hooks.http_post = _fake_http_post

        def _open_browser(_url: str) -> None:
            pass

        output_hook, input_hook = make_fake_console(["auth_code"])
        hooks.open_browser = _open_browser
        hooks.console_output = output_hook
        hooks.console_input = input_hook

        saved_tokens: list[OAuthTokens] = []

        def _save(t: OAuthTokens) -> None:
            saved_tokens.append(t)

        hooks.save_outlook_tokens = _save

        result = outlook_load_or_authorize()

        assert result["access_token"] == "new_access"

    def test_runs_auth_flow_if_no_cached_tokens(self) -> None:
        """Test that auth flow runs when no cached tokens exist."""
        config = _make_config()

        hooks.load_outlook_config = make_fake_outlook_config(config)
        hooks.load_outlook_tokens = make_fake_no_tokens()
        hooks.current_time = make_fake_current_time(1000)

        response = dump_json_str(
            {
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "expires_in": 3600,
            }
        )
        hooks.http_post = make_fake_http_send(response)

        def _open_browser(_url: str) -> None:
            pass

        output_hook, input_hook = make_fake_console(["auth_code"])
        hooks.open_browser = _open_browser
        hooks.console_output = output_hook
        hooks.console_input = input_hook

        saved_tokens: list[OAuthTokens] = []

        def _save(t: OAuthTokens) -> None:
            saved_tokens.append(t)

        hooks.save_outlook_tokens = _save

        result = outlook_load_or_authorize()

        assert result["access_token"] == "new_access"
