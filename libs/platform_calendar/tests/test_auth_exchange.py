"""Auth URL building, code exchange, token refresh."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, CalendarErrorCode
from platform_core.json_utils import dump_json_str

from platform_calendar.auth import (
    build_auth_url,
    exchange_code_for_tokens,
    refresh_access_token,
)
from platform_calendar.fakes import (
    make_fake_current_time,
    make_fake_http_post,
    make_raising_http_post,
)
from platform_calendar.testing import (
    hooks,
)
from platform_calendar.types import OAuthCredentials


class TestBuildAuthUrl:
    def test_builds_url_with_params(self) -> None:
        creds = OAuthCredentials(
            client_id="test_client_id",
            client_secret="test_secret",
            redirect_uri="http://localhost:8080",
        )
        url = build_auth_url(creds, code_challenge="challenge123", state="state456")

        assert "accounts.google.com" in url
        assert "client_id=test_client_id" in url
        assert "redirect_uri=http" in url
        assert "code_challenge=challenge123" in url
        assert "state=state456" in url
        assert "response_type=code" in url
        assert "access_type=offline" in url


class TestExchangeCodeForTokens:
    def test_successful_exchange(self) -> None:
        token_response = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        hooks.http_post = make_fake_http_post(dump_json_str(token_response))
        hooks.current_time = make_fake_current_time(1735200000)

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        tokens = exchange_code_for_tokens(
            creds,
            code="auth_code",
            code_verifier="verifier",
        )

        assert tokens["access_token"] == "new_access_token"
        assert tokens["refresh_token"] == "new_refresh_token"
        assert tokens["expires_at"] == 1735200000 + 3600
        assert tokens["token_type"] == "Bearer"

    def test_exchange_network_error(self) -> None:
        hooks.http_post = make_raising_http_post(ConnectionError("Network error"))

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

    def test_exchange_invalid_json(self) -> None:
        hooks.http_post = make_fake_http_post("not valid json")

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        with pytest.raises(AppError) as exc_info:
            exchange_code_for_tokens(creds, code="code", code_verifier="verifier")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.AUTH_FAILED
        assert "Invalid JSON" in error.message

    def test_exchange_not_object(self) -> None:
        hooks.http_post = make_fake_http_post("[]")

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        with pytest.raises(AppError) as exc_info:
            exchange_code_for_tokens(creds, code="code", code_verifier="verifier")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.AUTH_FAILED
        assert "Expected JSON object" in error.message

    def test_exchange_error_response(self) -> None:
        error_response = {
            "error": "invalid_grant",
            "error_description": "Code expired",
        }
        hooks.http_post = make_fake_http_post(dump_json_str(error_response))

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        with pytest.raises(AppError) as exc_info:
            exchange_code_for_tokens(creds, code="code", code_verifier="verifier")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.AUTH_FAILED
        assert "Code expired" in error.message

    def test_exchange_error_without_description(self) -> None:
        error_response = {"error": "invalid_grant"}
        hooks.http_post = make_fake_http_post(dump_json_str(error_response))

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        with pytest.raises(AppError) as exc_info:
            exchange_code_for_tokens(creds, code="code", code_verifier="verifier")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.AUTH_FAILED
        assert "invalid_grant" in error.message

    def test_exchange_no_refresh_token(self) -> None:
        token_response = {
            "access_token": "access",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        hooks.http_post = make_fake_http_post(dump_json_str(token_response))
        hooks.current_time = make_fake_current_time(1735200000)

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        with pytest.raises(AppError) as exc_info:
            exchange_code_for_tokens(creds, code="code", code_verifier="verifier")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.AUTH_FAILED
        assert "No refresh token" in error.message


class TestRefreshAccessToken:
    def test_successful_refresh(self) -> None:
        token_response = {
            "access_token": "new_access_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        hooks.http_post = make_fake_http_post(dump_json_str(token_response))
        hooks.current_time = make_fake_current_time(1735200000)

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        tokens = refresh_access_token(creds, "old_refresh_token")

        assert tokens["access_token"] == "new_access_token"
        assert tokens["refresh_token"] == "old_refresh_token"  # Kept original
        assert tokens["expires_at"] == 1735200000 + 3600

    def test_refresh_network_error(self) -> None:
        hooks.http_post = make_raising_http_post(ConnectionError("Network error"))

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

    def test_refresh_invalid_json(self) -> None:
        hooks.http_post = make_fake_http_post("not json")

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        with pytest.raises(AppError) as exc_info:
            refresh_access_token(creds, "refresh_token")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.TOKEN_EXPIRED
        assert "Invalid JSON" in error.message

    def test_refresh_not_object(self) -> None:
        hooks.http_post = make_fake_http_post('"string"')

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        with pytest.raises(AppError) as exc_info:
            refresh_access_token(creds, "refresh_token")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.TOKEN_EXPIRED
        assert "Expected JSON object" in error.message

    def test_refresh_error_response(self) -> None:
        error_response = {
            "error": "invalid_grant",
            "error_description": "Token revoked",
        }
        hooks.http_post = make_fake_http_post(dump_json_str(error_response))

        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )

        with pytest.raises(AppError) as exc_info:
            refresh_access_token(creds, "refresh_token")
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.TOKEN_EXPIRED
        assert "Token revoked" in error.message
