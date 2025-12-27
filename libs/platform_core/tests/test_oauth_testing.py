"""Tests for platform_core.oauth_testing module."""

from __future__ import annotations

import pytest

from platform_core.oauth_testing import (
    make_advancing_current_time,
    make_error_response_json,
    make_fake_current_time,
    make_fake_http_post,
    make_raising_http_post,
    make_sequenced_http_post,
    make_test_credentials,
    make_test_token_response,
    make_test_tokens,
    make_token_response_json,
)


class TestMakeFakeHttpPost:
    def test_returns_fixed_response(self) -> None:
        hook = make_fake_http_post('{"success": true}')
        result = hook("http://example.com", {"header": "value"}, "body")
        assert result == '{"success": true}'

    def test_ignores_inputs(self) -> None:
        hook = make_fake_http_post("response")
        result1 = hook("http://first.com", {}, "first")
        result2 = hook("http://second.com", {"x": "y"}, "second")
        assert result1 == result2 == "response"


class TestMakeRaisingHttpPost:
    def test_raises_connection_error(self) -> None:
        hook = make_raising_http_post(ConnectionError("Network down"))
        with pytest.raises(ConnectionError) as exc_info:
            hook("http://example.com", {}, "body")
        assert "Network down" in str(exc_info.value)

    def test_raises_os_error(self) -> None:
        hook = make_raising_http_post(OSError("Socket error"))
        with pytest.raises(OSError):
            hook("http://example.com", {}, "body")

    def test_raises_custom_exception(self) -> None:
        hook = make_raising_http_post(ValueError("Custom error"))
        with pytest.raises(ValueError):
            hook("http://example.com", {}, "body")


class TestMakeSequencedHttpPost:
    def test_returns_responses_in_order(self) -> None:
        hook = make_sequenced_http_post(["first", "second", "third"])
        assert hook("url", {}, "body") == "first"
        assert hook("url", {}, "body") == "second"
        assert hook("url", {}, "body") == "third"

    def test_raises_exceptions_in_sequence(self) -> None:
        hook = make_sequenced_http_post(
            [
                "success",
                ConnectionError("Failed"),
                "recovered",
            ]
        )

        assert hook("url", {}, "body") == "success"
        with pytest.raises(ConnectionError):
            hook("url", {}, "body")
        assert hook("url", {}, "body") == "recovered"

    def test_raises_when_exhausted(self) -> None:
        hook = make_sequenced_http_post(["only_one"])
        hook("url", {}, "body")  # First call succeeds
        with pytest.raises(RuntimeError) as exc_info:
            hook("url", {}, "body")  # Second call fails
        assert "No more responses" in str(exc_info.value)


class TestMakeFakeCurrentTime:
    def test_returns_fixed_timestamp(self) -> None:
        hook = make_fake_current_time(1735200000)
        assert hook() == 1735200000
        assert hook() == 1735200000
        assert hook() == 1735200000

    def test_different_timestamps(self) -> None:
        hook1 = make_fake_current_time(1000)
        hook2 = make_fake_current_time(2000)
        assert hook1() == 1000
        assert hook2() == 2000


class TestMakeAdvancingCurrentTime:
    def test_advances_by_increment(self) -> None:
        hook = make_advancing_current_time(1000, increment=100)
        assert hook() == 1000
        assert hook() == 1100
        assert hook() == 1200

    def test_default_increment_is_one(self) -> None:
        hook = make_advancing_current_time(1000)
        assert hook() == 1000
        assert hook() == 1001
        assert hook() == 1002


class TestMakeTokenResponseJson:
    def test_default_values(self) -> None:
        json_str = make_token_response_json()
        assert '"access_token":"test_access_token"' in json_str
        assert '"refresh_token":"test_refresh_token"' in json_str
        assert '"expires_in":3600' in json_str
        assert '"token_type":"Bearer"' in json_str

    def test_custom_access_token(self) -> None:
        json_str = make_token_response_json(access_token="custom_access")
        assert '"access_token":"custom_access"' in json_str

    def test_no_refresh_token(self) -> None:
        json_str = make_token_response_json(refresh_token=None)
        assert "refresh_token" not in json_str

    def test_custom_expires_in(self) -> None:
        json_str = make_token_response_json(expires_in=7200)
        assert '"expires_in":7200' in json_str


class TestMakeErrorResponseJson:
    def test_error_only(self) -> None:
        json_str = make_error_response_json(error="invalid_grant")
        assert '"error":"invalid_grant"' in json_str
        assert "error_description" not in json_str

    def test_with_description(self) -> None:
        json_str = make_error_response_json(
            error="invalid_grant",
            error_description="Token expired",
        )
        assert '"error":"invalid_grant"' in json_str
        assert '"error_description":"Token expired"' in json_str


class TestMakeTestCredentials:
    def test_default_values(self) -> None:
        creds = make_test_credentials()
        assert creds["client_id"] == "test_client_id"
        assert creds["client_secret"] == "test_client_secret"
        assert creds["redirect_uri"] == "http://localhost:8080/callback"

    def test_custom_values(self) -> None:
        creds = make_test_credentials(
            client_id="custom_id",
            client_secret="custom_secret",
            redirect_uri="http://custom.test",
        )
        assert creds["client_id"] == "custom_id"
        assert creds["client_secret"] == "custom_secret"
        assert creds["redirect_uri"] == "http://custom.test"


class TestMakeTestTokens:
    def test_default_values(self) -> None:
        tokens = make_test_tokens()
        assert tokens["access_token"] == "test_access_token"
        assert tokens["refresh_token"] == "test_refresh_token"
        assert tokens["expires_at"] == 1735200000
        assert tokens["token_type"] == "Bearer"

    def test_custom_values(self) -> None:
        tokens = make_test_tokens(
            access_token="custom_access",
            refresh_token="custom_refresh",
            expires_at=9999999999,
        )
        assert tokens["access_token"] == "custom_access"
        assert tokens["refresh_token"] == "custom_refresh"
        assert tokens["expires_at"] == 9999999999

    def test_expired_flag(self) -> None:
        current_time = 1735200000
        tokens = make_test_tokens(expired=True, current_time=current_time)
        assert tokens["expires_at"] < current_time

    def test_expired_uses_default_current_time(self) -> None:
        tokens = make_test_tokens(expired=True)
        # Should be 100 seconds before default time (1735200000)
        assert tokens["expires_at"] == 1735200000 - 100


class TestMakeTestTokenResponse:
    def test_default_values(self) -> None:
        response = make_test_token_response()
        assert response["access_token"] == "test_access_token"
        assert response["refresh_token"] == "test_refresh_token"
        assert response["expires_in"] == 3600
        assert response["token_type"] == "Bearer"

    def test_custom_values(self) -> None:
        response = make_test_token_response(
            access_token="custom_access",
            refresh_token=None,
            expires_in=7200,
        )
        assert response["access_token"] == "custom_access"
        assert response["refresh_token"] is None
        assert response["expires_in"] == 7200
