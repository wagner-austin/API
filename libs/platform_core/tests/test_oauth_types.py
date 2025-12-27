"""Tests for platform_core.oauth_types module."""

from __future__ import annotations

import pytest

from platform_core.json_utils import JSONObject, JSONTypeError
from platform_core.oauth_types import (
    OAuthCredentials,
    OAuthTokenResponse,
    OAuthTokens,
    decode_oauth_credentials,
    decode_oauth_token_response,
    decode_oauth_tokens,
    encode_oauth_credentials,
    encode_oauth_token_response,
    encode_oauth_tokens,
)


class TestOAuthCredentialsEncodeDecode:
    def test_encode_credentials(self) -> None:
        credentials = OAuthCredentials(
            client_id="test_client_id",
            client_secret="test_secret",
            redirect_uri="http://localhost:8080",
        )
        result = encode_oauth_credentials(credentials)

        assert result["client_id"] == "test_client_id"
        assert result["client_secret"] == "test_secret"
        assert result["redirect_uri"] == "http://localhost:8080"

    def test_decode_credentials(self) -> None:
        data: JSONObject = {
            "client_id": "test_client_id",
            "client_secret": "test_secret",
            "redirect_uri": "http://localhost:8080",
        }
        result = decode_oauth_credentials(data)

        assert result["client_id"] == "test_client_id"
        assert result["client_secret"] == "test_secret"
        assert result["redirect_uri"] == "http://localhost:8080"

    def test_decode_credentials_missing_client_id(self) -> None:
        data: JSONObject = {
            "client_secret": "test_secret",
            "redirect_uri": "http://localhost:8080",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_credentials(data)
        assert "client_id" in str(exc_info.value)

    def test_decode_credentials_missing_client_secret(self) -> None:
        data: JSONObject = {
            "client_id": "test_client_id",
            "redirect_uri": "http://localhost:8080",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_credentials(data)
        assert "client_secret" in str(exc_info.value)

    def test_decode_credentials_missing_redirect_uri(self) -> None:
        data: JSONObject = {
            "client_id": "test_client_id",
            "client_secret": "test_secret",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_credentials(data)
        assert "redirect_uri" in str(exc_info.value)

    def test_decode_credentials_wrong_type(self) -> None:
        data: JSONObject = {
            "client_id": 123,
            "client_secret": "test_secret",
            "redirect_uri": "http://localhost:8080",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_credentials(data)
        assert "client_id" in str(exc_info.value)

    def test_roundtrip_credentials(self) -> None:
        original = OAuthCredentials(
            client_id="roundtrip_id",
            client_secret="roundtrip_secret",
            redirect_uri="http://roundtrip.test",
        )
        encoded = encode_oauth_credentials(original)
        decoded = decode_oauth_credentials(encoded)

        assert decoded["client_id"] == original["client_id"]
        assert decoded["client_secret"] == original["client_secret"]
        assert decoded["redirect_uri"] == original["redirect_uri"]


class TestOAuthTokensEncodeDecode:
    def test_encode_tokens(self) -> None:
        tokens = OAuthTokens(
            access_token="access_123",
            refresh_token="refresh_456",
            expires_at=1735200000,
            token_type="Bearer",
        )
        result = encode_oauth_tokens(tokens)

        assert result["access_token"] == "access_123"
        assert result["refresh_token"] == "refresh_456"
        assert result["expires_at"] == 1735200000
        assert result["token_type"] == "Bearer"

    def test_decode_tokens(self) -> None:
        data: JSONObject = {
            "access_token": "access_123",
            "refresh_token": "refresh_456",
            "expires_at": 1735200000,
            "token_type": "Bearer",
        }
        result = decode_oauth_tokens(data)

        assert result["access_token"] == "access_123"
        assert result["refresh_token"] == "refresh_456"
        assert result["expires_at"] == 1735200000
        assert result["token_type"] == "Bearer"

    def test_decode_tokens_missing_access_token(self) -> None:
        data: JSONObject = {
            "refresh_token": "refresh_456",
            "expires_at": 1735200000,
            "token_type": "Bearer",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_tokens(data)
        assert "access_token" in str(exc_info.value)

    def test_decode_tokens_missing_refresh_token(self) -> None:
        data: JSONObject = {
            "access_token": "access_123",
            "expires_at": 1735200000,
            "token_type": "Bearer",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_tokens(data)
        assert "refresh_token" in str(exc_info.value)

    def test_decode_tokens_missing_expires_at(self) -> None:
        data: JSONObject = {
            "access_token": "access_123",
            "refresh_token": "refresh_456",
            "token_type": "Bearer",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_tokens(data)
        assert "expires_at" in str(exc_info.value)

    def test_decode_tokens_missing_token_type(self) -> None:
        data: JSONObject = {
            "access_token": "access_123",
            "refresh_token": "refresh_456",
            "expires_at": 1735200000,
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_tokens(data)
        assert "token_type" in str(exc_info.value)

    def test_decode_tokens_invalid_token_type(self) -> None:
        data: JSONObject = {
            "access_token": "access_123",
            "refresh_token": "refresh_456",
            "expires_at": 1735200000,
            "token_type": "Basic",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_tokens(data)
        assert "token_type" in str(exc_info.value)
        assert "Bearer" in str(exc_info.value)

    def test_decode_tokens_expires_at_wrong_type(self) -> None:
        data: JSONObject = {
            "access_token": "access_123",
            "refresh_token": "refresh_456",
            "expires_at": "not_an_int",
            "token_type": "Bearer",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_tokens(data)
        assert "expires_at" in str(exc_info.value)

    def test_roundtrip_tokens(self) -> None:
        original = OAuthTokens(
            access_token="roundtrip_access",
            refresh_token="roundtrip_refresh",
            expires_at=1735300000,
            token_type="Bearer",
        )
        encoded = encode_oauth_tokens(original)
        decoded = decode_oauth_tokens(encoded)

        assert decoded["access_token"] == original["access_token"]
        assert decoded["refresh_token"] == original["refresh_token"]
        assert decoded["expires_at"] == original["expires_at"]
        assert decoded["token_type"] == original["token_type"]


class TestOAuthTokenResponseEncodeDecode:
    def test_encode_token_response_with_refresh(self) -> None:
        response = OAuthTokenResponse(
            access_token="access_123",
            refresh_token="refresh_456",
            expires_in=3600,
            token_type="Bearer",
        )
        result = encode_oauth_token_response(response)

        assert result["access_token"] == "access_123"
        assert result["refresh_token"] == "refresh_456"
        assert result["expires_in"] == 3600
        assert result["token_type"] == "Bearer"

    def test_encode_token_response_without_refresh(self) -> None:
        response = OAuthTokenResponse(
            access_token="access_123",
            refresh_token=None,
            expires_in=3600,
            token_type="Bearer",
        )
        result = encode_oauth_token_response(response)

        assert result["access_token"] == "access_123"
        assert result["refresh_token"] is None
        assert result["expires_in"] == 3600
        assert result["token_type"] == "Bearer"

    def test_decode_token_response_with_refresh(self) -> None:
        data: JSONObject = {
            "access_token": "access_123",
            "refresh_token": "refresh_456",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        result = decode_oauth_token_response(data)

        assert result["access_token"] == "access_123"
        assert result["refresh_token"] == "refresh_456"
        assert result["expires_in"] == 3600
        assert result["token_type"] == "Bearer"

    def test_decode_token_response_without_refresh(self) -> None:
        data: JSONObject = {
            "access_token": "access_123",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        result = decode_oauth_token_response(data)

        assert result["access_token"] == "access_123"
        assert result["refresh_token"] is None
        assert result["expires_in"] == 3600
        assert result["token_type"] == "Bearer"

    def test_decode_token_response_refresh_null(self) -> None:
        data: JSONObject = {
            "access_token": "access_123",
            "refresh_token": None,
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        result = decode_oauth_token_response(data)

        assert result["refresh_token"] is None

    def test_decode_token_response_missing_access_token(self) -> None:
        data: JSONObject = {
            "refresh_token": "refresh_456",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_token_response(data)
        assert "access_token" in str(exc_info.value)

    def test_decode_token_response_missing_expires_in(self) -> None:
        data: JSONObject = {
            "access_token": "access_123",
            "token_type": "Bearer",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_token_response(data)
        assert "expires_in" in str(exc_info.value)

    def test_decode_token_response_missing_token_type(self) -> None:
        data: JSONObject = {
            "access_token": "access_123",
            "expires_in": 3600,
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_oauth_token_response(data)
        assert "token_type" in str(exc_info.value)

    def test_roundtrip_token_response(self) -> None:
        original = OAuthTokenResponse(
            access_token="roundtrip_access",
            refresh_token="roundtrip_refresh",
            expires_in=7200,
            token_type="Bearer",
        )
        encoded = encode_oauth_token_response(original)
        decoded = decode_oauth_token_response(encoded)

        assert decoded["access_token"] == original["access_token"]
        assert decoded["refresh_token"] == original["refresh_token"]
        assert decoded["expires_in"] == original["expires_in"]
        assert decoded["token_type"] == original["token_type"]
