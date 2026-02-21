"""Tests for platform_email.types.oauth module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_email.types.oauth import (
    GmailOAuthConfig,
    OutlookOAuthConfig,
    decode_gmail_oauth_config,
    decode_outlook_oauth_config,
    encode_gmail_oauth_config,
    encode_outlook_oauth_config,
)

# =============================================================================
# OutlookOAuthConfig tests
# =============================================================================


class TestOutlookOAuthConfig:
    """Tests for OutlookOAuthConfig encode/decode functions."""

    def test_encode_outlook_oauth_config(self) -> None:
        """Test encoding an OutlookOAuthConfig to JSON."""
        config = OutlookOAuthConfig(
            client_id="azure-app-id-123",
            client_secret="azure-secret-456",
            redirect_uri="http://localhost",
            tenant_id="common",
        )
        result = encode_outlook_oauth_config(config)

        assert result["client_id"] == "azure-app-id-123"
        assert result["client_secret"] == "azure-secret-456"
        assert result["redirect_uri"] == "http://localhost"
        assert result["tenant_id"] == "common"

    def test_decode_outlook_oauth_config(self) -> None:
        """Test decoding an OutlookOAuthConfig from JSON."""
        data: JSONObject = {
            "client_id": "decoded-client-id",
            "client_secret": "decoded-secret",
            "redirect_uri": "http://localhost:8080/callback",
            "tenant_id": "tenant-123",
        }
        result = decode_outlook_oauth_config(data)

        assert result["client_id"] == "decoded-client-id"
        assert result["client_secret"] == "decoded-secret"
        assert result["redirect_uri"] == "http://localhost:8080/callback"
        assert result["tenant_id"] == "tenant-123"

    def test_decode_outlook_oauth_config_raises_for_missing_client_id(self) -> None:
        """Test that missing client_id raises JSONTypeError."""
        data: JSONObject = {
            "client_secret": "secret",
            "redirect_uri": "http://localhost",
            "tenant_id": "common",
        }
        with pytest.raises(JSONTypeError):
            decode_outlook_oauth_config(data)

    def test_decode_outlook_oauth_config_raises_for_missing_tenant_id(self) -> None:
        """Test that missing tenant_id raises JSONTypeError."""
        data: JSONObject = {
            "client_id": "id",
            "client_secret": "secret",
            "redirect_uri": "http://localhost",
        }
        with pytest.raises(JSONTypeError):
            decode_outlook_oauth_config(data)

    def test_roundtrip(self) -> None:
        """Test encode then decode preserves data."""
        original = OutlookOAuthConfig(
            client_id="roundtrip-id",
            client_secret="roundtrip-secret",
            redirect_uri="http://localhost/oauth",
            tenant_id="roundtrip-tenant",
        )
        encoded = encode_outlook_oauth_config(original)
        decoded = decode_outlook_oauth_config(encoded)
        assert decoded == original


# =============================================================================
# GmailOAuthConfig tests
# =============================================================================


class TestGmailOAuthConfig:
    """Tests for GmailOAuthConfig encode/decode functions."""

    def test_encode_gmail_oauth_config(self) -> None:
        """Test encoding a GmailOAuthConfig to JSON."""
        config = GmailOAuthConfig(
            client_id="google-client-id-123",
            client_secret="google-secret-456",
            redirect_uri="http://localhost:3000",
        )
        result = encode_gmail_oauth_config(config)

        assert result["client_id"] == "google-client-id-123"
        assert result["client_secret"] == "google-secret-456"
        assert result["redirect_uri"] == "http://localhost:3000"

    def test_decode_gmail_oauth_config(self) -> None:
        """Test decoding a GmailOAuthConfig from JSON."""
        data: JSONObject = {
            "client_id": "decoded-google-id",
            "client_secret": "decoded-google-secret",
            "redirect_uri": "urn:ietf:wg:oauth:2.0:oob",
        }
        result = decode_gmail_oauth_config(data)

        assert result["client_id"] == "decoded-google-id"
        assert result["client_secret"] == "decoded-google-secret"
        assert result["redirect_uri"] == "urn:ietf:wg:oauth:2.0:oob"

    def test_decode_gmail_oauth_config_raises_for_missing_client_id(self) -> None:
        """Test that missing client_id raises JSONTypeError."""
        data: JSONObject = {
            "client_secret": "secret",
            "redirect_uri": "http://localhost",
        }
        with pytest.raises(JSONTypeError):
            decode_gmail_oauth_config(data)

    def test_decode_gmail_oauth_config_raises_for_missing_client_secret(self) -> None:
        """Test that missing client_secret raises JSONTypeError."""
        data: JSONObject = {
            "client_id": "id",
            "redirect_uri": "http://localhost",
        }
        with pytest.raises(JSONTypeError):
            decode_gmail_oauth_config(data)

    def test_roundtrip(self) -> None:
        """Test encode then decode preserves data."""
        original = GmailOAuthConfig(
            client_id="gmail-roundtrip-id",
            client_secret="gmail-roundtrip-secret",
            redirect_uri="http://localhost/gmail/callback",
        )
        encoded = encode_gmail_oauth_config(original)
        decoded = decode_gmail_oauth_config(encoded)
        assert decoded == original
