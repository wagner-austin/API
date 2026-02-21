"""Tests for platform_email.config.outlook module."""

from __future__ import annotations

from platform_email.config.outlook import (
    DEFAULT_OUTLOOK_CREDENTIALS_PATH,
    DEFAULT_OUTLOOK_TOKENS_PATH,
    OUTLOOK_API_BASE,
    OUTLOOK_EMAIL_SCOPES,
    get_outlook_credentials_path,
    get_outlook_tokens_path,
    outlook_auth_url,
    outlook_token_url,
)


class TestOutlookAuthUrl:
    """Tests for outlook_auth_url function."""

    def test_generates_url_with_common_tenant(self) -> None:
        """Test URL generation with common tenant."""
        url = outlook_auth_url("common")
        assert url == "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"

    def test_generates_url_with_specific_tenant(self) -> None:
        """Test URL generation with specific tenant ID."""
        url = outlook_auth_url("12345-tenant-id-67890")
        assert "12345-tenant-id-67890" in url
        assert url.endswith("/oauth2/v2.0/authorize")


class TestOutlookTokenUrl:
    """Tests for outlook_token_url function."""

    def test_generates_url_with_common_tenant(self) -> None:
        """Test URL generation with common tenant."""
        url = outlook_token_url("common")
        assert url == "https://login.microsoftonline.com/common/oauth2/v2.0/token"

    def test_generates_url_with_specific_tenant(self) -> None:
        """Test URL generation with specific tenant ID."""
        url = outlook_token_url("my-tenant-123")
        assert "my-tenant-123" in url
        assert url.endswith("/oauth2/v2.0/token")


class TestOutlookConstants:
    """Tests for Outlook configuration constants."""

    def test_api_base_url(self) -> None:
        """Test API base URL is correct."""
        assert OUTLOOK_API_BASE == "https://graph.microsoft.com/v1.0"

    def test_email_scopes_includes_mail_read(self) -> None:
        """Test email scopes include Mail.Read."""
        assert "https://graph.microsoft.com/Mail.Read" in OUTLOOK_EMAIL_SCOPES

    def test_email_scopes_includes_mail_send(self) -> None:
        """Test email scopes include Mail.Send."""
        assert "https://graph.microsoft.com/Mail.Send" in OUTLOOK_EMAIL_SCOPES

    def test_email_scopes_includes_mail_readwrite(self) -> None:
        """Test email scopes include Mail.ReadWrite."""
        assert "https://graph.microsoft.com/Mail.ReadWrite" in OUTLOOK_EMAIL_SCOPES

    def test_email_scopes_includes_offline_access(self) -> None:
        """Test email scopes include offline_access."""
        assert "offline_access" in OUTLOOK_EMAIL_SCOPES


class TestOutlookPaths:
    """Tests for Outlook path functions."""

    def test_default_credentials_path_in_microsoft_dir(self) -> None:
        """Test default credentials path is in .microsoft directory."""
        assert ".microsoft" in str(DEFAULT_OUTLOOK_CREDENTIALS_PATH)
        assert "email_credentials.json" in str(DEFAULT_OUTLOOK_CREDENTIALS_PATH)

    def test_default_tokens_path_in_microsoft_dir(self) -> None:
        """Test default tokens path is in .microsoft directory."""
        assert ".microsoft" in str(DEFAULT_OUTLOOK_TOKENS_PATH)
        assert "email_tokens.json" in str(DEFAULT_OUTLOOK_TOKENS_PATH)

    def test_get_outlook_credentials_path_returns_default(self) -> None:
        """Test get_outlook_credentials_path returns the default path."""
        path = get_outlook_credentials_path()
        assert path == DEFAULT_OUTLOOK_CREDENTIALS_PATH

    def test_get_outlook_tokens_path_returns_default(self) -> None:
        """Test get_outlook_tokens_path returns the default path."""
        path = get_outlook_tokens_path()
        assert path == DEFAULT_OUTLOOK_TOKENS_PATH
