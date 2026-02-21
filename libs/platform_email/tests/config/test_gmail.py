"""Tests for platform_email.config.gmail module."""

from __future__ import annotations

from platform_email.config.gmail import (
    DEFAULT_GMAIL_CREDENTIALS_PATH,
    DEFAULT_GMAIL_TOKENS_PATH,
    GMAIL_API_BASE,
    GMAIL_AUTH_URL,
    GMAIL_EMAIL_SCOPES,
    GMAIL_TOKEN_URL,
    get_gmail_credentials_path,
    get_gmail_tokens_path,
)


class TestGmailConstants:
    """Tests for Gmail configuration constants."""

    def test_auth_url(self) -> None:
        """Test auth URL is correct."""
        assert GMAIL_AUTH_URL == "https://accounts.google.com/o/oauth2/v2/auth"

    def test_token_url(self) -> None:
        """Test token URL is correct."""
        assert GMAIL_TOKEN_URL == "https://oauth2.googleapis.com/token"

    def test_api_base_url(self) -> None:
        """Test API base URL is correct."""
        assert GMAIL_API_BASE == "https://gmail.googleapis.com/gmail/v1"

    def test_email_scopes_includes_readonly(self) -> None:
        """Test email scopes include gmail.readonly."""
        assert "https://www.googleapis.com/auth/gmail.readonly" in GMAIL_EMAIL_SCOPES

    def test_email_scopes_includes_send(self) -> None:
        """Test email scopes include gmail.send."""
        assert "https://www.googleapis.com/auth/gmail.send" in GMAIL_EMAIL_SCOPES

    def test_email_scopes_includes_modify(self) -> None:
        """Test email scopes include gmail.modify."""
        assert "https://www.googleapis.com/auth/gmail.modify" in GMAIL_EMAIL_SCOPES


class TestGmailPaths:
    """Tests for Gmail path functions."""

    def test_default_credentials_path_in_google_dir(self) -> None:
        """Test default credentials path is in .google directory."""
        assert ".google" in str(DEFAULT_GMAIL_CREDENTIALS_PATH)
        assert "email_credentials.json" in str(DEFAULT_GMAIL_CREDENTIALS_PATH)

    def test_default_tokens_path_in_google_dir(self) -> None:
        """Test default tokens path is in .google directory."""
        assert ".google" in str(DEFAULT_GMAIL_TOKENS_PATH)
        assert "email_tokens.json" in str(DEFAULT_GMAIL_TOKENS_PATH)

    def test_get_gmail_credentials_path_returns_default(self) -> None:
        """Test get_gmail_credentials_path returns the default path."""
        path = get_gmail_credentials_path()
        assert path == DEFAULT_GMAIL_CREDENTIALS_PATH

    def test_get_gmail_tokens_path_returns_default(self) -> None:
        """Test get_gmail_tokens_path returns the default path."""
        path = get_gmail_tokens_path()
        assert path == DEFAULT_GMAIL_TOKENS_PATH
