"""Tests for production token/config loader hooks in platform_email.testing."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.errors import AppError, EmailErrorCode
from platform_core.json_utils import dump_json_str

from platform_email.testing import (
    _prod_load_gmail_credentials,
    _prod_load_gmail_tokens,
    _prod_load_outlook_config,
    _prod_load_outlook_tokens,
    _prod_save_gmail_tokens,
    _prod_save_outlook_tokens,
    hooks,
    make_fake_path,
    reset_hooks,
)
from platform_email.types import OAuthTokens, encode_oauth_tokens


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


def _make_tokens() -> OAuthTokens:
    """Create test tokens."""
    return OAuthTokens(
        access_token="test_access",
        refresh_token="test_refresh",
        expires_at=9999999999,
        token_type="Bearer",
    )


class TestProdLoadOutlookTokens:
    """Tests for _prod_load_outlook_tokens."""

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        """Test returns None when tokens file doesn't exist."""
        tokens_path = tmp_path / "nonexistent" / "tokens.json"
        hooks.outlook_tokens_path = make_fake_path(str(tokens_path))

        result = _prod_load_outlook_tokens()
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        """Test returns None for invalid JSON."""
        tokens_path = tmp_path / "tokens.json"
        tokens_path.write_text("not valid json", encoding="utf-8")
        hooks.outlook_tokens_path = make_fake_path(str(tokens_path))

        result = _prod_load_outlook_tokens()
        assert result is None

    def test_loads_valid_tokens(self, tmp_path: Path) -> None:
        """Test loads valid tokens from file."""
        tokens_path = tmp_path / "tokens.json"
        tokens = _make_tokens()
        content = dump_json_str(encode_oauth_tokens(tokens))
        tokens_path.write_text(content, encoding="utf-8")
        hooks.outlook_tokens_path = make_fake_path(str(tokens_path))

        result = _prod_load_outlook_tokens()
        if result is None:
            pytest.fail("Expected tokens but got None")
        assert result["access_token"] == "test_access"


class TestProdSaveOutlookTokens:
    """Tests for _prod_save_outlook_tokens."""

    def test_saves_tokens_to_file(self, tmp_path: Path) -> None:
        """Test saves tokens to file."""
        tokens_path = tmp_path / "outlook" / "tokens.json"
        tokens = _make_tokens()
        hooks.outlook_tokens_path = make_fake_path(str(tokens_path))

        _prod_save_outlook_tokens(tokens)
        assert tokens_path.exists()
        content = tokens_path.read_text(encoding="utf-8")
        assert "test_access" in content


class TestProdLoadOutlookConfig:
    """Tests for _prod_load_outlook_config."""

    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        """Test raises AppError when credentials file missing."""
        creds_path = tmp_path / "nonexistent" / "credentials.json"
        hooks.outlook_credentials_path = make_fake_path(str(creds_path))

        with pytest.raises(AppError) as exc_info:
            _prod_load_outlook_config()
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.CREDENTIALS_NOT_FOUND

    def test_raises_for_invalid_json(self, tmp_path: Path) -> None:
        """Test raises AppError for invalid JSON."""
        creds_path = tmp_path / "credentials.json"
        creds_path.write_text("not json", encoding="utf-8")
        hooks.outlook_credentials_path = make_fake_path(str(creds_path))

        with pytest.raises(AppError) as exc_info:
            _prod_load_outlook_config()
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.CREDENTIALS_NOT_FOUND

    def test_loads_valid_config(self, tmp_path: Path) -> None:
        """Test loads valid config from file."""
        creds_path = tmp_path / "credentials.json"
        config = {
            "client_id": "test_id",
            "client_secret": "test_secret",
            "redirect_uri": "http://localhost",
            "tenant_id": "common",
        }
        creds_path.write_text(dump_json_str(config), encoding="utf-8")
        hooks.outlook_credentials_path = make_fake_path(str(creds_path))

        result = _prod_load_outlook_config()
        assert result["client_id"] == "test_id"


class TestProdLoadGmailTokens:
    """Tests for _prod_load_gmail_tokens."""

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        """Test returns None when tokens file doesn't exist."""
        tokens_path = tmp_path / "nonexistent" / "tokens.json"
        hooks.gmail_tokens_path = make_fake_path(str(tokens_path))

        result = _prod_load_gmail_tokens()
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        """Test returns None for invalid JSON."""
        tokens_path = tmp_path / "tokens.json"
        tokens_path.write_text("invalid", encoding="utf-8")
        hooks.gmail_tokens_path = make_fake_path(str(tokens_path))

        result = _prod_load_gmail_tokens()
        assert result is None

    def test_loads_valid_tokens(self, tmp_path: Path) -> None:
        """Test loads valid tokens from file."""
        tokens_path = tmp_path / "tokens.json"
        tokens = _make_tokens()
        content = dump_json_str(encode_oauth_tokens(tokens))
        tokens_path.write_text(content, encoding="utf-8")
        hooks.gmail_tokens_path = make_fake_path(str(tokens_path))

        result = _prod_load_gmail_tokens()
        if result is None:
            pytest.fail("Expected tokens but got None")
        assert result["access_token"] == "test_access"


class TestProdSaveGmailTokens:
    """Tests for _prod_save_gmail_tokens."""

    def test_saves_tokens_to_file(self, tmp_path: Path) -> None:
        """Test saves tokens to file."""
        tokens_path = tmp_path / "gmail" / "tokens.json"
        tokens = _make_tokens()
        hooks.gmail_tokens_path = make_fake_path(str(tokens_path))

        _prod_save_gmail_tokens(tokens)
        assert tokens_path.exists()


class TestProdLoadGmailCredentials:
    """Tests for _prod_load_gmail_credentials."""

    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        """Test raises AppError when credentials file missing."""
        creds_path = tmp_path / "nonexistent" / "credentials.json"
        hooks.gmail_credentials_path = make_fake_path(str(creds_path))

        with pytest.raises(AppError) as exc_info:
            _prod_load_gmail_credentials()
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.CREDENTIALS_NOT_FOUND

    def test_raises_for_invalid_json(self, tmp_path: Path) -> None:
        """Test raises AppError for invalid JSON."""
        creds_path = tmp_path / "credentials.json"
        creds_path.write_text("not json", encoding="utf-8")
        hooks.gmail_credentials_path = make_fake_path(str(creds_path))

        with pytest.raises(AppError) as exc_info:
            _prod_load_gmail_credentials()
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.CREDENTIALS_NOT_FOUND

    def test_raises_for_missing_installed_section(self, tmp_path: Path) -> None:
        """Test raises AppError when installed section missing."""
        creds_path = tmp_path / "credentials.json"
        creds_path.write_text('{"web": {}}', encoding="utf-8")
        hooks.gmail_credentials_path = make_fake_path(str(creds_path))

        with pytest.raises(AppError) as exc_info:
            _prod_load_gmail_credentials()
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.CREDENTIALS_NOT_FOUND
        assert "installed" in err.message

    def test_loads_valid_credentials(self, tmp_path: Path) -> None:
        """Test loads valid Google credentials from file."""
        creds_path = tmp_path / "credentials.json"
        config = {
            "installed": {
                "client_id": "google_client_id",
                "client_secret": "google_secret",
                "redirect_uris": ["http://localhost"],
            },
        }
        creds_path.write_text(dump_json_str(config), encoding="utf-8")
        hooks.gmail_credentials_path = make_fake_path(str(creds_path))

        result = _prod_load_gmail_credentials()
        assert result["client_id"] == "google_client_id"
        assert result["redirect_uri"] == "http://localhost"

    def test_uses_default_redirect_uri_when_empty(self, tmp_path: Path) -> None:
        """Test uses default redirect_uri when list is empty."""
        creds_path = tmp_path / "credentials.json"
        config = {
            "installed": {
                "client_id": "id",
                "client_secret": "secret",
                "redirect_uris": [],
            },
        }
        creds_path.write_text(dump_json_str(config), encoding="utf-8")
        hooks.gmail_credentials_path = make_fake_path(str(creds_path))

        result = _prod_load_gmail_credentials()
        assert result["redirect_uri"] == "http://localhost"

    def test_uses_default_redirect_uri_when_non_string(self, tmp_path: Path) -> None:
        """Test uses default redirect_uri when first item is not string."""
        creds_path = tmp_path / "credentials.json"
        config = {
            "installed": {
                "client_id": "id",
                "client_secret": "secret",
                "redirect_uris": [123, "http://localhost"],
            },
        }
        creds_path.write_text(dump_json_str(config), encoding="utf-8")
        hooks.gmail_credentials_path = make_fake_path(str(creds_path))

        result = _prod_load_gmail_credentials()
        assert result["redirect_uri"] == "http://localhost"
