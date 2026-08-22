"""Tests for factory functions in platform_email.testing module."""

from __future__ import annotations

import pytest

from platform_email.fake_hooks import (
    make_fake_attachment,
    make_fake_console,
    make_fake_current_time,
    make_fake_draft,
    make_fake_email,
    make_fake_file_system,
    make_fake_folder,
    make_fake_gmail_credentials,
    make_fake_http_delete,
    make_fake_http_get,
    make_fake_http_patch,
    make_fake_http_post,
    make_fake_no_tokens,
    make_fake_outlook_config,
    make_fake_tokens,
    make_raising_http_delete,
    make_raising_http_get,
    make_raising_http_patch,
    make_raising_http_post,
)
from platform_email.types import OAuthCredentials, OAuthTokens, OutlookOAuthConfig

# =============================================================================
# HTTP Hook Factory Tests
# =============================================================================


class TestMakeFakeHttpGet:
    """Tests for make_fake_http_get."""

    def test_returns_fixed_response(self) -> None:
        """Test that the hook returns the fixed response."""
        hook = make_fake_http_get('{"result": "ok"}')
        result = hook("https://api.example.com/data", {"Authorization": "Bearer token"})
        assert result == '{"result": "ok"}'


class TestMakeFakeHttpPost:
    """Tests for make_fake_http_post."""

    def test_returns_fixed_response(self) -> None:
        """Test that the hook returns the fixed response."""
        hook = make_fake_http_post('{"id": "123"}')
        result = hook(
            "https://api.example.com/create",
            {"Content-Type": "application/json"},
            '{"name": "test"}',
        )
        assert result == '{"id": "123"}'


class TestMakeFakeHttpPatch:
    """Tests for make_fake_http_patch."""

    def test_returns_fixed_response(self) -> None:
        """Test that the hook returns the fixed response."""
        hook = make_fake_http_patch('{"updated": true}')
        result = hook(
            "https://api.example.com/update",
            {"Content-Type": "application/json"},
            '{"field": "value"}',
        )
        assert result == '{"updated": true}'


class TestMakeFakeHttpDelete:
    """Tests for make_fake_http_delete."""

    def test_returns_none(self) -> None:
        """Test that the hook returns None."""
        hook = make_fake_http_delete()
        result = hook("https://api.example.com/delete/123", {"Authorization": "Bearer token"})
        assert result is None


# =============================================================================
# Raising HTTP Hook Factory Tests
# =============================================================================


class TestMakeRaisingHttpGet:
    """Tests for make_raising_http_get."""

    def test_raises_exception(self) -> None:
        """Test that the hook raises the specified exception."""
        hook = make_raising_http_get(ConnectionError("Network error"))
        with pytest.raises(ConnectionError) as exc_info:
            hook("https://api.example.com/data", {})
        assert "Network error" in str(exc_info.value)


class TestMakeRaisingHttpPost:
    """Tests for make_raising_http_post."""

    def test_raises_exception(self) -> None:
        """Test that the hook raises the specified exception."""
        hook = make_raising_http_post(OSError("Request failed"))
        with pytest.raises(OSError) as exc_info:
            hook("https://api.example.com/data", {}, "{}")
        assert "Request failed" in str(exc_info.value)


class TestMakeRaisingHttpPatch:
    """Tests for make_raising_http_patch."""

    def test_raises_exception(self) -> None:
        """Test that the hook raises the specified exception."""
        hook = make_raising_http_patch(TimeoutError("Timeout"))
        with pytest.raises(TimeoutError) as exc_info:
            hook("https://api.example.com/data", {}, "{}")
        assert "Timeout" in str(exc_info.value)


class TestMakeRaisingHttpDelete:
    """Tests for make_raising_http_delete."""

    def test_raises_exception(self) -> None:
        """Test that the hook raises the specified exception."""
        hook = make_raising_http_delete(PermissionError("Forbidden"))
        with pytest.raises(PermissionError) as exc_info:
            hook("https://api.example.com/delete/123", {})
        assert "Forbidden" in str(exc_info.value)


# =============================================================================
# Token Hook Factory Tests
# =============================================================================


class TestMakeFakeTokens:
    """Tests for make_fake_tokens."""

    def test_returns_tokens(self) -> None:
        """Test that the hook returns the tokens."""
        tokens = OAuthTokens(
            access_token="access123",
            refresh_token="refresh456",
            expires_at=9999999999,
            token_type="Bearer",
        )
        hook = make_fake_tokens(tokens)
        result = hook()
        assert result
        assert result["access_token"] == "access123"


class TestMakeFakeNoTokens:
    """Tests for make_fake_no_tokens."""

    def test_returns_none(self) -> None:
        """Test that the hook returns None."""
        hook = make_fake_no_tokens()
        result = hook()
        assert result is None


# =============================================================================
# Config Hook Factory Tests
# =============================================================================


class TestMakeFakeOutlookConfig:
    """Tests for make_fake_outlook_config."""

    def test_returns_config(self) -> None:
        """Test that the hook returns the config."""
        config = OutlookOAuthConfig(
            client_id="client123",
            client_secret="secret456",
            redirect_uri="http://localhost",
            tenant_id="common",
        )
        hook = make_fake_outlook_config(config)
        result = hook()
        assert result["client_id"] == "client123"


class TestMakeFakeGmailCredentials:
    """Tests for make_fake_gmail_credentials."""

    def test_returns_credentials(self) -> None:
        """Test that the hook returns the credentials."""
        creds = OAuthCredentials(
            client_id="google_client",
            client_secret="google_secret",
            redirect_uri="urn:ietf:wg:oauth:2.0:oob",
        )
        hook = make_fake_gmail_credentials(creds)
        result = hook()
        assert result["client_id"] == "google_client"


# =============================================================================
# Time Hook Factory Tests
# =============================================================================


class TestMakeFakeCurrentTime:
    """Tests for make_fake_current_time."""

    def test_returns_timestamp(self) -> None:
        """Test that the hook returns the fixed timestamp."""
        hook = make_fake_current_time(1704067200)  # 2024-01-01T00:00:00Z
        result = hook()
        assert result == 1704067200


# =============================================================================
# File System Hook Factory Tests
# =============================================================================


class TestMakeFakeFileSystem:
    """Tests for make_fake_file_system."""

    def test_read_returns_content(self) -> None:
        """Test reading a file returns its content."""
        read_hook, _write_hook, _exists_hook = make_fake_file_system(
            {
                "/path/to/file.txt": "Hello World",
            }
        )
        assert read_hook("/path/to/file.txt") == "Hello World"

    def test_read_raises_for_not_found(self) -> None:
        """Test reading nonexistent file raises FileNotFoundError."""
        read_hook, _write_hook, _exists_hook = make_fake_file_system({})
        with pytest.raises(FileNotFoundError):
            read_hook("/nonexistent.txt")

    def test_write_stores_content(self) -> None:
        """Test writing stores content that can be read."""
        read_hook, write_hook, _exists_hook = make_fake_file_system({})
        write_hook("/new/file.txt", "New content")
        assert read_hook("/new/file.txt") == "New content"

    def test_exists_returns_true_for_existing(self) -> None:
        """Test exists returns True for existing files."""
        _read_hook, _write_hook, exists_hook = make_fake_file_system(
            {
                "/existing.txt": "content",
            }
        )
        assert exists_hook("/existing.txt") is True

    def test_exists_returns_false_for_missing(self) -> None:
        """Test exists returns False for missing files."""
        _read_hook, _write_hook, exists_hook = make_fake_file_system({})
        assert exists_hook("/missing.txt") is False


# =============================================================================
# Console Hook Factory Tests
# =============================================================================


class TestMakeFakeConsole:
    """Tests for make_fake_console."""

    def test_input_returns_values_in_order(self) -> None:
        """Test input hook returns values in order."""
        _output_hook, input_hook = make_fake_console(["first", "second", "third"])
        assert input_hook("Prompt 1: ") == "first"
        assert input_hook("Prompt 2: ") == "second"
        assert input_hook("Prompt 3: ") == "third"

    def test_input_returns_empty_when_exhausted(self) -> None:
        """Test input hook returns empty string when inputs exhausted."""
        _output_hook, input_hook = make_fake_console(["only"])
        assert input_hook("Prompt: ") == "only"
        assert input_hook("Next: ") == ""


# =============================================================================
# Data Factory Tests
# =============================================================================


class TestMakeFakeEmail:
    """Tests for make_fake_email."""

    def test_creates_email_with_defaults(self) -> None:
        """Test creating email with default values."""
        email = make_fake_email()
        assert email["id"] == "test_email_1"
        assert email["folder_id"] == "inbox"
        assert email["body_type"] == "text"
        assert email["importance"] == "normal"

    def test_creates_email_with_custom_values(self) -> None:
        """Test creating email with custom values."""
        email = make_fake_email(
            email_id="custom_id",
            subject="Custom Subject",
            body="<p>HTML</p>",
            body_type="html",
            importance="high",
        )
        assert email["id"] == "custom_id"
        assert email["subject"] == "Custom Subject"
        assert email["body_type"] == "html"
        assert email["importance"] == "high"

    def test_creates_email_with_recipients(self) -> None:
        """Test creating email with multiple recipients."""
        email = make_fake_email(
            to=("a@test.com", "b@test.com"),
            cc=("c@test.com",),
            bcc=("d@test.com", "e@test.com"),
        )
        assert len(email["to"]) == 2
        assert len(email["cc"]) == 1
        assert len(email["bcc"]) == 2

    def test_creates_email_with_low_importance(self) -> None:
        """Test creating email with low importance."""
        email = make_fake_email(importance="low")
        assert email["importance"] == "low"


class TestMakeFakeFolder:
    """Tests for make_fake_folder."""

    def test_creates_folder_with_defaults(self) -> None:
        """Test creating folder with default values."""
        folder = make_fake_folder()
        assert folder["id"] == "inbox"
        assert folder["name"] == "Inbox"
        assert folder["folder_type"] == "inbox"

    def test_creates_folder_with_custom_values(self) -> None:
        """Test creating folder with custom values."""
        folder = make_fake_folder(
            folder_id="custom_folder",
            name="My Custom Folder",
            folder_type="custom",
            unread_count=5,
            total_count=100,
        )
        assert folder["id"] == "custom_folder"
        assert folder["name"] == "My Custom Folder"
        assert folder["folder_type"] == "custom"
        assert folder["unread_count"] == 5
        assert folder["total_count"] == 100


class TestMakeFakeAttachment:
    """Tests for make_fake_attachment."""

    def test_creates_attachment_with_defaults(self) -> None:
        """Test creating attachment with default values."""
        attachment = make_fake_attachment()
        assert attachment["id"] == "attachment_1"
        assert attachment["name"] == "document.pdf"
        assert attachment["content_type"] == "application/pdf"
        assert attachment["size"] == 1024
        assert attachment["content_bytes"] is None

    def test_creates_attachment_with_content(self) -> None:
        """Test creating attachment with content bytes."""
        attachment = make_fake_attachment(
            attachment_id="img_att",
            name="image.png",
            content_type="image/png",
            size=2048,
            content_bytes="iVBORw0KGgo=",
        )
        assert attachment["id"] == "img_att"
        assert attachment["content_bytes"] == "iVBORw0KGgo="


class TestMakeFakeDraft:
    """Tests for make_fake_draft."""

    def test_creates_draft_with_defaults(self) -> None:
        """Test creating draft with default values."""
        draft = make_fake_draft()
        assert draft["id"] == "draft_1"
        assert draft["subject"] == "Draft Subject"
        assert draft["body_type"] == "text"

    def test_creates_draft_with_custom_values(self) -> None:
        """Test creating draft with custom values."""
        draft = make_fake_draft(
            draft_id="my_draft",
            subject="Custom Draft",
            body="<html>Draft</html>",
            body_type="html",
            to=("a@test.com", "b@test.com"),
        )
        assert draft["id"] == "my_draft"
        assert draft["body_type"] == "html"
        assert len(draft["to"]) == 2

    def test_creates_draft_with_cc_and_bcc(self) -> None:
        """Test creating draft with CC and BCC recipients."""
        draft = make_fake_draft(
            draft_id="draft_cc_bcc",
            subject="Draft with CC/BCC",
            body="Body",
            to=("to@test.com",),
            cc=("cc1@test.com", "cc2@test.com"),
            bcc=("bcc@test.com",),
        )
        assert len(draft["cc"]) == 2
        assert draft["cc"][0]["address"] == "cc1@test.com"
        assert draft["cc"][1]["address"] == "cc2@test.com"
        assert len(draft["bcc"]) == 1
        assert draft["bcc"][0]["address"] == "bcc@test.com"
