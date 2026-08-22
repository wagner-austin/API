"""Email CLI: Graph API call helpers and message display."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.json_utils import narrow_json_to_dict

from platform_email import cli_commands
from platform_email.testing import hooks, reset_hooks


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


# =============================================================================
# Token Types
# =============================================================================


class TestApiGet:
    """Tests for _api_get function."""

    def test_makes_get_request(self) -> None:
        """Test that _api_get makes proper GET request."""
        calls: list[tuple[str, dict[str, str]]] = []

        def fake_http_get(url: str, headers: dict[str, str]) -> str:
            calls.append((url, headers))
            return '{"data": "value"}'

        hooks.http_get = fake_http_get
        result = cli_commands._api_get("token123", "/me/messages")
        assert result["data"] == "value"
        assert len(calls) == 1
        assert "Bearer token123" in calls[0][1]["Authorization"]


class TestApiPost:
    """Tests for _api_post function."""

    def test_makes_post_request(self) -> None:
        """Test that _api_post makes proper POST request."""
        calls: list[tuple[str, dict[str, str], str]] = []

        def fake_http_post(url: str, headers: dict[str, str], body: str) -> str:
            calls.append((url, headers, body))
            return '{"result": "ok"}'

        hooks.http_post = fake_http_post
        result = cli_commands._api_post("token123", "/me/sendMail", {"message": {}})
        assert result["result"] == "ok"
        assert len(calls) == 1

    def test_returns_empty_dict_for_empty_response(self) -> None:
        """Test returns empty dict for empty response."""
        hooks.http_post = lambda u, h, b: "   "
        result = cli_commands._api_post("token", "/path", {})
        assert result == {}


# =============================================================================
# Display Helpers
# =============================================================================


class TestFormatRecipients:
    """Tests for _format_recipients function."""

    def test_empty_string_returns_empty_list(self) -> None:
        """Test empty string produces empty list."""
        result = cli_commands._format_recipients("")
        assert result == []

    def test_single_address(self) -> None:
        """Test single email address produces one recipient."""
        result = cli_commands._format_recipients("user@example.com")
        assert len(result) == 1
        assert result[0] == {"emailAddress": {"address": "user@example.com"}}

    def test_multiple_addresses(self) -> None:
        """Test comma-separated addresses produce multiple recipients."""
        result = cli_commands._format_recipients("a@b.com,c@d.com,e@f.com")
        assert len(result) == 3
        assert result[0] == {"emailAddress": {"address": "a@b.com"}}
        assert result[1] == {"emailAddress": {"address": "c@d.com"}}
        assert result[2] == {"emailAddress": {"address": "e@f.com"}}

    def test_strips_whitespace(self) -> None:
        """Test whitespace around addresses is stripped."""
        result = cli_commands._format_recipients("  a@b.com , c@d.com  ")
        assert len(result) == 2
        assert result[0] == {"emailAddress": {"address": "a@b.com"}}
        assert result[1] == {"emailAddress": {"address": "c@d.com"}}

    def test_skips_empty_entries(self) -> None:
        """Test empty entries from trailing commas are skipped."""
        result = cli_commands._format_recipients("a@b.com,,c@d.com,")
        assert len(result) == 2
        assert result[0] == {"emailAddress": {"address": "a@b.com"}}
        assert result[1] == {"emailAddress": {"address": "c@d.com"}}


class TestBuildAttachments:
    """Tests for _build_attachments function."""

    def test_single_file(self) -> None:
        """Test builds attachment object for a single file."""
        hooks.read_file_bytes = lambda p: b"hello world"

        result = cli_commands._build_attachments(("/path/to/document.pdf",))
        assert len(result) == 1
        att = narrow_json_to_dict(result[0])
        assert att["@odata.type"] == "#microsoft.graph.fileAttachment"
        assert att["name"] == "document.pdf"
        assert att["contentType"] == "application/pdf"
        assert att["contentBytes"] != ""

    def test_multiple_files(self) -> None:
        """Test builds attachment objects for multiple files."""
        hooks.read_file_bytes = lambda p: b"\x00\x01\x02"

        result = cli_commands._build_attachments(("/a/photo.png", "/b/notes.txt"))
        assert len(result) == 2
        att0 = narrow_json_to_dict(result[0])
        att1 = narrow_json_to_dict(result[1])
        assert att0["name"] == "photo.png"
        assert att0["contentType"] == "image/png"
        assert att1["name"] == "notes.txt"
        assert att1["contentType"] == "text/plain"

    def test_unknown_mime_type_defaults_to_octet_stream(self) -> None:
        """Test unknown file extension defaults to application/octet-stream."""
        hooks.read_file_bytes = lambda p: b"data"

        result = cli_commands._build_attachments(("/path/to/file.xyz123",))
        assert len(result) == 1
        att = narrow_json_to_dict(result[0])
        assert att["contentType"] == "application/octet-stream"

    def test_base64_encodes_content(self) -> None:
        """Test file content is base64-encoded."""
        import base64

        raw_bytes = b"\x89PNG\r\n\x1a\n"
        hooks.read_file_bytes = lambda p: raw_bytes

        result = cli_commands._build_attachments(("/img.png",))
        att = narrow_json_to_dict(result[0])
        expected_encoded = base64.b64encode(raw_bytes).decode("ascii")
        assert att["contentBytes"] == expected_encoded

    def test_empty_tuple_returns_empty_list(self) -> None:
        """Test empty tuple produces empty list."""
        result = cli_commands._build_attachments(())
        assert result == []


class TestDisplayMessageRows:
    """Tests for _display_message_rows function."""

    def test_renders_unread_message(self) -> None:
        """Test renders unread message with asterisk marker."""
        messages: list[str] = []
        hooks.console_output = lambda m: messages.append(m)

        cli_commands._display_message_rows(
            [
                {
                    "subject": "Test Subject",
                    "isRead": False,
                    "from": {"emailAddress": {"address": "sender@test.com"}},
                    "receivedDateTime": "2026-01-15T10:00:00Z",
                }
            ]
        )
        output = " ".join(messages)
        assert "Test Subject" in output
        assert "sender@test.com" in output
        assert "2026-01-15" in output

    def test_renders_read_message(self) -> None:
        """Test renders read message without asterisk marker."""
        messages: list[str] = []
        hooks.console_output = lambda m: messages.append(m)

        cli_commands._display_message_rows(
            [
                {
                    "subject": "Read Email",
                    "isRead": True,
                    "receivedDateTime": "2026-01-15",
                }
            ]
        )
        output = " ".join(messages)
        assert "Read Email" in output

    def test_truncates_long_subject(self) -> None:
        """Test subjects longer than 50 chars are truncated."""
        messages: list[str] = []
        hooks.console_output = lambda m: messages.append(m)

        long_subject = "A" * 60
        cli_commands._display_message_rows(
            [
                {
                    "subject": long_subject,
                    "isRead": True,
                    "receivedDateTime": "2026-01-15",
                }
            ]
        )
        output = " ".join(messages)
        assert "A" * 50 in output
        assert "A" * 51 not in output

    def test_handles_missing_from_field(self) -> None:
        """Test handles message with no from field."""
        messages: list[str] = []
        hooks.console_output = lambda m: messages.append(m)

        cli_commands._display_message_rows(
            [
                {
                    "subject": "No Sender",
                    "isRead": True,
                    "receivedDateTime": "2026-01-15",
                }
            ]
        )
        output = " ".join(messages)
        assert "No Sender" in output

    def test_handles_non_dict_email_address(self) -> None:
        """Test handles emailAddress that is not a dict."""
        messages: list[str] = []
        hooks.console_output = lambda m: messages.append(m)

        cli_commands._display_message_rows(
            [
                {
                    "subject": "Bad From",
                    "isRead": True,
                    "from": {"emailAddress": "not_a_dict"},
                    "receivedDateTime": "2026-01-15",
                }
            ]
        )
        output = " ".join(messages)
        assert "Bad From" in output


# =============================================================================
# Commands
# =============================================================================
