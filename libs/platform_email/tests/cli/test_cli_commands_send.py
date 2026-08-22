"""Email CLI: send and search commands."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime

import pytest

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


class TestCmdSend:
    """Tests for cmd_send command."""

    def test_shows_error_when_not_authenticated(self) -> None:
        """Test shows error when not authenticated."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        cli_commands.cmd_send("to@test.com", "Subject", "/body.txt")
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_shows_error_when_body_file_not_found(self) -> None:
        """Test shows error when body file does not exist."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.file_exists = lambda p: False

        cli_commands.cmd_send("to@test.com", "Subject", "/missing.txt")
        output = " ".join(messages)
        assert "Body file not found" in output
        assert "/missing.txt" in output

    def test_sends_email(self) -> None:
        """Test sends email when authenticated."""
        messages: list[str] = []
        post_calls: list[tuple[str, dict[str, str], str]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Hello from file"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        cli_commands.cmd_send("to@test.com", "Subject", "/body.txt")
        output = " ".join(messages)
        assert "Email sent" in output
        assert len(post_calls) == 1
        assert '"contentType":"Text"' in post_calls[0][2]
        assert "Hello from file" in post_calls[0][2]

    def test_sends_email_as_html(self) -> None:
        """Test sends email as HTML with pre tags when html=True."""
        messages: list[str] = []
        post_calls: list[tuple[str, dict[str, str], str]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Line1\nLine2"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        cli_commands.cmd_send("to@test.com", "Subject", "/body.txt", html=True)
        output = " ".join(messages)
        assert "Email sent" in output
        assert len(post_calls) == 1
        assert '"contentType":"HTML"' in post_calls[0][2]
        assert "<pre" in post_calls[0][2]
        assert "</pre>" in post_calls[0][2]
        assert "Line1" in post_calls[0][2]

    def test_sends_email_with_cc(self) -> None:
        """Test sends email with CC recipients in API payload."""
        post_calls: list[tuple[str, dict[str, str], str]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: None
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Body"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        cli_commands.cmd_send("to@test.com", "Subject", "/body.txt", cc="cc1@test.com,cc2@test.com")
        assert len(post_calls) == 1
        assert "cc1@test.com" in post_calls[0][2]
        assert "cc2@test.com" in post_calls[0][2]

    def test_sends_email_with_bcc(self) -> None:
        """Test sends email with BCC recipients in API payload."""
        post_calls: list[tuple[str, dict[str, str], str]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: None
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Body"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        cli_commands.cmd_send("to@test.com", "Subject", "/body.txt", bcc="bcc@test.com")
        assert len(post_calls) == 1
        assert "bcc@test.com" in post_calls[0][2]

    def test_success_message_includes_cc_and_bcc(self) -> None:
        """Test success message mentions CC and BCC when present."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Body"
        hooks.http_post = lambda u, h, b: ""

        cli_commands.cmd_send("to@test.com", "Subj", "/body.txt", cc="cc@x.com", bcc="bcc@x.com")
        output = " ".join(messages)
        assert "CC: cc@x.com" in output
        assert "BCC: bcc@x.com" in output

    def test_empty_cc_bcc_omitted_from_success_message(self) -> None:
        """Test success message does not mention CC/BCC when empty."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Body"
        hooks.http_post = lambda u, h, b: ""

        cli_commands.cmd_send("to@test.com", "Subj", "/body.txt")
        output = " ".join(messages)
        assert "Email sent to to@test.com" in output
        assert "CC" not in output
        assert "BCC" not in output

    def test_sends_email_with_attachments(self) -> None:
        """Test sends email with attachments in API payload."""
        post_calls: list[tuple[str, dict[str, str], str]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: None
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Body"
        hooks.read_file_bytes = lambda p: b"file content"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        cli_commands.cmd_send("to@test.com", "Subject", "/body.txt", attachments=("/path/doc.pdf",))
        assert len(post_calls) == 1
        assert "attachments" in post_calls[0][2]
        assert "#microsoft.graph.fileAttachment" in post_calls[0][2]
        assert "doc.pdf" in post_calls[0][2]

    def test_attachment_not_found_shows_error(self) -> None:
        """Test shows error when attachment file does not exist."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        file_map = {"/body.txt": True, "/missing.pdf": False}
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.file_exists = lambda p: file_map.get(p, False)

        cli_commands.cmd_send("to@test.com", "Subject", "/body.txt", attachments=("/missing.pdf",))
        output = " ".join(messages)
        assert "Attachment not found" in output
        assert "/missing.pdf" in output

    def test_success_message_includes_attachment_names(self) -> None:
        """Test success message lists attachment filenames."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Body"
        hooks.read_file_bytes = lambda p: b"data"
        hooks.http_post = lambda u, h, b: ""

        cli_commands.cmd_send(
            "to@test.com",
            "Subj",
            "/body.txt",
            attachments=("/path/doc.pdf", "/path/img.png"),
        )
        output = " ".join(messages)
        assert "Attachments: doc.pdf, img.png" in output

    def test_no_attachments_key_when_empty(self) -> None:
        """Test no attachments key in payload when no files attached."""
        post_calls: list[tuple[str, dict[str, str], str]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: None
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Body"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        cli_commands.cmd_send("to@test.com", "Subject", "/body.txt")
        assert len(post_calls) == 1
        assert "attachments" not in post_calls[0][2]


class TestCmdSearch:
    """Tests for cmd_search command."""

    def test_shows_error_when_not_authenticated(self) -> None:
        """Test shows error when not authenticated."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        cli_commands.cmd_search("test query")
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_displays_search_results(self) -> None:
        """Test displays matching emails from search."""
        messages: list[str] = []
        get_calls: list[tuple[str, dict[str, str]]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        search_response = (
            '{"value": [{"subject": "TU+11 Registration", "isRead": false, '
            '"from": {"emailAddress": {"address": "bergul@mit.edu"}}, '
            '"receivedDateTime": "2026-02-20T10:00:00Z"}]}'
        )

        def fake_get(url: str, headers: dict[str, str]) -> str:
            get_calls.append((url, headers))
            return search_response

        hooks.http_get = fake_get

        cli_commands.cmd_search("TU+11")
        output = " ".join(messages)
        assert "TU+11 Registration" in output
        assert "bergul@mit.edu" in output
        assert "$search=" in get_calls[0][0]

    def test_displays_no_results_message(self) -> None:
        """Test displays message when search returns no results."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.http_get = lambda u, h: '{"value": []}'

        cli_commands.cmd_search("nonexistent")
        output = " ".join(messages)
        assert "No results found" in output

    def test_handles_invalid_response(self) -> None:
        """Test handles invalid response from API."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.http_get = lambda u, h: '{"value": "not_a_list"}'

        cli_commands.cmd_search("test")
        output = " ".join(messages)
        assert "Invalid response" in output

    def test_respects_count_parameter(self) -> None:
        """Test passes count to API as $top parameter."""
        get_calls: list[tuple[str, dict[str, str]]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: None

        def fake_get(url: str, headers: dict[str, str]) -> str:
            get_calls.append((url, headers))
            return '{"value": []}'

        hooks.http_get = fake_get

        cli_commands.cmd_search("test", count=5)
        assert "$top=5" in get_calls[0][0]

    def test_skips_non_dict_messages(self) -> None:
        """Test skips non-dict items in search results."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        response = (
            '{"value": ["not_a_dict", {"subject": "Valid Result", "isRead": true, '
            '"from": {"emailAddress": {"address": "x@y.com"}}, '
            '"receivedDateTime": "2026-01-15"}]}'
        )
        hooks.http_get = lambda u, h: response

        cli_commands.cmd_search("test")
        output = " ".join(messages)
        assert "Valid Result" in output


# =============================================================================
# Argument Parsing
# =============================================================================
