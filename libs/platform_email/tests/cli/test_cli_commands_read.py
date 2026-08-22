"""Email CLI: auth/folders/list/read commands."""

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


class TestCmdAuth:
    """Tests for cmd_auth command."""

    def test_shows_error_when_no_credentials(self) -> None:
        """Test shows error when credentials missing."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        cli_commands.cmd_auth()
        output = " ".join(messages)
        assert "Missing credentials" in output

    def test_generates_auth_url(self) -> None:
        """Test generates auth URL with credentials."""
        messages: list[str] = []
        inputs: list[str] = [""]  # Empty code to abort
        input_idx = [0]

        def fake_output(msg: str) -> None:
            messages.append(msg)

        def fake_input(prompt: str) -> str:
            idx = input_idx[0]
            input_idx[0] += 1
            return inputs[idx] if idx < len(inputs) else ""

        env = {
            "OUTLOOK_CLIENT_ID": "test_client_id",
            "OUTLOOK_CLIENT_SECRET": "test_secret",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.console_output = fake_output
        hooks.console_input = fake_input

        cli_commands.cmd_auth()
        output = " ".join(messages)
        assert "login.microsoftonline.com" in output
        assert "No code provided" in output

    def test_successful_auth_flow(self) -> None:
        """Test successful authorization flow."""
        messages: list[str] = []
        env: dict[str, str] = {
            "OUTLOOK_CLIENT_ID": "test_client_id",
            "OUTLOOK_CLIENT_SECRET": "test_secret",
        }
        env_updates: list[tuple[str, str]] = []

        def fake_output(msg: str) -> None:
            messages.append(msg)

        def fake_input(prompt: str) -> str:
            return "auth_code_123"

        def fake_set_env(k: str, v: str) -> None:
            env_updates.append((k, v))
            env[k] = v

        def fake_http_post(url: str, headers: dict[str, str], body: str) -> str:
            return """{
                "access_token": "new_token",
                "refresh_token": "new_refresh",
                "expires_in": 3600,
                "token_type": "Bearer"
            }"""

        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_set_env = fake_set_env
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = fake_output
        hooks.console_input = fake_input
        hooks.http_post = fake_http_post

        cli_commands.cmd_auth()
        output = " ".join(messages)
        assert "Authorization successful" in output
        # Verify tokens were saved
        token_keys = [k for k, v in env_updates]
        assert "OUTLOOK_ACCESS_TOKEN" in token_keys


class TestCmdFolders:
    """Tests for cmd_folders command."""

    def test_shows_error_when_not_authenticated(self) -> None:
        """Test shows error when not authenticated."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        cli_commands.cmd_folders()
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_lists_folders(self) -> None:
        """Test lists folders when authenticated."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        folder_response = (
            '{"value": [{"displayName": "Inbox", "unreadItemCount": 5, "totalItemCount": 10}]}'
        )
        hooks.http_get = lambda u, h: folder_response

        cli_commands.cmd_folders()
        output = " ".join(messages)
        assert "Inbox" in output
        assert "5 unread" in output

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

        cli_commands.cmd_folders()
        output = " ".join(messages)
        assert "Invalid response" in output

    def test_skips_non_dict_folders(self) -> None:
        """Test skips non-dict items in folders list."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        folder_response = '{"value": ["not_a_dict", {"displayName": "Valid"}]}'
        hooks.http_get = lambda u, h: folder_response

        cli_commands.cmd_folders()
        output = " ".join(messages)
        assert "Valid" in output

    def test_folder_without_unread(self) -> None:
        """Test folder with zero unread shows no unread count."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        folder_response = (
            '{"value": [{"displayName": "Sent", "unreadItemCount": 0, "totalItemCount": 5}]}'
        )
        hooks.http_get = lambda u, h: folder_response

        cli_commands.cmd_folders()
        output = " ".join(messages)
        assert "Sent" in output
        assert "unread" not in output.lower()


class TestCmdList:
    """Tests for cmd_list command."""

    def test_shows_error_when_not_authenticated(self) -> None:
        """Test shows error when not authenticated."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        cli_commands.cmd_list()
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_lists_emails(self) -> None:
        """Test lists emails when authenticated."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        email_response = (
            '{"value": [{"subject": "Test Email", "isRead": false, '
            '"from": {"emailAddress": {"address": "sender@test.com"}}, '
            '"receivedDateTime": "2025-01-15T10:00:00Z"}]}'
        )
        hooks.http_get = lambda u, h: email_response

        cli_commands.cmd_list()
        output = " ".join(messages)
        assert "Test Email" in output
        assert "sender@test.com" in output

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

        cli_commands.cmd_list()
        output = " ".join(messages)
        assert "Invalid response" in output

    def test_skips_non_dict_messages(self) -> None:
        """Test skips non-dict items in messages list."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        # Contains non-dict item and valid message
        email_response = (
            '{"value": ["not_a_dict", {"subject": "Valid", "isRead": true, '
            '"from": {"emailAddress": {"address": "x@y.com"}}, '
            '"receivedDateTime": "2025-01-15"}]}'
        )
        hooks.http_get = lambda u, h: email_response

        cli_commands.cmd_list()
        output = " ".join(messages)
        assert "Valid" in output

    def test_read_message_displays_marker(self) -> None:
        """Test read message shows different styling."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        email_response = (
            '{"value": [{"subject": "Read Email", "isRead": true, '
            '"receivedDateTime": "2025-01-15"}]}'
        )
        hooks.http_get = lambda u, h: email_response

        cli_commands.cmd_list()
        # Output should not have unread marker '*'
        output = " ".join(messages)
        assert "Read Email" in output

    def test_handles_non_dict_email_address(self) -> None:
        """Test handles emailAddress that is not a dict."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        # emailAddress is not a dict
        email_response = (
            '{"value": [{"subject": "Test", "isRead": true, '
            '"from": {"emailAddress": "not_a_dict"}, '
            '"receivedDateTime": "2025-01-15"}]}'
        )
        hooks.http_get = lambda u, h: email_response

        cli_commands.cmd_list()
        output = " ".join(messages)
        assert "Test" in output


class TestCmdRead:
    """Tests for cmd_read command."""

    def test_shows_error_when_not_authenticated(self) -> None:
        """Test shows error when not authenticated."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        cli_commands.cmd_read(1)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_shows_error_for_invalid_index(self) -> None:
        """Test shows error for invalid index."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.http_get = lambda u, h: '{"value": []}'

        cli_commands.cmd_read(1)
        output = " ".join(messages)
        assert "Invalid index" in output

    def test_handles_invalid_message_list(self) -> None:
        """Test handles invalid message list."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.http_get = lambda u, h: '{"value": "not_a_list"}'

        cli_commands.cmd_read(1)
        output = " ".join(messages)
        assert "Invalid response" in output

    def test_handles_non_dict_message(self) -> None:
        """Test handles non-dict message in list."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.http_get = lambda u, h: '{"value": ["not_a_dict"]}'

        cli_commands.cmd_read(1)
        output = " ".join(messages)
        assert "Invalid message" in output

    def test_reads_email_successfully(self) -> None:
        """Test reading an email displays content."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        get_count = [0]

        def fake_get(url: str, headers: dict[str, str]) -> str:
            get_count[0] += 1
            if get_count[0] == 1:
                # First call: list messages
                return '{"value": [{"id": "msg1", "subject": "Test"}]}'
            # Second call: get specific message
            return (
                '{"id": "msg1", "subject": "Test Subject", '
                '"from": {"emailAddress": {"address": "from@test.com", "name": "Sender"}}, '
                '"receivedDateTime": "2025-01-15T10:00:00Z", '
                '"body": {"content": "Hello World", "contentType": "text"}}'
            )

        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.http_get = fake_get

        cli_commands.cmd_read(1)
        output = " ".join(messages)
        assert "Test Subject" in output
        assert "Hello World" in output
        assert "from@test.com" in output

    def test_strips_html_from_body(self) -> None:
        """Test HTML tags are stripped from body."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        get_count = [0]

        def fake_get(url: str, headers: dict[str, str]) -> str:
            get_count[0] += 1
            if get_count[0] == 1:
                return '{"value": [{"id": "msg1"}]}'
            return (
                '{"id": "msg1", "subject": "HTML Email", '
                '"from": {"emailAddress": {"address": "a@b.com"}}, '
                '"receivedDateTime": "2025-01-15T10:00:00Z", '
                '"body": {"content": "<html><body><p>Paragraph</p></body></html>", '
                '"contentType": "html"}}'
            )

        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.http_get = fake_get

        cli_commands.cmd_read(1)
        output = " ".join(messages)
        assert "Paragraph" in output
        assert "<p>" not in output

    def test_truncates_long_body(self) -> None:
        """Test long body is truncated."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        get_count = [0]
        long_body = "A" * 3000  # Exceeds 2000 char limit

        def fake_get(url: str, headers: dict[str, str]) -> str:
            get_count[0] += 1
            if get_count[0] == 1:
                return '{"value": [{"id": "msg1"}]}'
            return (
                '{"id": "msg1", "subject": "Long", '
                '"from": {"emailAddress": {"address": "a@b.com"}}, '
                '"receivedDateTime": "2025-01-15", '
                f'"body": {{"content": "{long_body}", "contentType": "text"}}}}'
            )

        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.http_get = fake_get

        cli_commands.cmd_read(1)
        output = " ".join(messages)
        assert "truncated" in output

    def test_handles_non_dict_from(self) -> None:
        """Test handles from field that is not a dict."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        get_count = [0]

        def fake_get(url: str, headers: dict[str, str]) -> str:
            get_count[0] += 1
            if get_count[0] == 1:
                return '{"value": [{"id": "msg1"}]}'
            # from is not a dict
            return (
                '{"id": "msg1", "subject": "Test", '
                '"from": "not_a_dict", '
                '"receivedDateTime": "2025-01-15", '
                '"body": {"content": "Body", "contentType": "text"}}'
            )

        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.http_get = fake_get

        cli_commands.cmd_read(1)
        output = " ".join(messages)
        assert "Test" in output

    def test_handles_non_dict_body(self) -> None:
        """Test handles body field that is not a dict."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        get_count = [0]

        def fake_get(url: str, headers: dict[str, str]) -> str:
            get_count[0] += 1
            if get_count[0] == 1:
                return '{"value": [{"id": "msg1"}]}'
            # body is not a dict
            return (
                '{"id": "msg1", "subject": "Test", '
                '"from": {"emailAddress": {"address": "a@b.com"}}, '
                '"receivedDateTime": "2025-01-15", '
                '"body": "not_a_dict"}'
            )

        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.http_get = fake_get

        cli_commands.cmd_read(1)
        output = " ".join(messages)
        assert "Test" in output

    def test_handles_non_dict_email_address_in_read(self) -> None:
        """Test handles emailAddress that is not a dict in cmd_read."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        get_count = [0]

        def fake_get(url: str, headers: dict[str, str]) -> str:
            get_count[0] += 1
            if get_count[0] == 1:
                return '{"value": [{"id": "msg1"}]}'
            # emailAddress is not a dict
            return (
                '{"id": "msg1", "subject": "Test", '
                '"from": {"emailAddress": "not_a_dict"}, '
                '"receivedDateTime": "2025-01-15", '
                '"body": {"content": "Body", "contentType": "text"}}'
            )

        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.http_get = fake_get

        cli_commands.cmd_read(1)
        output = " ".join(messages)
        assert "Test" in output
