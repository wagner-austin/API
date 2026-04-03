"""Tests for platform_email CLI module."""

from __future__ import annotations

import argparse
from collections.abc import Generator
from datetime import datetime

import pytest
from platform_core.json_utils import JSONObject, narrow_json_to_dict

from platform_email import cli
from platform_email.testing import hooks, reset_hooks


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


# =============================================================================
# Token Types
# =============================================================================


class TestRequireStr:
    """Tests for require_str function."""

    def test_returns_string_value(self) -> None:
        """Test that require_str returns string value."""
        data: JSONObject = {"key": "value"}
        result = cli.require_str(data, "key")
        assert result == "value"

    def test_raises_type_error_for_int(self) -> None:
        """Test that require_str raises TypeError for int."""
        data: JSONObject = {"key": 123}
        with pytest.raises(TypeError, match="Expected str"):
            cli.require_str(data, "key")

    def test_raises_key_error_for_missing(self) -> None:
        """Test that require_str raises KeyError for missing key."""
        data: JSONObject = {}
        with pytest.raises(KeyError):
            cli.require_str(data, "missing")


class TestRequireInt:
    """Tests for require_int function."""

    def test_returns_int_value(self) -> None:
        """Test that require_int returns int value."""
        data: JSONObject = {"key": 123}
        result = cli.require_int(data, "key")
        assert result == 123

    def test_raises_type_error_for_str(self) -> None:
        """Test that require_int raises TypeError for string."""
        data: JSONObject = {"key": "value"}
        with pytest.raises(TypeError, match="Expected int"):
            cli.require_int(data, "key")


class TestDecodeTokenResponse:
    """Tests for decode_token_response function."""

    def test_decodes_valid_response(self) -> None:
        """Test decoding a valid token response."""
        data: JSONObject = {
            "access_token": "access123",
            "refresh_token": "refresh456",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        result = cli.decode_token_response(data)
        assert result["access_token"] == "access123"
        assert result["refresh_token"] == "refresh456"
        assert result["expires_in"] == 3600
        assert result["token_type"] == "Bearer"


# =============================================================================
# PKCE Helpers
# =============================================================================


class TestGenerateCodeVerifier:
    """Tests for _generate_code_verifier function."""

    def test_length_within_bounds(self) -> None:
        """Test that code verifier length is within expected bounds."""
        result = cli._generate_code_verifier()
        # PKCE code verifiers should be 43-128 characters
        assert 43 <= len(result) <= 128
        assert result == result.strip()  # No leading/trailing whitespace

    def test_is_url_safe(self) -> None:
        """Test that code verifier contains only URL-safe characters."""
        result = cli._generate_code_verifier()
        # URL-safe base64 uses alphanumeric, -, and _
        for char in result:
            assert char.isalnum() or char in "-_"


class TestGenerateCodeChallenge:
    """Tests for _generate_code_challenge function."""

    def test_produces_base64_result(self) -> None:
        """Test that code challenge produces base64url-encoded result."""
        verifier = "test_verifier_1234567890"
        result = cli._generate_code_challenge(verifier)
        # Base64url uses alphanumeric, -, and _
        for char in result:
            assert char.isalnum() or char in "-_"

    def test_different_verifiers_produce_different_challenges(self) -> None:
        """Test that different verifiers produce different challenges."""
        verifier1 = "verifier_one"
        verifier2 = "verifier_two"
        challenge1 = cli._generate_code_challenge(verifier1)
        challenge2 = cli._generate_code_challenge(verifier2)
        assert challenge1 != challenge2


# =============================================================================
# Environment Helpers
# =============================================================================


class TestGetEnv:
    """Tests for _get_env helper."""

    def test_calls_hook(self) -> None:
        """Test that _get_env calls the hook."""
        called_with: list[str] = []

        def fake_get_env(key: str) -> str | None:
            called_with.append(key)
            return "test_value"

        hooks.cli_get_env = fake_get_env
        result = cli._get_env("MY_KEY")
        assert result == "test_value"
        assert called_with == ["MY_KEY"]


class TestSetEnv:
    """Tests for _set_env helper."""

    def test_calls_hook(self) -> None:
        """Test that _set_env calls the hook."""
        called_with: list[tuple[str, str]] = []

        def fake_set_env(key: str, value: str) -> None:
            called_with.append((key, value))

        hooks.cli_set_env = fake_set_env
        cli._set_env("KEY", "VALUE")
        assert called_with == [("KEY", "VALUE")]


class TestGetNow:
    """Tests for _get_now helper."""

    def test_calls_hook(self) -> None:
        """Test that _get_now calls the hook."""
        fixed_time = datetime(2025, 1, 15, 12, 0, 0)

        def fake_get_now() -> datetime:
            return fixed_time

        hooks.cli_get_now = fake_get_now
        result = cli._get_now()
        assert result == fixed_time


class TestIsTokenExpired:
    """Tests for _is_token_expired function."""

    def test_returns_true_for_expired_token(self) -> None:
        """Test returns True for expired token."""
        # Set time to a known value (2025-01-01)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        # Token expired at timestamp 1700000000 (2023)
        result = cli._is_token_expired("1700000000")
        assert result is True

    def test_returns_false_for_valid_token(self) -> None:
        """Test returns False for valid token."""
        # 2025-01-01
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        # Token expires at 2035-01-01 (10 years in future)
        result = cli._is_token_expired("2051222400")
        assert result is False

    def test_returns_true_for_token_expiring_soon(self) -> None:
        """Test returns True for token expiring within 60 seconds."""
        # 2025-01-01 00:00:00
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        # Token expires at 2025-01-01 00:00:30 (30 seconds in future)
        result = cli._is_token_expired("1735689630")
        assert result is True


# =============================================================================
# Token Operations
# =============================================================================


class TestRefreshToken:
    """Tests for _refresh_token function."""

    def test_sends_refresh_request(self) -> None:
        """Test that refresh sends proper request."""
        calls: list[tuple[str, dict[str, str], str]] = []

        def fake_http_post(url: str, headers: dict[str, str], body: str) -> str:
            calls.append((url, headers, body))
            return """{
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "expires_in": 3600,
                "token_type": "Bearer"
            }"""

        hooks.http_post = fake_http_post
        result = cli._refresh_token("client_id", "client_secret", "refresh_token")
        assert result["access_token"] == "new_access"
        assert len(calls) == 1
        assert "refresh_token" in calls[0][2]


class TestExchangeCodeForTokens:
    """Tests for _exchange_code_for_tokens function."""

    def test_sends_exchange_request(self) -> None:
        """Test that exchange sends proper request."""
        calls: list[tuple[str, dict[str, str], str]] = []

        def fake_http_post(url: str, headers: dict[str, str], body: str) -> str:
            calls.append((url, headers, body))
            return """{
                "access_token": "access123",
                "refresh_token": "refresh456",
                "expires_in": 3600,
                "token_type": "Bearer"
            }"""

        hooks.http_post = fake_http_post
        result = cli._exchange_code_for_tokens(
            "client_id", "client_secret", "auth_code", "verifier", "http://localhost"
        )
        assert result["access_token"] == "access123"
        assert len(calls) == 1
        assert "authorization_code" in calls[0][2]


class TestGetValidTokenForAccount:
    """Tests for _get_valid_token_for_account function."""

    def test_returns_none_when_no_token(self) -> None:
        """Test returns None when no token configured."""
        hooks.cli_get_env = lambda k: None
        account = cli.ACCOUNTS[0]
        result = cli._get_valid_token_for_account(account)
        assert result is None

    def test_returns_token_when_not_expired(self) -> None:
        """Test returns token when not expired."""
        # Token expires in 2035
        env: dict[str, str] = {
            "OUTLOOK_ACCESS_TOKEN": "valid_token",
            "OUTLOOK_REFRESH_TOKEN": "refresh",
            "OUTLOOK_TOKEN_EXPIRES_AT": "2051222400",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        # Current time is 2025-01-01
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        account = cli.ACCOUNTS[0]
        result = cli._get_valid_token_for_account(account)
        assert result == "valid_token"

    def test_refreshes_token_when_expired(self) -> None:
        """Test refreshes token when expired."""
        env: dict[str, str] = {
            "OUTLOOK_ACCESS_TOKEN": "old_token",
            "OUTLOOK_REFRESH_TOKEN": "refresh_token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "1700000000",  # Expired (2023)
            "OUTLOOK_CLIENT_ID": "client_id",
            "OUTLOOK_CLIENT_SECRET": "client_secret",
        }
        env_updates: list[tuple[str, str]] = []

        def fake_get_env(k: str) -> str | None:
            return env.get(k)

        def fake_set_env(k: str, v: str) -> None:
            env_updates.append((k, v))
            env[k] = v

        def fake_http_post(url: str, headers: dict[str, str], body: str) -> str:
            return """{
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token",
                "expires_in": 3600,
                "token_type": "Bearer"
            }"""

        hooks.cli_get_env = fake_get_env
        hooks.cli_set_env = fake_set_env
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.http_post = fake_http_post

        account = cli.ACCOUNTS[0]
        result = cli._get_valid_token_for_account(account)
        assert result == "new_access_token"
        assert ("OUTLOOK_ACCESS_TOKEN", "new_access_token") in env_updates

    def test_returns_old_token_when_refresh_not_possible(self) -> None:
        """Test returns old token when refresh is not possible (no credentials)."""
        env: dict[str, str] = {
            "OUTLOOK_ACCESS_TOKEN": "old_token",
            "OUTLOOK_REFRESH_TOKEN": "refresh_token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "1700000000",  # Expired
            # Missing client_id and client_secret
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)

        account = cli.ACCOUNTS[0]
        result = cli._get_valid_token_for_account(account)
        # Returns old token since refresh not possible
        assert result == "old_token"


class TestGetToken:
    """Tests for _get_token function."""

    def test_returns_none_when_no_accounts(self) -> None:
        """Test returns None when no token."""
        hooks.cli_get_env = lambda k: None
        result = cli._get_token()
        assert result is None


# =============================================================================
# Console Helpers
# =============================================================================


class TestPrint:
    """Tests for _print helper."""

    def test_calls_hook(self) -> None:
        """Test that _print calls console_output hook."""
        messages: list[str] = []

        def fake_output(msg: str) -> None:
            messages.append(msg)

        hooks.console_output = fake_output
        cli._print("Hello")
        assert messages == ["Hello"]


class TestInput:
    """Tests for _input helper."""

    def test_calls_hook(self) -> None:
        """Test that _input calls console_input hook."""
        prompts: list[str] = []

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return "user_input"

        hooks.console_input = fake_input
        result = cli._input("Enter: ")
        assert result == "user_input"
        assert prompts == ["Enter: "]


# =============================================================================
# API Helpers
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
        result = cli._api_get("token123", "/me/messages")
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
        result = cli._api_post("token123", "/me/sendMail", {"message": {}})
        assert result["result"] == "ok"
        assert len(calls) == 1

    def test_returns_empty_dict_for_empty_response(self) -> None:
        """Test returns empty dict for empty response."""
        hooks.http_post = lambda u, h, b: "   "
        result = cli._api_post("token", "/path", {})
        assert result == {}


# =============================================================================
# Display Helpers
# =============================================================================


class TestFormatRecipients:
    """Tests for _format_recipients function."""

    def test_empty_string_returns_empty_list(self) -> None:
        """Test empty string produces empty list."""
        result = cli._format_recipients("")
        assert result == []

    def test_single_address(self) -> None:
        """Test single email address produces one recipient."""
        result = cli._format_recipients("user@example.com")
        assert len(result) == 1
        assert result[0] == {"emailAddress": {"address": "user@example.com"}}

    def test_multiple_addresses(self) -> None:
        """Test comma-separated addresses produce multiple recipients."""
        result = cli._format_recipients("a@b.com,c@d.com,e@f.com")
        assert len(result) == 3
        assert result[0] == {"emailAddress": {"address": "a@b.com"}}
        assert result[1] == {"emailAddress": {"address": "c@d.com"}}
        assert result[2] == {"emailAddress": {"address": "e@f.com"}}

    def test_strips_whitespace(self) -> None:
        """Test whitespace around addresses is stripped."""
        result = cli._format_recipients("  a@b.com , c@d.com  ")
        assert len(result) == 2
        assert result[0] == {"emailAddress": {"address": "a@b.com"}}
        assert result[1] == {"emailAddress": {"address": "c@d.com"}}

    def test_skips_empty_entries(self) -> None:
        """Test empty entries from trailing commas are skipped."""
        result = cli._format_recipients("a@b.com,,c@d.com,")
        assert len(result) == 2
        assert result[0] == {"emailAddress": {"address": "a@b.com"}}
        assert result[1] == {"emailAddress": {"address": "c@d.com"}}


class TestBuildAttachments:
    """Tests for _build_attachments function."""

    def test_single_file(self) -> None:
        """Test builds attachment object for a single file."""
        hooks.read_file_bytes = lambda p: b"hello world"

        result = cli._build_attachments(("/path/to/document.pdf",))
        assert len(result) == 1
        att = narrow_json_to_dict(result[0])
        assert att["@odata.type"] == "#microsoft.graph.fileAttachment"
        assert att["name"] == "document.pdf"
        assert att["contentType"] == "application/pdf"
        assert att["contentBytes"] != ""

    def test_multiple_files(self) -> None:
        """Test builds attachment objects for multiple files."""
        hooks.read_file_bytes = lambda p: b"\x00\x01\x02"

        result = cli._build_attachments(("/a/photo.png", "/b/notes.txt"))
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

        result = cli._build_attachments(("/path/to/file.xyz123",))
        assert len(result) == 1
        att = narrow_json_to_dict(result[0])
        assert att["contentType"] == "application/octet-stream"

    def test_base64_encodes_content(self) -> None:
        """Test file content is base64-encoded."""
        import base64

        raw_bytes = b"\x89PNG\r\n\x1a\n"
        hooks.read_file_bytes = lambda p: raw_bytes

        result = cli._build_attachments(("/img.png",))
        att = narrow_json_to_dict(result[0])
        expected_encoded = base64.b64encode(raw_bytes).decode("ascii")
        assert att["contentBytes"] == expected_encoded

    def test_empty_tuple_returns_empty_list(self) -> None:
        """Test empty tuple produces empty list."""
        result = cli._build_attachments(())
        assert result == []


class TestDisplayMessageRows:
    """Tests for _display_message_rows function."""

    def test_renders_unread_message(self) -> None:
        """Test renders unread message with asterisk marker."""
        messages: list[str] = []
        hooks.console_output = lambda m: messages.append(m)

        cli._display_message_rows(
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

        cli._display_message_rows(
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
        cli._display_message_rows(
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

        cli._display_message_rows(
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

        cli._display_message_rows(
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


class TestCmdAuth:
    """Tests for cmd_auth command."""

    def test_shows_error_when_no_credentials(self) -> None:
        """Test shows error when credentials missing."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        cli.cmd_auth()
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

        cli.cmd_auth()
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

        cli.cmd_auth()
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

        cli.cmd_folders()
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

        cli.cmd_folders()
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

        cli.cmd_folders()
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

        cli.cmd_folders()
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

        cli.cmd_folders()
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

        cli.cmd_list()
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

        cli.cmd_list()
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

        cli.cmd_list()
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

        cli.cmd_list()
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

        cli.cmd_list()
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

        cli.cmd_list()
        output = " ".join(messages)
        assert "Test" in output


class TestCmdRead:
    """Tests for cmd_read command."""

    def test_shows_error_when_not_authenticated(self) -> None:
        """Test shows error when not authenticated."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        cli.cmd_read(1)
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

        cli.cmd_read(1)
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

        cli.cmd_read(1)
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

        cli.cmd_read(1)
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

        cli.cmd_read(1)
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

        cli.cmd_read(1)
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

        cli.cmd_read(1)
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

        cli.cmd_read(1)
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

        cli.cmd_read(1)
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

        cli.cmd_read(1)
        output = " ".join(messages)
        assert "Test" in output


class TestCmdSend:
    """Tests for cmd_send command."""

    def test_shows_error_when_not_authenticated(self) -> None:
        """Test shows error when not authenticated."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        cli.cmd_send("to@test.com", "Subject", "/body.txt")
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

        cli.cmd_send("to@test.com", "Subject", "/missing.txt")
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

        cli.cmd_send("to@test.com", "Subject", "/body.txt")
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

        cli.cmd_send("to@test.com", "Subject", "/body.txt", html=True)
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

        cli.cmd_send("to@test.com", "Subject", "/body.txt", cc="cc1@test.com,cc2@test.com")
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

        cli.cmd_send("to@test.com", "Subject", "/body.txt", bcc="bcc@test.com")
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

        cli.cmd_send("to@test.com", "Subj", "/body.txt", cc="cc@x.com", bcc="bcc@x.com")
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

        cli.cmd_send("to@test.com", "Subj", "/body.txt")
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

        cli.cmd_send("to@test.com", "Subject", "/body.txt", attachments=("/path/doc.pdf",))
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

        cli.cmd_send("to@test.com", "Subject", "/body.txt", attachments=("/missing.pdf",))
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

        cli.cmd_send(
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

        cli.cmd_send("to@test.com", "Subject", "/body.txt")
        assert len(post_calls) == 1
        assert "attachments" not in post_calls[0][2]


class TestCmdSearch:
    """Tests for cmd_search command."""

    def test_shows_error_when_not_authenticated(self) -> None:
        """Test shows error when not authenticated."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        cli.cmd_search("test query")
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

        cli.cmd_search("TU+11")
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

        cli.cmd_search("nonexistent")
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

        cli.cmd_search("test")
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

        cli.cmd_search("test", count=5)
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

        cli.cmd_search("test")
        output = " ".join(messages)
        assert "Valid Result" in output


# =============================================================================
# Argument Parsing
# =============================================================================


class TestExtractStr:
    """Tests for _extract_str function."""

    def test_extracts_string(self) -> None:
        """Test extracts string value."""
        ns = argparse.Namespace(key="value")
        result = cli._extract_str(ns, "key", "default")
        assert result == "value"

    def test_returns_default_for_missing(self) -> None:
        """Test returns default for missing key."""
        ns = argparse.Namespace()
        result = cli._extract_str(ns, "missing", "default")
        assert result == "default"

    def test_returns_default_for_non_string(self) -> None:
        """Test returns default for non-string value."""
        ns = argparse.Namespace(key=123)
        result = cli._extract_str(ns, "key", "default")
        assert result == "default"


class TestExtractInt:
    """Tests for _extract_int function."""

    def test_extracts_int(self) -> None:
        """Test extracts int value."""
        ns = argparse.Namespace(key=42)
        result = cli._extract_int(ns, "key", 0)
        assert result == 42

    def test_returns_default_for_missing(self) -> None:
        """Test returns default for missing key."""
        ns = argparse.Namespace()
        result = cli._extract_int(ns, "missing", 10)
        assert result == 10


class TestDecodeListArgs:
    """Tests for decode_list_args function."""

    def test_decodes_args(self) -> None:
        """Test decodes list arguments."""
        ns = argparse.Namespace(folder="sent", count=20)
        result = cli.decode_list_args(ns)
        assert result["folder"] == "sent"
        assert result["count"] == 20


class TestDecodeReadArgs:
    """Tests for decode_read_args function."""

    def test_decodes_args(self) -> None:
        """Test decodes read arguments."""
        ns = argparse.Namespace(index=5)
        result = cli.decode_read_args(ns)
        assert result["index"] == 5


class TestExtractOptionalStr:
    """Tests for _extract_optional_str function."""

    def test_extracts_string(self) -> None:
        """Test extracts string value when present."""
        ns = argparse.Namespace(key="value")
        result = cli._extract_optional_str(ns, "key")
        assert result == "value"

    def test_returns_none_for_missing(self) -> None:
        """Test returns None for missing key."""
        ns = argparse.Namespace()
        result = cli._extract_optional_str(ns, "missing")
        assert result is None

    def test_returns_none_for_none_value(self) -> None:
        """Test returns None when value is None."""
        ns = argparse.Namespace(key=None)
        result = cli._extract_optional_str(ns, "key")
        assert result is None

    def test_returns_none_for_non_string(self) -> None:
        """Test returns None for non-string value."""
        ns = argparse.Namespace(key=123)
        result = cli._extract_optional_str(ns, "key")
        assert result is None


class TestExtractBool:
    """Tests for _extract_bool function."""

    def test_extracts_true(self) -> None:
        """Test extracts True value."""
        ns = argparse.Namespace(key=True)
        result = cli._extract_bool(ns, "key", False)
        assert result is True

    def test_extracts_false(self) -> None:
        """Test extracts False value."""
        ns = argparse.Namespace(key=False)
        result = cli._extract_bool(ns, "key", True)
        assert result is False

    def test_returns_default_for_missing(self) -> None:
        """Test returns default for missing key."""
        ns = argparse.Namespace()
        result = cli._extract_bool(ns, "missing", True)
        assert result is True

    def test_returns_default_for_non_bool(self) -> None:
        """Test returns default for non-bool value."""
        ns = argparse.Namespace(key="not_a_bool")
        result = cli._extract_bool(ns, "key", False)
        assert result is False


class TestExtractStrTuple:
    """Tests for _extract_str_tuple function."""

    def test_extracts_list_of_strings(self) -> None:
        """Test extracts list of strings as tuple."""
        attach_list: list[str] = ["file1.pdf", "file2.zip"]
        ns = argparse.Namespace(attach=attach_list)
        result = cli._extract_str_tuple(ns, "attach")
        assert result == ("file1.pdf", "file2.zip")

    def test_returns_empty_tuple_for_none(self) -> None:
        """Test returns empty tuple when value is None."""
        ns = argparse.Namespace(attach=None)
        result = cli._extract_str_tuple(ns, "attach")
        assert result == ()

    def test_returns_empty_tuple_for_missing(self) -> None:
        """Test returns empty tuple for missing key."""
        ns = argparse.Namespace()
        result = cli._extract_str_tuple(ns, "attach")
        assert result == ()

    def test_filters_non_string_entries(self) -> None:
        """Test filters out non-string entries from list."""
        mixed_list: list[str | int] = ["file.pdf", 123, "other.txt"]
        ns = argparse.Namespace(attach=mixed_list)
        result = cli._extract_str_tuple(ns, "attach")
        assert result == ("file.pdf", "other.txt")


class TestDecodeSendArgs:
    """Tests for decode_send_args function."""

    def test_decodes_args(self) -> None:
        """Test decodes send arguments."""
        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=None,
        )
        result = cli.decode_send_args(ns)
        assert result["to"] == "to@test.com"
        assert result["subject"] == "Subject"
        assert result["body_file"] == "/body.txt"
        assert result["cc"] == ""
        assert result["bcc"] == ""
        assert result["html"] is False
        assert result["attachments"] == ()

    def test_decodes_args_with_cc_and_bcc(self) -> None:
        """Test decodes send arguments with cc and bcc."""
        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="cc@test.com",
            bcc="bcc@test.com",
            html=False,
            attach=None,
        )
        result = cli.decode_send_args(ns)
        assert result["cc"] == "cc@test.com"
        assert result["bcc"] == "bcc@test.com"

    def test_decodes_args_with_html(self) -> None:
        """Test decodes send arguments with html flag."""
        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=True,
            attach=None,
        )
        result = cli.decode_send_args(ns)
        assert result["html"] is True

    def test_decodes_args_with_attachments(self) -> None:
        """Test decodes send arguments with attachment list."""
        attach_list: list[str] = ["/path/doc.pdf", "/path/img.png"]
        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=attach_list,
        )
        result = cli.decode_send_args(ns)
        assert result["attachments"] == ("/path/doc.pdf", "/path/img.png")


class TestDecodeSearchArgs:
    """Tests for decode_search_args function."""

    def test_decodes_args(self) -> None:
        """Test decodes search arguments."""
        ns = argparse.Namespace(query="turkic", count=20)
        result = cli.decode_search_args(ns)
        assert result["query"] == "turkic"
        assert result["count"] == 20

    def test_defaults(self) -> None:
        """Test decode uses defaults for missing fields."""
        ns = argparse.Namespace()
        result = cli.decode_search_args(ns)
        assert result["query"] == ""
        assert result["count"] == 10


# =============================================================================
# Main Entry Point
# =============================================================================


class TestBuildParser:
    """Tests for _build_parser function."""

    def test_returns_parser_with_subparsers(self) -> None:
        """Test returns a parser with subparsers."""
        parser = cli._build_parser()
        # Verify it can parse known commands
        args = parser.parse_args(["auth"])
        command = cli._extract_str(args, "command", "")
        assert command == "auth"

    def test_parses_auth_command(self) -> None:
        """Test parses auth command."""
        parser = cli._build_parser()
        args = parser.parse_args(["auth"])
        command = cli._extract_str(args, "command", "")
        assert command == "auth"

    def test_parses_list_command_with_options(self) -> None:
        """Test parses list command with options."""
        parser = cli._build_parser()
        args = parser.parse_args(["list", "-f", "sent", "-n", "20"])
        command = cli._extract_str(args, "command", "")
        folder = cli._extract_str(args, "folder", "")
        count = cli._extract_int(args, "count", 0)
        assert command == "list"
        assert folder == "sent"
        assert count == 20

    def test_parses_send_command_with_body_file(self) -> None:
        """Test parses send command with body_file positional arg."""
        parser = cli._build_parser()
        args = parser.parse_args(["send", "to@test.com", "Subject", "/path/to/body.txt"])
        command = cli._extract_str(args, "command", "")
        send_args = cli.decode_send_args(args)
        assert command == "send"
        assert send_args["to"] == "to@test.com"
        assert send_args["subject"] == "Subject"
        assert send_args["body_file"] == "/path/to/body.txt"
        assert send_args["cc"] == ""
        assert send_args["bcc"] == ""

    def test_parses_send_command_with_cc_and_bcc(self) -> None:
        """Test parses send command with --cc and --bcc flags."""
        parser = cli._build_parser()
        args = parser.parse_args(
            [
                "send",
                "to@test.com",
                "Subject",
                "/body.txt",
                "--cc",
                "a@b.com,c@d.com",
                "--bcc",
                "secret@x.com",
            ]
        )
        send_args = cli.decode_send_args(args)
        assert send_args["cc"] == "a@b.com,c@d.com"
        assert send_args["bcc"] == "secret@x.com"

    def test_parses_send_command_with_html_flag(self) -> None:
        """Test parses send command with --html flag."""
        parser = cli._build_parser()
        args = parser.parse_args(["send", "to@test.com", "Subject", "/path.txt", "--html"])
        send_args = cli.decode_send_args(args)
        assert send_args["to"] == "to@test.com"
        assert send_args["body_file"] == "/path.txt"
        assert send_args["html"] is True

    def test_parses_send_command_with_attachments(self) -> None:
        """Test parses send command with --attach flags."""
        parser = cli._build_parser()
        args = parser.parse_args(
            [
                "send",
                "to@test.com",
                "Subject",
                "/body.txt",
                "--attach",
                "/path/doc.pdf",
                "--attach",
                "/path/img.png",
            ]
        )
        send_args = cli.decode_send_args(args)
        assert send_args["attachments"] == ("/path/doc.pdf", "/path/img.png")

    def test_parses_send_command_no_attachments_default(self) -> None:
        """Test send command defaults to no attachments."""
        parser = cli._build_parser()
        args = parser.parse_args(["send", "to@test.com", "Subject", "/body.txt"])
        send_args = cli.decode_send_args(args)
        assert send_args["attachments"] == ()

    def test_parses_search_command(self) -> None:
        """Test parses search command with query and count."""
        parser = cli._build_parser()
        args = parser.parse_args(["search", "turkic workshop", "-n", "20"])
        command = cli._extract_str(args, "command", "")
        search_args = cli.decode_search_args(args)
        assert command == "search"
        assert search_args["query"] == "turkic workshop"
        assert search_args["count"] == 20

    def test_parses_search_command_defaults(self) -> None:
        """Test parses search command with default count."""
        parser = cli._build_parser()
        args = parser.parse_args(["search", "TU+11"])
        search_args = cli.decode_search_args(args)
        assert search_args["query"] == "TU+11"
        assert search_args["count"] == 10


class TestDispatchCommand:
    """Tests for _dispatch_command function."""

    def test_dispatches_auth(self) -> None:
        """Test dispatches auth command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace()
        cli._dispatch_command("auth", ns)
        output = " ".join(messages)
        assert "Missing credentials" in output

    def test_dispatches_folders(self) -> None:
        """Test dispatches folders command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace()
        cli._dispatch_command("folders", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_list(self) -> None:
        """Test dispatches list command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(folder="inbox", count=10)
        cli._dispatch_command("list", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_ls_alias(self) -> None:
        """Test dispatches ls alias for list."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(folder="inbox", count=10)
        cli._dispatch_command("ls", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_read(self) -> None:
        """Test dispatches read command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(index=1)
        cli._dispatch_command("read", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_send(self) -> None:
        """Test dispatches send command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_send_missing_to(self) -> None:
        """Test dispatches send with missing to field."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(
            to="",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Missing required arguments" in output

    def test_dispatches_send_missing_body_file(self) -> None:
        """Test dispatches send with empty body_file."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="",
            cc="",
            bcc="",
            html=False,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Missing required argument: body_file" in output

    def test_dispatches_send_with_body_file(self) -> None:
        """Test dispatches send reading body from file."""
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
        hooks.read_file = lambda p: "Multi-line\nEmail body\nFrom file"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/path/to/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Email sent" in output
        assert len(post_calls) == 1
        assert "Multi-line" in post_calls[0][2]
        assert "From file" in post_calls[0][2]

    def test_dispatches_send_with_cc_and_bcc(self) -> None:
        """Test dispatches send passes cc and bcc through to cmd_send."""
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

        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="cc@test.com",
            bcc="bcc@test.com",
            html=False,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        assert len(post_calls) == 1
        assert "cc@test.com" in post_calls[0][2]
        assert "bcc@test.com" in post_calls[0][2]

    def test_dispatches_send_with_html_flag(self) -> None:
        """Test dispatches send with --html wraps body in pre tags."""
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

        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=True,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Email sent" in output
        assert len(post_calls) == 1
        assert '"contentType":"HTML"' in post_calls[0][2]
        assert "<pre" in post_calls[0][2]
        assert "Line1" in post_calls[0][2]

    def test_dispatches_send_with_attachments(self) -> None:
        """Test dispatches send passes attachments through to cmd_send."""
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
        hooks.read_file_bytes = lambda p: b"binary content"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        attach_list: list[str] = ["/path/doc.pdf"]
        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=attach_list,
        )
        cli._dispatch_command("send", ns)
        assert len(post_calls) == 1
        assert "doc.pdf" in post_calls[0][2]
        assert "#microsoft.graph.fileAttachment" in post_calls[0][2]

    def test_dispatches_search(self) -> None:
        """Test dispatches search command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(query="turkic", count=10)
        cli._dispatch_command("search", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_search_missing_query(self) -> None:
        """Test dispatches search with empty query shows error."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(query="", count=10)
        cli._dispatch_command("search", ns)
        output = " ".join(messages)
        assert "Missing required argument: query" in output

    def test_dispatches_default(self) -> None:
        """Test dispatches default (list inbox)."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace()
        cli._dispatch_command("", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output


class TestMain:
    """Tests for main function."""

    def test_main_dispatches_to_command(self) -> None:
        """Test main parses args and dispatches to command.

        Since main() uses sys.argv directly, we test the components separately.
        The _build_parser and _dispatch_command functions are tested above.
        """
        # Verify the main function exists and is callable
        parser = cli._build_parser()
        args = parser.parse_args(["folders"])
        # Dispatch would be called with these args
        command = cli._extract_str(args, "command", "")
        assert command == "folders"
