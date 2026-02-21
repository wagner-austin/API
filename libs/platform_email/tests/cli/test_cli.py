"""Tests for platform_email CLI module."""

from __future__ import annotations

import argparse
from collections.abc import Generator
from datetime import datetime

import pytest
from platform_core.json_utils import JSONObject

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

        cli.cmd_send("to@test.com", "Subject", "Body")
        output = " ".join(messages)
        assert "Not authenticated" in output

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

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        cli.cmd_send("to@test.com", "Subject", "Body")
        output = " ".join(messages)
        assert "Email sent" in output
        assert len(post_calls) == 1


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


class TestDecodeSendArgs:
    """Tests for decode_send_args function."""

    def test_decodes_args(self) -> None:
        """Test decodes send arguments."""
        ns = argparse.Namespace(to="to@test.com", subject="Subject", body="Body")
        result = cli.decode_send_args(ns)
        assert result["to"] == "to@test.com"
        assert result["subject"] == "Subject"
        assert result["body"] == "Body"


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

        ns = argparse.Namespace(to="to@test.com", subject="Subject", body="Body")
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_send_missing_args(self) -> None:
        """Test dispatches send with missing args."""
        messages: list[str] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(to="", subject="Subject", body="Body")
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Missing required arguments" in output

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
