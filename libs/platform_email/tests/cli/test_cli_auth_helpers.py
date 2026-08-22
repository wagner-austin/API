"""Email CLI: token machinery, env, console, API helpers."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime

import pytest
from platform_core.json_utils import JSONObject

from platform_email import cli_auth
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
        result = cli_auth.require_str(data, "key")
        assert result == "value"

    def test_raises_type_error_for_int(self) -> None:
        """Test that require_str raises TypeError for int."""
        data: JSONObject = {"key": 123}
        with pytest.raises(TypeError, match="Expected str"):
            cli_auth.require_str(data, "key")

    def test_raises_key_error_for_missing(self) -> None:
        """Test that require_str raises KeyError for missing key."""
        data: JSONObject = {}
        with pytest.raises(KeyError):
            cli_auth.require_str(data, "missing")


class TestRequireInt:
    """Tests for require_int function."""

    def test_returns_int_value(self) -> None:
        """Test that require_int returns int value."""
        data: JSONObject = {"key": 123}
        result = cli_auth.require_int(data, "key")
        assert result == 123

    def test_raises_type_error_for_str(self) -> None:
        """Test that require_int raises TypeError for string."""
        data: JSONObject = {"key": "value"}
        with pytest.raises(TypeError, match="Expected int"):
            cli_auth.require_int(data, "key")


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
        result = cli_auth.decode_token_response(data)
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
        result = cli_auth._generate_code_verifier()
        # PKCE code verifiers should be 43-128 characters
        assert 43 <= len(result) <= 128
        assert result == result.strip()  # No leading/trailing whitespace

    def test_is_url_safe(self) -> None:
        """Test that code verifier contains only URL-safe characters."""
        result = cli_auth._generate_code_verifier()
        # URL-safe base64 uses alphanumeric, -, and _
        for char in result:
            assert char.isalnum() or char in "-_"


class TestGenerateCodeChallenge:
    """Tests for _generate_code_challenge function."""

    def test_produces_base64_result(self) -> None:
        """Test that code challenge produces base64url-encoded result."""
        verifier = "test_verifier_1234567890"
        result = cli_auth._generate_code_challenge(verifier)
        # Base64url uses alphanumeric, -, and _
        for char in result:
            assert char.isalnum() or char in "-_"

    def test_different_verifiers_produce_different_challenges(self) -> None:
        """Test that different verifiers produce different challenges."""
        verifier1 = "verifier_one"
        verifier2 = "verifier_two"
        challenge1 = cli_auth._generate_code_challenge(verifier1)
        challenge2 = cli_auth._generate_code_challenge(verifier2)
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
        result = cli_auth._get_env("MY_KEY")
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
        cli_auth._set_env("KEY", "VALUE")
        assert called_with == [("KEY", "VALUE")]


class TestGetNow:
    """Tests for _get_now helper."""

    def test_calls_hook(self) -> None:
        """Test that _get_now calls the hook."""
        fixed_time = datetime(2025, 1, 15, 12, 0, 0)

        def fake_get_now() -> datetime:
            return fixed_time

        hooks.cli_get_now = fake_get_now
        result = cli_auth._get_now()
        assert result == fixed_time


class TestIsTokenExpired:
    """Tests for _is_token_expired function."""

    def test_returns_true_for_expired_token(self) -> None:
        """Test returns True for expired token."""
        # Set time to a known value (2025-01-01)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        # Token expired at timestamp 1700000000 (2023)
        result = cli_auth._is_token_expired("1700000000")
        assert result is True

    def test_returns_false_for_valid_token(self) -> None:
        """Test returns False for valid token."""
        # 2025-01-01
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        # Token expires at 2035-01-01 (10 years in future)
        result = cli_auth._is_token_expired("2051222400")
        assert result is False

    def test_returns_true_for_token_expiring_soon(self) -> None:
        """Test returns True for token expiring within 60 seconds."""
        # 2025-01-01 00:00:00
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        # Token expires at 2025-01-01 00:00:30 (30 seconds in future)
        result = cli_auth._is_token_expired("1735689630")
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
        result = cli_auth._refresh_token("client_id", "client_secret", "refresh_token")
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
        result = cli_auth._exchange_code_for_tokens(
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
        account = cli_auth.ACCOUNTS[0]
        result = cli_auth._get_valid_token_for_account(account)
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
        account = cli_auth.ACCOUNTS[0]
        result = cli_auth._get_valid_token_for_account(account)
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

        account = cli_auth.ACCOUNTS[0]
        result = cli_auth._get_valid_token_for_account(account)
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

        account = cli_auth.ACCOUNTS[0]
        result = cli_auth._get_valid_token_for_account(account)
        # Returns old token since refresh not possible
        assert result == "old_token"


class TestGetToken:
    """Tests for _get_token function."""

    def test_returns_none_when_no_accounts(self) -> None:
        """Test returns None when no token."""
        hooks.cli_get_env = lambda k: None
        result = cli_auth._get_token()
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
        cli_auth._print("Hello")
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
        result = cli_auth._input("Enter: ")
        assert result == "user_input"
        assert prompts == ["Enter: "]


# =============================================================================
# API Helpers
# =============================================================================
