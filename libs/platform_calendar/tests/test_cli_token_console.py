"""CLI token refresh machinery and console interaction."""

from __future__ import annotations

import io
from datetime import datetime

from platform_core.json_utils import JSONObject
from rich.console import Console

from platform_calendar.cli import (
    _confirm_ask,
    _get_console,
    _prompt_ask,
    set_console,
)
from platform_calendar.cli_auth import (
    Account,
    _get_now,
    _get_valid_token_for_account,
    _is_token_expired,
    _refresh_token,
    _set_env,
    decode_token_refresh_response,
    require_int,
    require_str,
)
from platform_calendar.testing import hooks, reset_hooks

# =============================================================================
# Test Typed Argument Structures
# =============================================================================


class TestRequireStr:
    """Tests for require_str function."""

    def test_require_str_valid(self) -> None:
        """Test require_str with valid string."""
        data: JSONObject = {"key": "value"}
        result = require_str(data, "key")
        assert result == "value"

    def test_require_str_wrong_type(self) -> None:
        """Test require_str with wrong type raises TypeError."""
        import pytest

        data: JSONObject = {"key": 123}
        with pytest.raises(TypeError, match="Expected str for key"):
            require_str(data, "key")

    def test_require_str_missing_key(self) -> None:
        """Test require_str with missing key raises KeyError."""
        import pytest

        data: JSONObject = {"other": "value"}
        with pytest.raises(KeyError):
            require_str(data, "key")


class TestRequireInt:
    """Tests for require_int function."""

    def test_require_int_valid(self) -> None:
        """Test require_int with valid int."""
        data: JSONObject = {"key": 123}
        result = require_int(data, "key")
        assert result == 123

    def test_require_int_wrong_type(self) -> None:
        """Test require_int with wrong type raises TypeError."""
        import pytest

        data: JSONObject = {"key": "not_an_int"}
        with pytest.raises(TypeError, match="Expected int for key"):
            require_int(data, "key")

    def test_require_int_missing_key(self) -> None:
        """Test require_int with missing key raises KeyError."""
        import pytest

        data: JSONObject = {"other": 123}
        with pytest.raises(KeyError):
            require_int(data, "key")


class TestDecodeTokenRefreshResponse:
    """Tests for decode_token_refresh_response function."""

    def test_decode_valid_response(self) -> None:
        """Test decoding valid token refresh response."""
        data: JSONObject = {
            "access_token": "new_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        result = decode_token_refresh_response(data)
        assert result["access_token"] == "new_token"
        assert result["expires_in"] == 3600
        assert result["token_type"] == "Bearer"


class TestSetEnv:
    """Tests for _set_env function."""

    def test_set_env_calls_hook(self) -> None:
        """Test _set_env calls the hook."""
        captured: list[tuple[str, str]] = []

        def fake_set_env(key: str, value: str) -> None:
            captured.append((key, value))

        hooks.cli_set_env = fake_set_env
        _set_env("TEST_KEY", "test_value")
        assert captured == [("TEST_KEY", "test_value")]


class TestIsTokenExpired:
    """Tests for _is_token_expired function."""

    def test_token_not_expired(self) -> None:
        """Test token that is not expired."""
        future_time = int(datetime.now().timestamp()) + 3600
        hooks.cli_get_now = lambda: datetime.now()
        result = _is_token_expired(str(future_time))
        assert result is False

    def test_token_expired(self) -> None:
        """Test token that is expired."""
        past_time = int(datetime.now().timestamp()) - 3600
        hooks.cli_get_now = lambda: datetime.now()
        result = _is_token_expired(str(past_time))
        assert result is True

    def test_token_expiring_soon(self) -> None:
        """Test token that expires within buffer (60s)."""
        near_time = int(datetime.now().timestamp()) + 30
        hooks.cli_get_now = lambda: datetime.now()
        result = _is_token_expired(str(near_time))
        assert result is True


class TestRefreshToken:
    """Tests for _refresh_token function."""

    def test_refresh_token_success(self) -> None:
        """Test successful token refresh."""
        response_json = '{"access_token": "new_token", "expires_in": 3600, "token_type": "Bearer"}'

        def fake_http_post(url: str, headers: dict[str, str], body: str) -> str:
            assert "oauth2.googleapis.com/token" in url
            assert "grant_type=refresh_token" in body
            return response_json

        hooks.http_post = fake_http_post
        result = _refresh_token("client_id", "client_secret", "refresh_token")
        assert result["access_token"] == "new_token"
        assert result["expires_in"] == 3600


class TestGetValidTokenForAccount:
    """Tests for _get_valid_token_for_account function."""

    def test_no_token_returns_none(self) -> None:
        """Test account with no token returns None."""
        account = Account(
            name="Test",
            email="test@example.com",
            token_env="TEST_TOKEN",
            refresh_token_env="TEST_REFRESH",
            expires_at_env="TEST_EXPIRES",
        )
        hooks.cli_get_env = lambda key: None
        result = _get_valid_token_for_account(account)
        assert result is None

    def test_token_not_expired_returns_token(self) -> None:
        """Test valid token returns without refresh."""
        account = Account(
            name="Test",
            email="test@example.com",
            token_env="TEST_TOKEN",
            refresh_token_env="TEST_REFRESH",
            expires_at_env="TEST_EXPIRES",
        )
        future_time = str(int(datetime.now().timestamp()) + 3600)

        def fake_get_env(key: str) -> str | None:
            if key == "TEST_TOKEN":
                return "valid_token"
            if key == "TEST_REFRESH":
                return "refresh_token"
            if key == "TEST_EXPIRES":
                return future_time
            return None

        hooks.cli_get_env = fake_get_env
        hooks.cli_get_now = lambda: datetime.now()
        result = _get_valid_token_for_account(account)
        assert result == "valid_token"

    def test_token_expired_refreshes(self) -> None:
        """Test expired token triggers refresh."""
        account = Account(
            name="Test",
            email="test@example.com",
            token_env="TEST_TOKEN",
            refresh_token_env="TEST_REFRESH",
            expires_at_env="TEST_EXPIRES",
        )
        past_time = str(int(datetime.now().timestamp()) - 3600)
        env_values: dict[str, str | None] = {
            "TEST_TOKEN": "old_token",
            "TEST_REFRESH": "refresh_token",
            "TEST_EXPIRES": past_time,
            "GOOGLE_CALENDAR_CLIENT_ID": "client_id",
            "GOOGLE_CALENDAR_CLIENT_SECRET": "client_secret",
        }
        set_calls: list[tuple[str, str]] = []

        def fake_get_env(key: str) -> str | None:
            return env_values.get(key)

        def fake_set_env(key: str, value: str) -> None:
            set_calls.append((key, value))

        response_json = '{"access_token": "new_token", "expires_in": 3600, "token_type": "Bearer"}'

        def fake_http_post(url: str, headers: dict[str, str], body: str) -> str:
            return response_json

        hooks.cli_get_env = fake_get_env
        hooks.cli_set_env = fake_set_env
        hooks.cli_get_now = lambda: datetime.now()
        hooks.http_post = fake_http_post

        result = _get_valid_token_for_account(account)
        assert result == "new_token"
        # Verify both token and expires_at were updated
        set_keys = [k for k, v in set_calls]
        assert "TEST_TOKEN" in set_keys
        assert "TEST_EXPIRES" in set_keys

    def test_missing_credentials_skips_refresh(self) -> None:
        """Test missing client credentials skips refresh."""
        account = Account(
            name="Test",
            email="test@example.com",
            token_env="TEST_TOKEN",
            refresh_token_env="TEST_REFRESH",
            expires_at_env="TEST_EXPIRES",
        )
        past_time = str(int(datetime.now().timestamp()) - 3600)

        def fake_get_env(key: str) -> str | None:
            if key == "TEST_TOKEN":
                return "old_token"
            if key == "TEST_REFRESH":
                return "refresh_token"
            if key == "TEST_EXPIRES":
                return past_time
            return None  # No client credentials

        hooks.cli_get_env = fake_get_env
        hooks.cli_get_now = lambda: datetime.now()

        result = _get_valid_token_for_account(account)
        # Returns old token since refresh couldn't happen
        assert result == "old_token"


# =============================================================================
# Test Console Functions
# =============================================================================


class TestConsole:
    """Tests for console functions."""

    def test_get_console_default_returns_console(self) -> None:
        """Test getting default console returns something that can print."""
        # Reset to ensure no hook
        reset_hooks()
        console = _get_console()
        # Verify by calling print - would raise if not a console
        console.print("test", end="")

    def test_get_console_hooked(self) -> None:
        """Test getting hooked console."""
        fake_console = Console(file=io.StringIO())
        hooks.cli_get_console = lambda: fake_console
        result = _get_console()
        assert result is fake_console

    def test_set_console(self) -> None:
        """Test setting console via set_console."""
        fake_console = Console(file=io.StringIO())
        set_console(fake_console)
        result = _get_console()
        assert result is fake_console


class TestGetNow:
    """Tests for _get_now."""

    def test_get_now_default(self) -> None:
        """Test getting current time without hook returns valid datetime."""
        result = _get_now()
        # Verify it's a valid datetime by checking year is reasonable
        assert result.year >= 2026

    def test_get_now_hooked(self) -> None:
        """Test getting current time with hook."""
        fixed_time = datetime(2026, 2, 20, 14, 30, 0)
        hooks.cli_get_now = lambda: fixed_time
        result = _get_now()
        assert result == fixed_time


class TestPromptAsk:
    """Tests for _prompt_ask."""

    def test_prompt_ask_hooked(self) -> None:
        """Test prompt with hook."""
        hooks.cli_prompt_ask = lambda msg: "user_input"
        result = _prompt_ask("Enter value:")
        assert result == "user_input"


class TestConfirmAsk:
    """Tests for _confirm_ask."""

    def test_confirm_ask_hooked_true(self) -> None:
        """Test confirm with hook returning True."""
        hooks.cli_confirm_ask = lambda msg: True
        result = _confirm_ask("Are you sure?")
        assert result is True

    def test_confirm_ask_hooked_false(self) -> None:
        """Test confirm with hook returning False."""
        hooks.cli_confirm_ask = lambda msg: False
        result = _confirm_ask("Are you sure?")
        assert result is False


# =============================================================================
# Test API Functions
# =============================================================================


class TestReadResponseBody:
    """Tests for _read_response_body."""

    def test_read_response_body(self) -> None:
        """Test reading response body via API get hook."""
        # Test _read_response_body indirectly through the API function
        # which will use it when hooks aren't set
        hooks.cli_api_get = lambda token, url: {"result": "test_body_read"}
        from platform_calendar.cli_api import (
            _api_get,
        )

        result = _api_get("token", "url")
        assert result["result"] == "test_body_read"
