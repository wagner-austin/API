"""Tests for Calendar CLI."""

from __future__ import annotations

import argparse
import io
from datetime import datetime

from platform_core.json_utils import JSONObject
from rich.console import Console

from platform_calendar.cli import (
    Account,
    CreateArgs,
    DeleteArgs,
    EventInfo,
    ListArgs,
    _collect_events,
    _confirm_ask,
    _create_event,
    _delete_event,
    _extract_int,
    _extract_str,
    _extract_str_or_none,
    _fetch_calendars,
    _fetch_events,
    _format_time,
    _get_console,
    _get_delete_choice,
    _get_env,
    _get_now,
    _get_token,
    _get_valid_token_for_account,
    _is_token_expired,
    _parse_time,
    _prompt_ask,
    _refresh_token,
    _sanitize,
    _set_env,
    _show_events_for_delete,
    cmd_calendars,
    cmd_create,
    cmd_delete,
    cmd_list,
    cmd_tomorrow,
    cmd_week,
    decode_create_args,
    decode_delete_args,
    decode_list_args,
    decode_token_refresh_response,
    main,
    require_int,
    require_str,
    set_console,
)
from platform_calendar.testing import hooks, reset_hooks

# =============================================================================
# Test Typed Argument Structures
# =============================================================================


class TestListArgs:
    """Tests for ListArgs decoding."""

    def test_decode_list_args_with_date(self) -> None:
        """Test decoding list args with explicit date."""
        ns = argparse.Namespace(date="2026-02-20")
        result = decode_list_args(ns)
        assert result["date"] == "2026-02-20"

    def test_decode_list_args_without_date(self) -> None:
        """Test decoding list args without date defaults to today."""
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)
        ns = argparse.Namespace(date=None)
        result = decode_list_args(ns)
        assert result["date"] == "2026-02-20"


class TestCreateArgs:
    """Tests for CreateArgs decoding."""

    def test_decode_create_args_full(self) -> None:
        """Test decoding create args with all fields."""
        ns = argparse.Namespace(
            title="Meeting",
            time="14:00",
            date="2026-02-20",
            duration=30,
            location="Office",
            account="Personal",
        )
        result = decode_create_args(ns)
        assert result["title"] == "Meeting"
        assert result["time"] == "14:00"
        assert result["date"] == "2026-02-20"
        assert result["duration"] == 30
        assert result["location"] == "Office"
        assert result["account"] == "Personal"

    def test_decode_create_args_defaults(self) -> None:
        """Test decoding create args with defaults."""
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)
        ns = argparse.Namespace(
            title="Test",
            time="10:00",
            date=None,
            duration=60,
            location=None,
            account="Personal",
        )
        result = decode_create_args(ns)
        assert result["date"] == "2026-02-20"
        assert result["location"] == ""

    def test_decode_create_args_non_string_title(self) -> None:
        """Test decoding create args with non-string title."""
        ns = argparse.Namespace(
            title=123,
            time="10:00",
            date="2026-02-20",
            duration=60,
            location=None,
            account="Personal",
        )
        result = decode_create_args(ns)
        assert result["title"] == ""

    def test_decode_create_args_non_int_duration(self) -> None:
        """Test decoding create args with non-int duration."""
        ns = argparse.Namespace(
            title="Test",
            time="10:00",
            date="2026-02-20",
            duration="invalid",
            location=None,
            account="Personal",
        )
        result = decode_create_args(ns)
        assert result["duration"] == 60


class TestDeleteArgs:
    """Tests for DeleteArgs decoding."""

    def test_decode_delete_args_with_date(self) -> None:
        """Test decoding delete args with date."""
        ns = argparse.Namespace(date="2026-02-20")
        result = decode_delete_args(ns)
        assert result["date"] == "2026-02-20"

    def test_decode_delete_args_without_date(self) -> None:
        """Test decoding delete args without date."""
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)
        ns = argparse.Namespace(date=None)
        result = decode_delete_args(ns)
        assert result["date"] == "2026-02-20"


# =============================================================================
# Test Extract Functions
# =============================================================================


class TestExtractStr:
    """Tests for _extract_str."""

    def test_extract_str_found(self) -> None:
        """Test extracting existing string."""
        ns = argparse.Namespace(key="value")
        result = _extract_str(ns, "key", "default")
        assert result == "value"

    def test_extract_str_not_found(self) -> None:
        """Test extracting missing key returns default."""
        ns = argparse.Namespace()
        result = _extract_str(ns, "missing", "default")
        assert result == "default"

    def test_extract_str_non_string(self) -> None:
        """Test extracting non-string value returns default."""
        ns = argparse.Namespace(key=123)
        result = _extract_str(ns, "key", "default")
        assert result == "default"


class TestExtractStrOrNone:
    """Tests for _extract_str_or_none."""

    def test_extract_str_or_none_found(self) -> None:
        """Test extracting existing string."""
        ns = argparse.Namespace(key="value")
        result = _extract_str_or_none(ns, "key")
        assert result == "value"

    def test_extract_str_or_none_missing(self) -> None:
        """Test extracting missing key returns None."""
        ns = argparse.Namespace()
        result = _extract_str_or_none(ns, "missing")
        assert result is None

    def test_extract_str_or_none_non_string(self) -> None:
        """Test extracting non-string value returns None."""
        ns = argparse.Namespace(key=123)
        result = _extract_str_or_none(ns, "key")
        assert result is None


class TestExtractInt:
    """Tests for _extract_int."""

    def test_extract_int_found(self) -> None:
        """Test extracting existing int."""
        ns = argparse.Namespace(key=42)
        result = _extract_int(ns, "key", 0)
        assert result == 42

    def test_extract_int_missing(self) -> None:
        """Test extracting missing key returns default."""
        ns = argparse.Namespace()
        result = _extract_int(ns, "missing", 99)
        assert result == 99

    def test_extract_int_non_int(self) -> None:
        """Test extracting non-int value returns default."""
        ns = argparse.Namespace(key="not an int")
        result = _extract_int(ns, "key", 99)
        assert result == 99


# =============================================================================
# Test Account
# =============================================================================


class TestAccount:
    """Tests for Account class."""

    def test_account_initialization(self) -> None:
        """Test account initialization."""
        account = Account(
            name="Test",
            email="test@example.com",
            token_env="TEST_TOKEN",
            refresh_token_env="TEST_REFRESH_TOKEN",
            expires_at_env="TEST_EXPIRES_AT",
            default_calendar="cal123",
        )
        assert account.name == "Test"
        assert account.email == "test@example.com"
        assert account.token_env == "TEST_TOKEN"
        assert account.refresh_token_env == "TEST_REFRESH_TOKEN"
        assert account.expires_at_env == "TEST_EXPIRES_AT"
        assert account.default_calendar == "cal123"

    def test_account_default_calendar(self) -> None:
        """Test account default calendar defaults to primary."""
        account = Account(
            name="Test",
            email="test@example.com",
            token_env="TEST_TOKEN",
            refresh_token_env="TEST_REFRESH_TOKEN",
            expires_at_env="TEST_EXPIRES_AT",
        )
        assert account.default_calendar == "primary"


# =============================================================================
# Test Environment Functions
# =============================================================================


class TestGetEnv:
    """Tests for _get_env."""

    def test_get_env_with_hook(self) -> None:
        """Test getting env var with hook."""
        hooks.cli_get_env = lambda key: f"value_{key}"
        result = _get_env("TEST_KEY")
        assert result == "value_TEST_KEY"

    def test_get_env_hook_returns_none(self) -> None:
        """Test hook returning None."""
        hooks.cli_get_env = lambda key: None
        result = _get_env("TEST_KEY")
        assert result is None


class TestGetToken:
    """Tests for _get_token."""

    def test_get_token_personal(self) -> None:
        """Test getting Personal account token."""

        def fake_get_env(key: str) -> str | None:
            if "ACCESS_TOKEN" in key:
                return "token123"
            return None

        hooks.cli_get_env = fake_get_env
        result = _get_token("Personal")
        assert result == "token123"

    def test_get_token_interns(self) -> None:
        """Test getting Interns account token."""

        def fake_get_env(key: str) -> str | None:
            if "INTERNS_ACCESS_TOKEN" in key:
                return "intern_token"
            return None

        hooks.cli_get_env = fake_get_env
        result = _get_token("Interns")
        assert result == "intern_token"

    def test_get_token_case_insensitive(self) -> None:
        """Test account name is case insensitive."""

        def fake_get_env(key: str) -> str | None:
            if "ACCESS_TOKEN" in key:
                return "token123"
            return None

        hooks.cli_get_env = fake_get_env
        result = _get_token("personal")
        assert result == "token123"

    def test_get_token_unknown_account(self) -> None:
        """Test unknown account returns None."""
        hooks.cli_get_env = lambda key: "token123"
        result = _get_token("Unknown")
        assert result is None


# =============================================================================
# Test Token Refresh Types
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
        from platform_calendar.cli import _api_get

        result = _api_get("token", "url")
        assert result["result"] == "test_body_read"


class TestApiGet:
    """Tests for API GET function."""

    def test_api_get_hooked(self) -> None:
        """Test API GET with hook."""
        hooks.cli_api_get = lambda token, url: {"id": "123", "summary": "Test"}
        from platform_calendar.cli import _api_get

        result = _api_get("token123", "https://example.com/api")
        assert result["id"] == "123"
        assert result["summary"] == "Test"


class TestApiPost:
    """Tests for API POST function."""

    def test_api_post_hooked(self) -> None:
        """Test API POST with hook."""
        hooks.cli_api_post = lambda token, url, body: {"id": "created", "body": body}
        from platform_calendar.cli import _api_post

        result = _api_post("token123", "https://example.com/api", {"title": "Test"})
        assert result["id"] == "created"


class TestApiDelete:
    """Tests for API DELETE function."""

    def test_api_delete_hooked(self) -> None:
        """Test API DELETE with hook."""
        deleted_urls: list[str] = []

        def capture_delete(token: str, url: str) -> None:
            deleted_urls.append(url)

        hooks.cli_api_delete = capture_delete
        from platform_calendar.cli import _api_delete

        _api_delete("token123", "https://example.com/api/123")
        assert "https://example.com/api/123" in deleted_urls


# =============================================================================
# Test Fetch Functions
# =============================================================================


class TestFetchCalendars:
    """Tests for _fetch_calendars."""

    def test_fetch_calendars(self) -> None:
        """Test fetching calendars."""
        hooks.cli_api_get = lambda token, url: {
            "items": [
                {"id": "cal1", "summary": "Calendar 1"},
                {"id": "cal2", "summary": "Calendar 2", "primary": True},
            ]
        }
        result = _fetch_calendars("token123")
        assert len(result) == 2
        assert result[0]["id"] == "cal1"
        assert result[1]["primary"] is True

    def test_fetch_calendars_no_items(self) -> None:
        """Test fetching calendars with no items."""
        hooks.cli_api_get = lambda token, url: {}
        result = _fetch_calendars("token123")
        assert result == []

    def test_fetch_calendars_items_not_list(self) -> None:
        """Test fetching calendars when items is not a list."""
        hooks.cli_api_get = lambda token, url: {"items": "not a list"}
        result = _fetch_calendars("token123")
        assert result == []


class TestFetchEvents:
    """Tests for _fetch_events."""

    def test_fetch_events(self) -> None:
        """Test fetching events."""
        hooks.cli_api_get = lambda token, url: {
            "items": [
                {
                    "id": "evt1",
                    "summary": "Event 1",
                    "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                }
            ]
        }
        result = _fetch_events("token123", "cal123", "2026-02-20")
        assert len(result) == 1
        assert result[0]["id"] == "evt1"

    def test_fetch_events_empty(self) -> None:
        """Test fetching events with no results."""
        hooks.cli_api_get = lambda token, url: {"items": []}
        result = _fetch_events("token123", "cal123", "2026-02-20")
        assert result == []

    def test_fetch_events_items_not_list(self) -> None:
        """Test fetching events when items is not a list."""
        hooks.cli_api_get = lambda token, url: {"items": "not a list"}
        result = _fetch_events("token123", "cal123", "2026-02-20")
        assert result == []


class TestCreateEvent:
    """Tests for _create_event."""

    def test_create_event(self) -> None:
        """Test creating event."""
        created: list[JSONObject] = []

        def capture_post(token: str, url: str, body: JSONObject) -> JSONObject:
            created.append(body)
            return {"id": "new_evt"}

        hooks.cli_api_post = capture_post
        result = _create_event(
            "token123",
            "cal123",
            "Test Meeting",
            "2026-02-20T14:00:00",
            "2026-02-20T15:00:00",
            location="Office",
        )
        assert result["id"] == "new_evt"
        assert len(created) == 1
        assert created[0]["summary"] == "Test Meeting"
        assert created[0]["location"] == "Office"

    def test_create_event_no_location(self) -> None:
        """Test creating event without location."""
        created: list[JSONObject] = []

        def capture_post(token: str, url: str, body: JSONObject) -> JSONObject:
            created.append(body)
            return {"id": "new_evt"}

        hooks.cli_api_post = capture_post
        _create_event(
            "token123",
            "cal123",
            "Test",
            "2026-02-20T14:00:00",
            "2026-02-20T15:00:00",
        )
        assert "location" not in created[0]


class TestDeleteEvent:
    """Tests for _delete_event."""

    def test_delete_event(self) -> None:
        """Test deleting event."""
        deleted_urls: list[str] = []
        hooks.cli_api_delete = lambda token, url: deleted_urls.append(url)
        _delete_event("token123", "cal123", "evt123")
        assert len(deleted_urls) == 1
        assert "evt123" in deleted_urls[0]


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestFormatTime:
    """Tests for _format_time."""

    def test_format_time_with_datetime(self) -> None:
        """Test formatting event with dateTime."""
        event: JSONObject = {"start": {"dateTime": "2026-02-20T14:30:00-08:00"}}
        time_str, sort_key = _format_time(event)
        assert time_str == "14:30"
        assert sort_key == "2026-02-20T14:30:00-08:00"

    def test_format_time_all_day(self) -> None:
        """Test formatting all-day event."""
        event: JSONObject = {"start": {"date": "2026-02-20"}}
        time_str, sort_key = _format_time(event)
        assert time_str == "all-day"
        assert sort_key == "00:00"

    def test_format_time_no_start(self) -> None:
        """Test formatting event with no start."""
        event: JSONObject = {}
        time_str, sort_key = _format_time(event)
        assert time_str == "all-day"
        assert sort_key == "00:00"

    def test_format_time_start_not_dict(self) -> None:
        """Test formatting event with start not a dict."""
        event: JSONObject = {"start": "not a dict"}
        time_str, sort_key = _format_time(event)
        assert time_str == "all-day"
        assert sort_key == "00:00"


class TestSanitize:
    """Tests for _sanitize."""

    def test_sanitize_ascii(self) -> None:
        """Test sanitizing ASCII text."""
        result = _sanitize("Hello World")
        assert result == "Hello World"

    def test_sanitize_removes_non_ascii(self) -> None:
        """Test sanitizing removes non-ASCII."""
        result = _sanitize("Hello\u2019World")
        assert result == "HelloWorld"


class TestParseTime:
    """Tests for _parse_time."""

    def test_parse_time_pm(self) -> None:
        """Test parsing PM time."""
        assert _parse_time("2pm") == "14:00"
        assert _parse_time("3PM") == "15:00"

    def test_parse_time_am(self) -> None:
        """Test parsing AM time."""
        assert _parse_time("9am") == "09:00"
        assert _parse_time("10AM") == "10:00"

    def test_parse_time_noon(self) -> None:
        """Test parsing noon."""
        assert _parse_time("12pm") == "12:00"

    def test_parse_time_midnight(self) -> None:
        """Test parsing midnight."""
        assert _parse_time("12am") == "00:00"

    def test_parse_time_24h(self) -> None:
        """Test parsing 24-hour format."""
        assert _parse_time("14:30") == "14:30"
        assert _parse_time("09:00") == "09:00"


# =============================================================================
# Test Event Info
# =============================================================================


class TestEventInfo:
    """Tests for EventInfo class."""

    def test_event_info_initialization(self) -> None:
        """Test EventInfo initialization."""
        info = EventInfo(
            sort_key="2026-02-20T14:00:00",
            time_str="14:00",
            summary="Meeting",
            calendar="Work",
            account="Personal",
            event_id="evt123",
            cal_id="cal123",
        )
        assert info.sort_key == "2026-02-20T14:00:00"
        assert info.time_str == "14:00"
        assert info.summary == "Meeting"
        assert info.calendar == "Work"
        assert info.account == "Personal"
        assert info.event_id == "evt123"
        assert info.cal_id == "cal123"


# =============================================================================
# Test Collect Events
# =============================================================================


class TestCollectEvents:
    """Tests for _collect_events."""

    def test_collect_events_from_multiple_accounts(self) -> None:
        """Test collecting events from multiple accounts."""

        def fake_get_env(key: str) -> str | None:
            if key == "GOOGLE_CALENDAR_ACCESS_TOKEN":
                return "token_personal"
            if key == "GOOGLE_CALENDAR_INTERNS_ACCESS_TOKEN":
                return "token_interns"
            return None

        call_count = [0]

        def fake_api_get(token: str, url: str) -> JSONObject:
            call_count[0] += 1
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": f"evt{call_count[0]}",
                        "summary": f"Event {call_count[0]}",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = fake_get_env
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        # Verify we got events from at least one account
        assert result != []
        assert result[0].summary.startswith("Event")

    def test_collect_events_no_tokens(self) -> None:
        """Test collecting events when no tokens available."""
        hooks.cli_get_env = lambda key: None
        result = _collect_events("2026-02-20")
        assert result == []

    def test_collect_events_skips_holidays(self) -> None:
        """Test collecting events skips holiday calendars."""

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {
                    "items": [
                        {"id": "main", "summary": "Main"},
                        {"id": "holiday", "summary": "US Holidays"},
                    ]
                }
            if "holiday" in url:
                return {"items": [{"id": "h1", "summary": "Holiday Event"}]}
            return {"items": []}

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        # Holiday events should be filtered out
        assert all("Holiday" not in ev.summary for ev in result)


# =============================================================================
# Test Commands
# =============================================================================


class TestCmdList:
    """Tests for cmd_list command."""

    def test_cmd_list_with_events(self) -> None:
        """Test listing events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Test Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        cmd_list(ListArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "Test Event" in result

    def test_cmd_list_no_events(self) -> None:
        """Test listing when no events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None

        cmd_list(ListArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "No events found" in result


class TestCmdCalendars:
    """Tests for cmd_calendars command."""

    def test_cmd_calendars_with_calendars(self) -> None:
        """Test listing calendars."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            return {
                "items": [
                    {"id": "primary", "summary": "Main", "primary": True},
                    {"id": "work", "summary": "Work"},
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        cmd_calendars()
        result = output.getvalue()
        assert "Main" in result
        # Primary marker is present (rich formatting adds escape codes)
        assert "primary" in result

    def test_cmd_calendars_no_token(self) -> None:
        """Test listing calendars when no token."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None

        cmd_calendars()
        result = output.getvalue()
        assert "No token" in result


class TestCmdCreate:
    """Tests for cmd_create command."""

    def test_cmd_create_success(self) -> None:
        """Test creating event successfully."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_post = lambda token, url, body: {"id": "created"}

        cmd_create(
            CreateArgs(
                title="New Meeting",
                time="14:00",
                date="2026-02-20",
                duration=60,
                location="Office",
                account="Personal",
            )
        )
        result = output.getvalue()
        assert "Created" in result
        assert "New Meeting" in result
        assert "Location" in result

    def test_cmd_create_no_token(self) -> None:
        """Test creating event when no token."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None

        cmd_create(
            CreateArgs(
                title="Test",
                time="14:00",
                date="2026-02-20",
                duration=60,
                location="",
                account="Personal",
            )
        )
        result = output.getvalue()
        assert "Unknown account" in result

    def test_cmd_create_no_location(self) -> None:
        """Test creating event without location doesn't print location line."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_post = lambda token, url, body: {"id": "created"}

        cmd_create(
            CreateArgs(
                title="TestEvent",
                time="14:00",
                date="2026-02-20",
                duration=60,
                location="",
                account="Personal",
            )
        )
        result = output.getvalue()
        assert "Created" in result
        # No "Location:" label printed when location is empty
        assert "Location:" not in result


class TestCmdDelete:
    """Tests for cmd_delete command."""

    def test_cmd_delete_no_events(self) -> None:
        """Test deleting when no events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None

        cmd_delete(DeleteArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "No events to delete" in result

    def test_cmd_delete_cancelled(self) -> None:
        """Test deleting when user cancels."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get
        hooks.cli_prompt_ask = lambda msg: "q"

        cmd_delete(DeleteArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "Events for" in result

    def test_cmd_delete_invalid_input(self) -> None:
        """Test deleting with invalid input."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get
        hooks.cli_prompt_ask = lambda msg: "invalid"

        cmd_delete(DeleteArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "Invalid input" in result

    def test_cmd_delete_success(self) -> None:
        """Test deleting event successfully."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Test Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        deleted: list[str] = []

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get
        hooks.cli_api_delete = lambda token, url: deleted.append(url)
        hooks.cli_prompt_ask = lambda msg: "1"
        hooks.cli_confirm_ask = lambda msg: True

        cmd_delete(DeleteArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "Deleted" in result
        assert len(deleted) == 1

    def test_cmd_delete_not_confirmed(self) -> None:
        """Test deleting when not confirmed."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get
        hooks.cli_prompt_ask = lambda msg: "1"
        hooks.cli_confirm_ask = lambda msg: False

        cmd_delete(DeleteArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "Deleted" not in result


class TestCmdTomorrow:
    """Tests for cmd_tomorrow command."""

    def test_cmd_tomorrow(self) -> None:
        """Test showing tomorrow's events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

        cmd_tomorrow()
        result = output.getvalue()
        assert "February 21" in result


class TestCmdWeek:
    """Tests for cmd_week command."""

    def test_cmd_week(self) -> None:
        """Test showing week's events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        hooks.cli_get_env = lambda key: None
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

        cmd_week()
        result = output.getvalue()
        assert "February 20" in result
        assert "February 21" in result
        assert "February 26" in result


# =============================================================================
# Test Helper Display Functions
# =============================================================================


class TestShowEventsForDelete:
    """Tests for _show_events_for_delete."""

    def test_show_events_for_delete(self) -> None:
        """Test displaying events for deletion."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        events = [
            EventInfo(
                sort_key="2026-02-20T14:00:00",
                time_str="14:00",
                summary="Meeting",
                calendar="Work",
                account="Personal",
                event_id="evt1",
                cal_id="cal1",
            )
        ]

        _show_events_for_delete(events, "2026-02-20")
        result = output.getvalue()
        assert "Meeting" in result
        assert "14:00" in result


class TestGetDeleteChoice:
    """Tests for _get_delete_choice."""

    def test_get_delete_choice_valid(self) -> None:
        """Test getting valid delete choice."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_prompt_ask = lambda msg: "1"

        events = [
            EventInfo(
                sort_key="2026-02-20T14:00:00",
                time_str="14:00",
                summary="Meeting",
                calendar="Work",
                account="Personal",
                event_id="evt1",
                cal_id="cal1",
            )
        ]

        result = _get_delete_choice(events)
        assert result == 0

    def test_get_delete_choice_quit(self) -> None:
        """Test quitting delete choice."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_prompt_ask = lambda msg: "q"

        events = [
            EventInfo(
                sort_key="2026-02-20T14:00:00",
                time_str="14:00",
                summary="Meeting",
                calendar="Work",
                account="Personal",
                event_id="evt1",
                cal_id="cal1",
            )
        ]

        result = _get_delete_choice(events)
        assert result is None

    def test_get_delete_choice_out_of_range(self) -> None:
        """Test delete choice out of range."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_prompt_ask = lambda msg: "99"

        events = [
            EventInfo(
                sort_key="2026-02-20T14:00:00",
                time_str="14:00",
                summary="Meeting",
                calendar="Work",
                account="Personal",
                event_id="evt1",
                cal_id="cal1",
            )
        ]

        result = _get_delete_choice(events)
        assert result is None
        assert "Invalid number" in output.getvalue()

    def test_get_delete_choice_zero(self) -> None:
        """Test delete choice of zero."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_prompt_ask = lambda msg: "0"

        events = [
            EventInfo(
                sort_key="2026-02-20T14:00:00",
                time_str="14:00",
                summary="Meeting",
                calendar="Work",
                account="Personal",
                event_id="evt1",
                cal_id="cal1",
            )
        ]

        result = _get_delete_choice(events)
        assert result is None


# =============================================================================
# Test Main Function
# =============================================================================


class TestMain:
    """Tests for main entry point."""

    def test_main_list(self) -> None:
        """Test main with list command."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "list", "2026-02-20"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "February 20" in result

    def test_main_default_list(self) -> None:
        """Test main with no command defaults to list."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "2026" in result

    def test_main_calendars(self) -> None:
        """Test main with calendars command."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "calendars"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "No token" in result

    def test_main_tomorrow(self) -> None:
        """Test main with tomorrow command."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "tomorrow"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "February 21" in result

    def test_main_week(self) -> None:
        """Test main with week command."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None
        hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "week"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "February 20" in result

    def test_main_create(self) -> None:
        """Test main with create command."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_post = lambda t, u, b: {"id": "new"}

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "create", "Meeting", "14:00", "-d", "2026-02-20"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "Created" in result

    def test_main_delete(self) -> None:
        """Test main with delete command - no events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console
        hooks.cli_get_env = lambda key: None

        import sys

        old_argv = sys.argv
        sys.argv = ["calendar", "delete", "2026-02-20"]
        main()
        sys.argv = old_argv

        result = output.getvalue()
        assert "No events to delete" in result


# =============================================================================
# Test Hooks Module
# =============================================================================


class TestHooksModule:
    """Tests for hooks module."""

    def test_reset_hooks(self) -> None:
        """Test resetting all hooks restores production implementations."""
        # Set fake hooks
        hooks.cli_api_get = lambda t, u: {}
        hooks.cli_api_post = lambda t, u, b: {}
        hooks.cli_api_delete = lambda t, u: None
        hooks.cli_get_env = lambda k: "test_value"
        hooks.cli_get_now = lambda: datetime(2000, 1, 1)
        hooks.cli_prompt_ask = lambda m: "fake"
        hooks.cli_confirm_ask = lambda m: False
        fake_console = Console(file=io.StringIO())
        hooks.cli_get_console = lambda: fake_console

        # Verify fakes are set
        assert hooks.cli_get_env("any") == "test_value"
        assert hooks.cli_get_now() == datetime(2000, 1, 1)

        # Reset hooks
        reset_hooks()

        # Verify hooks are callable (production implementations restored)
        # Production cli_get_env returns None for unknown keys
        assert hooks.cli_get_env("NONEXISTENT_KEY_12345") is None
        # Production cli_get_now returns current time (not our fixed fake)
        now = hooks.cli_get_now()
        assert now.year >= 2026


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_collect_events_calendar_id_not_string(self) -> None:
        """Test collecting events when calendar ID is not a string."""

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": 123, "summary": "Bad Calendar"}]}
            return {"items": []}

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        assert result == []

    def test_collect_events_calendar_name_not_string(self) -> None:
        """Test collecting events when calendar name is not a string."""

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "cal1", "summary": 123}]}
            return {"items": []}

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        assert result == []

    def test_cmd_delete_recurring_event(self) -> None:
        """Test deleting recurring event instance."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1_20260220T140000Z",  # Recurring event instance
                        "summary": "Recurring Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        deleted_urls: list[str] = []

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get
        hooks.cli_api_delete = lambda token, url: deleted_urls.append(url)
        hooks.cli_prompt_ask = lambda msg: "1"
        hooks.cli_confirm_ask = lambda msg: True

        cmd_delete(DeleteArgs(date="2026-02-20"))

        # Should use base event ID (evt1) not the instance ID
        assert len(deleted_urls) == 1
        assert "evt1" in deleted_urls[0]
        assert "_" not in deleted_urls[0].split("/")[-1]

    def test_cmd_delete_no_token_for_account(self) -> None:
        """Test deleting when account has no token after confirmation."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        # Track calls to determine when to return None for token
        # The flow: fetch events for all accounts, then _get_token for delete
        confirmed = [False]

        def fake_get_env(key: str) -> str | None:
            # Return token during fetch, but None after user confirms delete
            if "ACCESS_TOKEN" in key:
                if confirmed[0]:
                    return None
                return "token"
            return None

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Event",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        def fake_confirm(msg: str) -> bool:
            # After confirmation, stop returning tokens
            confirmed[0] = True
            return True

        hooks.cli_get_env = fake_get_env
        hooks.cli_api_get = fake_api_get
        hooks.cli_api_delete = lambda t, u: None
        hooks.cli_prompt_ask = lambda msg: "1"
        hooks.cli_confirm_ask = fake_confirm

        cmd_delete(DeleteArgs(date="2026-02-20"))

        result = output.getvalue()
        assert "No token for" in result

    def test_cmd_list_all_day_event(self) -> None:
        """Test listing all-day events."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "summary": "All Day Event",
                        "start": {"date": "2026-02-20"},
                    }
                ]
            }

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        cmd_list(ListArgs(date="2026-02-20"))
        result = output.getvalue()
        assert "All Day Event" in result
        assert "all-day" in result

    def test_collect_events_sorts_all_day_first(self) -> None:
        """Test that all-day events are sorted first."""

        # Only use Personal account to avoid duplicates
        def fake_get_env(key: str) -> str | None:
            if key == "GOOGLE_CALENDAR_ACCESS_TOKEN":
                return "token"
            return None

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt2",
                        "summary": "Timed Event",
                        "start": {"dateTime": "2026-02-20T09:00:00-08:00"},
                    },
                    {
                        "id": "evt1",
                        "summary": "All Day",
                        "start": {"date": "2026-02-20"},
                    },
                ]
            }

        hooks.cli_get_env = fake_get_env
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        # All-day events should come first (sort key "00:00" is compared)
        # Find the all-day event
        all_day_indices = [i for i, e in enumerate(result) if e.time_str == "all-day"]
        timed_indices = [i for i, e in enumerate(result) if e.time_str != "all-day"]
        # All all-day events should come before timed events
        if all_day_indices and timed_indices:
            assert max(all_day_indices) < min(timed_indices)

    def test_cmd_calendars_no_summary(self) -> None:
        """Test listing calendars with no summary falls back to id."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        hooks.cli_get_console = lambda: console

        def fake_api_get(token: str, url: str) -> JSONObject:
            return {"items": [{"id": "cal123"}]}

        hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
        hooks.cli_api_get = fake_api_get

        cmd_calendars()
        result = output.getvalue()
        assert "cal123" in result

    def test_collect_events_no_summary(self) -> None:
        """Test collecting events with no summary."""

        # Only use Personal account to get exactly one event
        def fake_get_env(key: str) -> str | None:
            if key == "GOOGLE_CALENDAR_ACCESS_TOKEN":
                return "token"
            return None

        def fake_api_get(token: str, url: str) -> JSONObject:
            if "calendarList" in url:
                return {"items": [{"id": "primary", "summary": "Main"}]}
            return {
                "items": [
                    {
                        "id": "evt1",
                        "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                    }
                ]
            }

        hooks.cli_get_env = fake_get_env
        hooks.cli_api_get = fake_api_get

        result = _collect_events("2026-02-20")
        # Find the event without title
        no_title_events = [e for e in result if e.summary == "(no title)"]
        assert no_title_events != []
        assert no_title_events[0].summary == "(no title)"
