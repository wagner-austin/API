"""CLI arg structures, extraction, accounts, env, tokens."""

from __future__ import annotations

import argparse
from datetime import datetime

from platform_calendar.cli import (
    _extract_int,
    _extract_str,
    _extract_str_or_none,
    decode_create_args,
    decode_delete_args,
    decode_list_args,
)
from platform_calendar.cli_auth import (
    Account,
    _get_env,
    _get_token,
)
from platform_calendar.testing import hooks

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
