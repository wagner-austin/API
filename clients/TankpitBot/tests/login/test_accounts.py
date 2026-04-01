"""Tests for browser.accounts module — multi-account registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest
from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
)

from tankpit_bot import _test_hooks
from tankpit_bot.browser.accounts import (
    Account,
    AccountNotFoundError,
    decode_account,
    decode_account_list,
    encode_account,
    encode_account_list,
    load_accounts,
    resolve_account,
    select_account,
)

# =============================================================================
# Encode / Decode
# =============================================================================


class TestEncodeAccount:
    """Tests for encode_account."""

    def test_encodes_all_fields(self) -> None:
        """Encodes username and password to JSON object."""
        account = Account(username="Artax", password="secret123")
        result = encode_account(account)

        assert result["username"] == "Artax"
        assert result["password"] == "secret123"
        assert len(result) == 2

    def test_roundtrip(self) -> None:
        """Encode then decode produces identical Account."""
        original = Account(username="Yuppler", password="pass456")
        encoded = encode_account(original)
        decoded = decode_account(encoded)

        assert decoded["username"] == original["username"]
        assert decoded["password"] == original["password"]


class TestDecodeAccount:
    """Tests for decode_account."""

    def test_decodes_valid_object(self) -> None:
        """Decodes a dict with username and password."""
        data: JSONObject = {"username": "Artax", "password": "secret123"}
        result = decode_account(data)

        assert result["username"] == "Artax"
        assert result["password"] == "secret123"

    def test_raises_on_missing_username(self) -> None:
        """Raises JSONTypeError when username is missing."""
        with pytest.raises(JSONTypeError):
            decode_account({"password": "secret123"})

    def test_raises_on_missing_password(self) -> None:
        """Raises JSONTypeError when password is missing."""
        with pytest.raises(JSONTypeError):
            decode_account({"username": "Artax"})

    def test_raises_on_non_string_username(self) -> None:
        """Raises JSONTypeError when username is not a string."""
        with pytest.raises(JSONTypeError):
            decode_account({"username": 123, "password": "secret"})

    def test_raises_on_non_string_password(self) -> None:
        """Raises JSONTypeError when password is not a string."""
        with pytest.raises(JSONTypeError):
            decode_account({"username": "Artax", "password": 456})


class TestEncodeAccountList:
    """Tests for encode_account_list."""

    def test_encodes_empty_list(self) -> None:
        """Empty list encodes to empty list."""
        assert encode_account_list([]) == []

    def test_encodes_multiple_accounts(self) -> None:
        """Encodes list of accounts to list of JSON objects."""
        accounts = [
            Account(username="Artax", password="p1"),
            Account(username="Yuppler", password="p2"),
        ]
        result = encode_account_list(accounts)

        assert len(result) == 2
        first = result[0]
        assert first == {"username": "Artax", "password": "p1"}

    def test_roundtrip_list(self) -> None:
        """Encode then decode list produces identical accounts."""
        originals = [
            Account(username="A", password="pa"),
            Account(username="B", password="pb"),
        ]
        encoded = encode_account_list(originals)
        decoded = decode_account_list(encoded)

        assert decoded[0]["username"] == "A"
        assert decoded[1]["username"] == "B"


class TestDecodeAccountList:
    """Tests for decode_account_list."""

    def test_decodes_valid_list(self) -> None:
        """Decodes a list of valid account dicts."""
        data: JSONValue = [
            {"username": "Artax", "password": "p1"},
            {"username": "Yuppler", "password": "p2"},
        ]
        result = decode_account_list(data)

        assert len(result) == 2
        assert result[0]["username"] == "Artax"
        assert result[1]["username"] == "Yuppler"

    def test_decodes_empty_list(self) -> None:
        """Empty JSON array produces empty list."""
        assert decode_account_list([]) == []

    def test_raises_on_non_list(self) -> None:
        """Raises JSONTypeError when data is not a list."""
        with pytest.raises(JSONTypeError, match="Expected list"):
            decode_account_list({"username": "Artax", "password": "p"})

    def test_raises_on_non_dict_entry(self) -> None:
        """Raises JSONTypeError when list entry is not a dict."""
        with pytest.raises(JSONTypeError, match=r"accounts\[0\]"):
            decode_account_list(["not a dict"])

    def test_raises_on_invalid_entry(self) -> None:
        """Raises JSONTypeError when entry has invalid fields."""
        with pytest.raises(JSONTypeError):
            decode_account_list([{"username": "ok"}])  # missing password


# =============================================================================
# load_accounts
# =============================================================================


class TestLoadAccounts:
    """Tests for load_accounts using _test_hooks."""

    def test_loads_valid_file(self, tmp_path: Path) -> None:
        """Loads and decodes a valid accounts.json file."""
        accounts_file = tmp_path / "accounts.json"
        content = dump_json_str(
            [
                {"username": "Artax", "password": "p1"},
                {"username": "Yuppler", "password": "p2"},
            ]
        )
        _test_hooks.write_text(accounts_file, content)

        result = load_accounts(accounts_file)

        assert len(result) == 2
        assert result[0]["username"] == "Artax"
        assert result[1]["password"] == "p2"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when file does not exist."""
        missing = tmp_path / "nope.json"

        with pytest.raises(FileNotFoundError):
            load_accounts(missing)

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        """Raises InvalidJsonError on malformed JSON."""
        bad_file = tmp_path / "bad.json"
        _test_hooks.write_text(bad_file, "not valid json{{{")

        with pytest.raises(InvalidJsonError):
            load_accounts(bad_file)

    def test_raises_on_non_list_json(self, tmp_path: Path) -> None:
        """Raises JSONTypeError when JSON root is not a list."""
        bad_file = tmp_path / "obj.json"
        _test_hooks.write_text(bad_file, dump_json_str({"username": "x", "password": "y"}))

        with pytest.raises(JSONTypeError, match="Expected list"):
            load_accounts(bad_file)


# =============================================================================
# select_account
# =============================================================================


class TestSelectAccount:
    """Tests for select_account."""

    ACCOUNTS: ClassVar[list[Account]] = [
        Account(username="Artax", password="p1"),
        Account(username="Yuppler", password="p2"),
        Account(username="Recon", password="p3"),
    ]

    def test_select_by_index_zero(self) -> None:
        """Selects first account by index 0."""
        result = select_account(self.ACCOUNTS, "0")
        assert result["username"] == "Artax"

    def test_select_by_index_one(self) -> None:
        """Selects second account by index 1."""
        result = select_account(self.ACCOUNTS, "1")
        assert result["username"] == "Yuppler"

    def test_select_by_index_last(self) -> None:
        """Selects last account by index."""
        result = select_account(self.ACCOUNTS, "2")
        assert result["username"] == "Recon"

    def test_select_by_name(self) -> None:
        """Selects account by username."""
        result = select_account(self.ACCOUNTS, "Yuppler")
        assert result["username"] == "Yuppler"
        assert result["password"] == "p2"

    def test_raises_on_index_out_of_range(self) -> None:
        """Raises AccountNotFoundError when index exceeds list length."""
        with pytest.raises(AccountNotFoundError, match="out of range"):
            select_account(self.ACCOUNTS, "5")

    def test_raises_on_negative_index(self) -> None:
        """Raises AccountNotFoundError for negative index."""
        with pytest.raises(AccountNotFoundError, match="out of range"):
            select_account(self.ACCOUNTS, "-1")

    def test_raises_on_unknown_name(self) -> None:
        """Raises AccountNotFoundError when name doesn't match."""
        with pytest.raises(AccountNotFoundError, match="not found"):
            select_account(self.ACCOUNTS, "Ghost")

    def test_error_message_lists_available(self) -> None:
        """AccountNotFoundError message lists available account names."""
        with pytest.raises(AccountNotFoundError, match=r"Artax.*Yuppler.*Recon"):
            select_account(self.ACCOUNTS, "Unknown")


# =============================================================================
# resolve_account
# =============================================================================


def _write_accounts_json(path: Path, accounts: list[dict[str, str]]) -> None:
    """Write accounts list to a JSON file via test hooks.

    Args:
        path: File path to write.
        accounts: List of account dicts to serialize.
    """
    _test_hooks.write_text(path, dump_json_str(accounts))


def _swap_env(
    env_vars: dict[str, str],
) -> tuple[Callable[[str], str | None], Callable[[str], str | None]]:
    """Create fake get_env and capture original for restore.

    Args:
        env_vars: Environment variable map for the fake.

    Returns:
        Tuple of (fake_get_env, original_get_env).
    """
    original = _test_hooks.get_env

    def fake_get_env(key: str) -> str | None:
        return env_vars.get(key)

    return fake_get_env, original


class TestResolveAccount:
    """Tests for resolve_account full resolution flow."""

    def test_env_vars_override_everything(self, tmp_path: Path) -> None:
        """TANKPIT_USERNAME + TANKPIT_PASSWORD take priority over accounts.json."""
        accounts_file = tmp_path / "accounts.json"
        _write_accounts_json(accounts_file, [{"username": "FileAccount", "password": "fp"}])

        fake, original = _swap_env(
            {
                "TANKPIT_USERNAME": "EnvUser",
                "TANKPIT_PASSWORD": "EnvPass",
                "TANKPIT_ACCOUNT": "0",
            }
        )
        _test_hooks.get_env = fake
        try:
            result = resolve_account(accounts_file)
        finally:
            _test_hooks.get_env = original

        assert result == Account(username="EnvUser", password="EnvPass")

    def test_selects_by_name_from_file(self, tmp_path: Path) -> None:
        """TANKPIT_ACCOUNT selects account by name from accounts.json."""
        accounts_file = tmp_path / "accounts.json"
        _write_accounts_json(
            accounts_file,
            [
                {"username": "Artax", "password": "ap"},
                {"username": "Yuppler", "password": "yp"},
            ],
        )

        fake, original = _swap_env({"TANKPIT_ACCOUNT": "Yuppler"})
        _test_hooks.get_env = fake
        try:
            result = resolve_account(accounts_file)
        finally:
            _test_hooks.get_env = original

        assert result == Account(username="Yuppler", password="yp")

    def test_selects_by_index_from_file(self, tmp_path: Path) -> None:
        """TANKPIT_ACCOUNT selects account by index from accounts.json."""
        accounts_file = tmp_path / "accounts.json"
        _write_accounts_json(
            accounts_file,
            [
                {"username": "Artax", "password": "ap"},
                {"username": "Yuppler", "password": "yp"},
            ],
        )

        fake, original = _swap_env({"TANKPIT_ACCOUNT": "1"})
        _test_hooks.get_env = fake
        try:
            result = resolve_account(accounts_file)
        finally:
            _test_hooks.get_env = original

        assert result == Account(username="Yuppler", password="yp")

    def test_defaults_to_first_account(self, tmp_path: Path) -> None:
        """Returns first account when no TANKPIT_ACCOUNT is set."""
        accounts_file = tmp_path / "accounts.json"
        _write_accounts_json(
            accounts_file,
            [
                {"username": "Artax", "password": "ap"},
                {"username": "Yuppler", "password": "yp"},
            ],
        )

        fake, original = _swap_env({})
        _test_hooks.get_env = fake
        try:
            result = resolve_account(accounts_file)
        finally:
            _test_hooks.get_env = original

        assert result == Account(username="Artax", password="ap")

    def test_returns_none_when_no_accounts_configured(self, tmp_path: Path) -> None:
        """Returns None when no env vars and no accounts.json."""
        missing = tmp_path / "nonexistent.json"

        fake, original = _swap_env({})
        _test_hooks.get_env = fake
        try:
            result = resolve_account(missing)
        finally:
            _test_hooks.get_env = original

        assert result is None

    def test_raises_when_selector_set_but_file_missing(self, tmp_path: Path) -> None:
        """Raises AccountNotFoundError when TANKPIT_ACCOUNT set but no file."""
        missing = tmp_path / "nonexistent.json"

        fake, original = _swap_env({"TANKPIT_ACCOUNT": "Artax"})
        _test_hooks.get_env = fake
        try:
            with pytest.raises(AccountNotFoundError, match="does not exist"):
                resolve_account(missing)
        finally:
            _test_hooks.get_env = original

    def test_raises_when_selector_not_found(self, tmp_path: Path) -> None:
        """Raises AccountNotFoundError when TANKPIT_ACCOUNT doesn't match."""
        accounts_file = tmp_path / "accounts.json"
        _write_accounts_json(accounts_file, [{"username": "Artax", "password": "ap"}])

        fake, original = _swap_env({"TANKPIT_ACCOUNT": "Ghost"})
        _test_hooks.get_env = fake
        try:
            with pytest.raises(AccountNotFoundError, match="not found"):
                resolve_account(accounts_file)
        finally:
            _test_hooks.get_env = original

    def test_returns_none_on_empty_accounts_list(self, tmp_path: Path) -> None:
        """Returns None when accounts.json is an empty array."""
        accounts_file = tmp_path / "accounts.json"
        _write_accounts_json(accounts_file, [])

        fake, original = _swap_env({})
        _test_hooks.get_env = fake
        try:
            result = resolve_account(accounts_file)
        finally:
            _test_hooks.get_env = original

        assert result is None

    def test_propagates_invalid_json_error(self, tmp_path: Path) -> None:
        """Propagates InvalidJsonError from malformed file."""
        bad_file = tmp_path / "bad.json"
        _test_hooks.write_text(bad_file, "{not json")

        fake, original = _swap_env({})
        _test_hooks.get_env = fake
        try:
            with pytest.raises(InvalidJsonError):
                resolve_account(bad_file)
        finally:
            _test_hooks.get_env = original

    def test_propagates_json_type_error(self, tmp_path: Path) -> None:
        """Propagates JSONTypeError from invalid account structure."""
        bad_file = tmp_path / "bad.json"
        _test_hooks.write_text(bad_file, dump_json_str([{"username": 123}]))

        fake, original = _swap_env({})
        _test_hooks.get_env = fake
        try:
            with pytest.raises(JSONTypeError):
                resolve_account(bad_file)
        finally:
            _test_hooks.get_env = original
