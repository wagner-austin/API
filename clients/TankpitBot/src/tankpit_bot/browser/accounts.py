"""Account registry for multi-account support.

Loads accounts from accounts.json and resolves which account to use
based on TANKPIT_ACCOUNT (name or index). Supports fleet mode where
each bot instance selects a different account.

Resolution order:
1. TANKPIT_USERNAME + TANKPIT_PASSWORD env vars (explicit override)
2. TANKPIT_ACCOUNT + accounts.json (account list selection)
3. First account in accounts.json (default when no selector)
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_str,
)
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks

log = get_logger(__name__)

# accounts.json lives next to .env in the project root — of a SOURCE
# CHECKOUT. Four parents up from an installed module is site-packages,
# which is why the container overrides by env instead.
_CHECKOUT_ACCOUNTS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "accounts.json"


def accounts_file_path() -> Path:
    """Resolve where the account pool lives.

    ``TANKPIT_ACCOUNTS_FILE`` names the file explicitly — the fleet
    container mounts the pool read-only at a fixed path, because the
    checkout-relative default resolves into site-packages once the
    package is pip-installed (the limitation the single-bot image
    documented until this override existed). Unset, the source
    checkout's project root is the location it always was.

    Returns:
        The account pool path.
    """
    override = _test_hooks.get_env("TANKPIT_ACCOUNTS_FILE")
    if override is None or override == "":
        return _CHECKOUT_ACCOUNTS_PATH
    return Path(override)


class AccountNotFoundError(Exception):
    """Raised when TANKPIT_ACCOUNT selector doesn't match any account."""


class Account(TypedDict):
    """A single Tankpit account.

    Attributes:
        username: Tankpit account username.
        password: Tankpit account password.
    """

    username: str
    password: str


def encode_account(account: Account) -> JSONObject:
    """Encode Account to JSON-serializable dict.

    Args:
        account: Account to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "username": account["username"],
        "password": account["password"],
    }
    return result


def decode_account(data: JSONObject) -> Account:
    """Decode Account from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated Account.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return Account(
        username=require_str(data, "username"),
        password=require_str(data, "password"),
    )


def encode_account_list(accounts: list[Account]) -> list[JSONValue]:
    """Encode a list of accounts to JSON-serializable list.

    Args:
        accounts: List of accounts to encode.

    Returns:
        JSON-serializable list representation.
    """
    result: list[JSONValue] = [encode_account(a) for a in accounts]
    return result


def decode_account_list(data: JSONValue) -> list[Account]:
    """Decode a list of accounts from JSON value.

    Args:
        data: JSON value expected to be a list of account objects.

    Returns:
        List of validated Account dicts.

    Raises:
        JSONTypeError: If data is not a list or entries are invalid.
    """
    if not isinstance(data, list):
        actual = type(data).__name__
        raise JSONTypeError(f"Expected list of accounts, got {actual}")

    accounts: list[Account] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            actual = type(entry).__name__
            raise JSONTypeError(f"accounts[{i}]: expected object, got {actual}")
        accounts.append(decode_account(entry))
    return accounts


def load_accounts(path: Path) -> list[Account]:
    """Load accounts from a JSON file.

    Args:
        path: Path to accounts.json file.

    Returns:
        List of validated accounts.

    Raises:
        FileNotFoundError: If the file does not exist.
        InvalidJsonError: If the file is not valid JSON.
        JSONTypeError: If the JSON structure is invalid.
    """
    raw = _test_hooks.read_text(path)
    data: JSONValue = load_json_str(raw)
    return decode_account_list(data)


def select_account(accounts: list[Account], selector: str) -> Account:
    """Select an account by name or index.

    Args:
        accounts: List of accounts to select from.
        selector: Account username or numeric index (e.g. "Artax" or "0").

    Returns:
        The matching account.

    Raises:
        AccountNotFoundError: If selector doesn't match any account.
    """
    # Try as numeric index first
    if selector.isdigit() or (selector.startswith("-") and selector[1:].isdigit()):
        idx = int(selector)
        if 0 <= idx < len(accounts):
            log.info("Selected account #%d: %s", idx, accounts[idx]["username"])
            return accounts[idx]
        raise AccountNotFoundError(
            f"TANKPIT_ACCOUNT={idx} out of range (have {len(accounts)} accounts)"
        )

    # Try as username
    for acct in accounts:
        if acct["username"] == selector:
            log.info("Selected account by name: %s", selector)
            return acct

    available = ", ".join(a["username"] for a in accounts)
    raise AccountNotFoundError(f"TANKPIT_ACCOUNT='{selector}' not found. Available: {available}")


def resolve_account(path: Path | None = None) -> Account | None:
    """Resolve which account to use for login.

    Resolution order:
    1. TANKPIT_USERNAME + TANKPIT_PASSWORD env vars (explicit override)
    2. TANKPIT_ACCOUNT selector + accounts.json
    3. First account in accounts.json (when no selector set)

    Args:
        path: Override path to accounts.json. Uses default project root if None.

    Returns:
        Account to use, or None if no accounts are configured.

    Raises:
        AccountNotFoundError: If TANKPIT_ACCOUNT selector doesn't match any account,
            or if TANKPIT_ACCOUNT is set but accounts.json is missing.
        FileNotFoundError: If accounts.json does not exist when loading.
        InvalidJsonError: If accounts.json is not valid JSON.
        JSONTypeError: If account entries have invalid structure.
    """
    # 1. Explicit env var override takes priority
    env_user = _test_hooks.get_env("TANKPIT_USERNAME")
    env_pass = _test_hooks.get_env("TANKPIT_PASSWORD")
    if env_user is not None and env_pass is not None:
        log.info("Using account from env vars: %s", env_user)
        return Account(username=env_user, password=env_pass)

    # 2. Load accounts.json
    accounts_path = path or accounts_file_path()
    selector = _test_hooks.get_env("TANKPIT_ACCOUNT")

    if not _test_hooks.path_exists(accounts_path):
        if selector is not None:
            raise AccountNotFoundError(f"TANKPIT_ACCOUNT is set but {accounts_path} does not exist")
        return None

    accounts = load_accounts(accounts_path)
    if not accounts:
        return None

    # 3. Select account
    if selector is not None:
        return select_account(accounts, selector)

    # 4. Default to first account
    log.info("Using default account: %s", accounts[0]["username"])
    return accounts[0]


__all__ = [
    "Account",
    "AccountNotFoundError",
    "accounts_file_path",
    "decode_account",
    "decode_account_list",
    "encode_account",
    "encode_account_list",
    "load_accounts",
    "resolve_account",
    "select_account",
]
