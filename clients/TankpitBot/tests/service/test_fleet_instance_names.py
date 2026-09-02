"""Tests for deriving a fleet instance name from an account.

``derive_instance`` is what spares a human from inventing instance
names: one account holds at most one live tank, so the account IS the
identity. These cover the sanitisation that turns an arbitrary
username into the namespace grammar ``spawn`` will accept.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot._test_hooks.fs import PathExistsProtocol, ReadTextProtocol
from tankpit_bot.service.fleet_config import derive_instance


@pytest.fixture()
def _accounts() -> Generator[list[str], None, None]:
    """Serve a configurable accounts file through the filesystem hooks."""
    usernames: list[str] = []
    original_exists: PathExistsProtocol = top_hooks.path_exists
    original_read: ReadTextProtocol = top_hooks.read_text

    def _path_exists(path: Path) -> bool:
        _ = path
        return bool(usernames)

    def _read_text(path: Path) -> str:
        _ = path
        entries = ", ".join(f'{{"username": "{name}", "password": "x"}}' for name in usernames)
        return f"[{entries}]"

    top_hooks.path_exists = _path_exists
    top_hooks.read_text = _read_text
    yield usernames
    top_hooks.path_exists = original_exists
    top_hooks.read_text = original_read


def test_derive_instance_falls_back_to_bot_without_accounts(_accounts: list[str]) -> None:
    """No account configured and none passed leaves the default identity."""
    assert derive_instance("") == "bot"


def test_derive_instance_uses_the_first_configured_account(_accounts: list[str]) -> None:
    """An empty selector takes the accounts file's first entry."""
    _accounts.extend(["Zephyr", "Second"])
    assert derive_instance("") == "zephyr"


def test_derive_instance_lowers_and_replaces_foreign_characters(
    _accounts: list[str],
) -> None:
    """Anything outside the grammar becomes a dash; hyphen and underscore stay."""
    assert derive_instance("Tank Pit.Bot/9") == "tank-pit-bot-9"
    assert derive_instance("keep-me_2") == "keep-me_2"


def test_derive_instance_prefixes_a_name_that_starts_non_alphanumeric(
    _accounts: list[str],
) -> None:
    """A leading dash is not a valid instance, so the name gains a ``b``.

    The sanitiser can produce one from an account whose first
    character was replaced -- ``spawn`` would then reject the very
    name this function exists to hand it.
    """
    assert derive_instance("_leading") == "b_leading"
    assert derive_instance(".dotted") == "b-dotted"


def test_derive_instance_prefixes_an_account_that_sanitises_to_nothing(
    _accounts: list[str],
) -> None:
    """An all-foreign username still yields a usable instance name."""
    assert derive_instance("...") == "b---"


def test_derive_instance_truncates_to_the_namespace_bound(_accounts: list[str]) -> None:
    """Long usernames are cut to 32 characters, prefix included."""
    assert derive_instance("a" * 40) == "a" * 32
    assert len(derive_instance("_" + "b" * 40)) == 32
