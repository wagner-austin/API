"""The networked server's own accounts: records, the file, and who is admitted."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str

from tankpit_bot.sim.lobby import SIM_ACCOUNT, LobbyError, SimAccountDict
from tankpit_bot.sim.net_accounts import (
    MemoryAccountBook,
    decode_account_book,
    decode_net_account,
    encode_account_book,
    encode_net_account,
    load_account_book,
    token_digest,
)
from tests.conftest import FakeFileSystem
from tests.sim._net_client import ACCOUNT, OTHER_ACCOUNT, TOKEN

_FILE = Path("config") / "net_accounts.json"


def test_a_token_is_kept_as_its_sha256() -> None:
    """The record stores a digest, never the token."""
    assert token_digest("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_an_account_round_trips_through_its_json() -> None:
    """Encode then decode is the record again, in a file of several."""
    assert decode_net_account(encode_net_account(ACCOUNT)) == ACCOUNT
    book = (ACCOUNT, OTHER_ACCOUNT)
    assert decode_account_book(encode_account_book(book)) == book


@pytest.mark.parametrize(
    "field, value",
    [
        ("token_sha256", "abc"),
        ("token_sha256", "A" * 64),
        ("account_id", ""),
        ("name", ""),
        ("rank", -1),
    ],
)
def test_an_invalid_record_is_refused(field: str, value: str | int) -> None:
    """A short or uppercase digest, an empty id or name, a negative rank."""
    record = encode_net_account(ACCOUNT)
    record[field] = value
    with pytest.raises(LobbyError, match="SIM_ACCOUNT_INVALID: account"):
        decode_net_account(record)


def test_a_file_listing_an_id_twice_is_refused() -> None:
    """One id, one account."""
    with pytest.raises(LobbyError, match=r"SIM_ACCOUNT_DUPLICATE: .*\['1001'\]"):
        decode_account_book(encode_account_book((ACCOUNT, ACCOUNT)))


def test_the_book_admits_an_account_by_its_token_and_reports_it() -> None:
    """A matching token yields what the join confirm reports."""
    assert MemoryAccountBook((ACCOUNT,)).verify("1001", TOKEN) == SimAccountDict(
        game_start="Oct. 05, 2026",
        name="austin",
        rank=3,
        active_forces=SIM_ACCOUNT["active_forces"],
    )


@pytest.mark.parametrize(("account_id", "token"), [("1001", "wrong"), ("9999", TOKEN)])
def test_the_book_denies_a_wrong_token_and_an_unknown_account_alike(
    account_id: str, token: str
) -> None:
    """The refusal does not say which, so it tells a guesser nothing."""
    with pytest.raises(LobbyError, match=f"SIM_LOBBY_DENIED: account '{account_id}' not admitted"):
        MemoryAccountBook((ACCOUNT,)).verify(account_id, token)


def test_an_account_file_loads_into_a_book(fake_fs: FakeFileSystem) -> None:
    """The file on disk is the book the server admits from."""
    fake_fs.write_text(_FILE, dump_json_str(encode_account_book((ACCOUNT, OTHER_ACCOUNT))))
    book = load_account_book(_FILE)
    assert book.verify("1001", TOKEN)["name"] == "austin"
