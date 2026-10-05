"""The sim server's own accounts: who may join a networked room.

A networked room admits a client only for an account it issued itself.
These are this server's accounts, never tankpit.com's: the multiplayer
track (board task b008ab91) plays TankPit-derived mechanics with our own
client, name and accounts, and handles nobody's tankpit.com
credentials.

An account record holds the SHA-256 of its token, never the token, so a
leaked account file names players without letting anyone join as them.
The client presents the token in its AUTH frame
(:func:`~tankpit_bot.sim.lobby.parse_auth_frame`), and the book compares
digests in constant time.

:class:`AccountBookProtocol` is the seam. :class:`MemoryAccountBook`
answers from records loaded at start (:func:`load_account_book`); a
persistent book answers the same two questions from a database.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Protocol, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot import _test_hooks
from tankpit_bot.sim.lobby import SIM_ACCOUNT, LobbyError, SimAccountDict

_SHA256_HEX_LENGTH = 64


class NetAccountDict(TypedDict):
    """One account this server issued.

    Attributes:
        account_id: The id the client names in its AUTH frame.
        token_sha256: Hex SHA-256 of the account's token.
        name: The tank's wire name in the room.
        rank: The tank's rank on joining.
        game_start: The date the join confirm reports the account began.
    """

    account_id: str
    token_sha256: str
    name: str
    rank: int
    game_start: str


class AccountBookProtocol(Protocol):
    """What a networked room asks of its accounts."""

    def verify(self, account_id: str, token: str) -> SimAccountDict:
        """The account a client's AUTH frame names, once its token checks.

        Args:
            account_id: The account the client claims.
            token: The token it presents.

        Returns:
            What the room's join confirms report for the account.

        Raises:
            LobbyError: If the account is unknown or the token does not
                match (``SIM_LOBBY_DENIED``).
        """
        ...


def token_digest(token: str) -> str:
    """The digest an account record stores for a token.

    Args:
        token: The token as the client presents it.

    Returns:
        Its hex SHA-256.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def encode_net_account(account: NetAccountDict) -> JSONObject:
    """Encode one account record for its JSON file.

    Args:
        account: The record.

    Returns:
        Its JSON object.
    """
    return {
        "account_id": account["account_id"],
        "token_sha256": account["token_sha256"],
        "name": account["name"],
        "rank": account["rank"],
        "game_start": account["game_start"],
    }


def decode_net_account(data: JSONObject) -> NetAccountDict:
    """Decode and validate one account record.

    Args:
        data: The record's JSON object.

    Returns:
        The record.

    Raises:
        LobbyError: If the digest is not 64 hex digits, a name or id is
            empty, or the rank is negative (``SIM_ACCOUNT_INVALID``).
        TypeError: If a field is missing or of the wrong JSON type.
    """
    account = NetAccountDict(
        account_id=require_str(data, "account_id"),
        token_sha256=require_str(data, "token_sha256"),
        name=require_str(data, "name"),
        rank=require_int(data, "rank"),
        game_start=require_str(data, "game_start"),
    )
    digest = account["token_sha256"]
    hex_digest = len(digest) == _SHA256_HEX_LENGTH and all(c in "0123456789abcdef" for c in digest)
    if not hex_digest or not account["account_id"] or not account["name"] or account["rank"] < 0:
        raise LobbyError(
            f"SIM_ACCOUNT_INVALID: account {account['account_id']!r} needs an id, a name,"
            " a rank of 0 or more and a 64-digit lowercase hex token_sha256"
        )
    return account


def encode_account_book(accounts: tuple[NetAccountDict, ...]) -> JSONObject:
    """Encode an account file.

    Args:
        accounts: The records, in file order.

    Returns:
        The file's JSON object.
    """
    records: list[JSONValue] = [encode_net_account(account) for account in accounts]
    return {"accounts": records}


def decode_account_book(data: JSONObject) -> tuple[NetAccountDict, ...]:
    """Decode an account file.

    Args:
        data: The file's JSON object.

    Returns:
        The records, in file order.

    Raises:
        LobbyError: If a record is invalid, or two share an account id
            (``SIM_ACCOUNT_INVALID``, ``SIM_ACCOUNT_DUPLICATE``).
        TypeError: If the file is not shaped as an account file.
    """
    accounts = tuple(
        decode_net_account(narrow_json_to_dict(item)) for item in require_list(data, "accounts")
    )
    ids = [account["account_id"] for account in accounts]
    duplicates = sorted({account_id for account_id in ids if ids.count(account_id) > 1})
    if duplicates:
        raise LobbyError(f"SIM_ACCOUNT_DUPLICATE: account ids listed twice: {duplicates}")
    return accounts


class MemoryAccountBook:
    """An account book answered from records held in memory."""

    def __init__(self, accounts: tuple[NetAccountDict, ...]) -> None:
        """Hold the records, keyed by account id.

        Args:
            accounts: The records, ids unique (:func:`decode_account_book`).
        """
        self._accounts = {account["account_id"]: account for account in accounts}

    def verify(self, account_id: str, token: str) -> SimAccountDict:
        """The account a client's AUTH frame names, once its token checks.

        Args:
            account_id: The account the client claims.
            token: The token it presents.

        Returns:
            What the room's join confirms report for the account.

        Raises:
            LobbyError: If the account is unknown or the token does not
                match (``SIM_LOBBY_DENIED``). The message does not say
                which, so it tells a guesser nothing.
        """
        account = self._accounts.get(account_id)
        if account is None or not hmac.compare_digest(account["token_sha256"], token_digest(token)):
            raise LobbyError(f"SIM_LOBBY_DENIED: account {account_id!r} not admitted")
        return SimAccountDict(
            game_start=account["game_start"],
            name=account["name"],
            rank=account["rank"],
            active_forces=SIM_ACCOUNT["active_forces"],
        )


def load_account_book(path: Path) -> MemoryAccountBook:
    """Load an account file into a book.

    Args:
        path: The JSON account file (``{"accounts": [...]}``).

    Returns:
        The book.

    Raises:
        LobbyError: If a record is invalid or an id repeats.
        TypeError: If the file is not shaped as an account file.
    """
    return MemoryAccountBook(
        decode_account_book(narrow_json_to_dict(load_json_str(_test_hooks.read_text(path))))
    )


__all__ = [
    "AccountBookProtocol",
    "MemoryAccountBook",
    "NetAccountDict",
    "decode_account_book",
    "decode_net_account",
    "encode_account_book",
    "encode_net_account",
    "load_account_book",
    "token_digest",
]
