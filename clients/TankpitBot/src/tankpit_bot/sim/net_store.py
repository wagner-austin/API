"""The networked sim's accounts and sessions, kept in Postgres.

The multiplayer track keeps its accounts in the platform's existing
Postgres (``platform-postgres`` in the root compose), in a database of
its own, ``tankpit_sim``: a database, not another container. Two tables:

* ``sim_accounts`` -- one row per account this server issued: the
  token's SHA-256 (never the token), the tank's name, and the rank and
  decoration levels the account carries into a room.
* ``sim_sessions`` -- one row per seat that left a room: where it sat,
  for how many ticks, and the rank, kills, deaths and levels it left
  with.

:class:`PostgresAccountBook` answers the
:class:`~tankpit_bot.sim.net_accounts.AccountBookProtocol` from those
tables, so a player rejoins as what it left as however often the server
restarts. The connection comes through
:data:`tankpit_bot._test_hooks.connect_database`; nothing here imports
the driver.

Decoration levels are stored as their nine digits joined by commas, so a
row holds only text and integers, the two types the cursor seam reads.
"""

from __future__ import annotations

import hmac
import secrets
from collections.abc import Sequence

from tankpit_bot import _test_hooks
from tankpit_bot.sim.lobby import LobbyError
from tankpit_bot.sim.net_accounts import (
    FRESH_DECORATIONS,
    AdmittedAccount,
    NetAccountDict,
    SeatResult,
    admitted,
    require_decorations,
    token_digest,
)

DATABASE = "tankpit_sim"
"""The database the sim's tables live in, on the platform's Postgres."""

SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS sim_accounts (
        account_id TEXT PRIMARY KEY CHECK (account_id <> ''),
        token_sha256 TEXT NOT NULL CHECK (token_sha256 ~ '^[0-9a-f]{64}$'),
        name TEXT NOT NULL UNIQUE CHECK (name <> ''),
        rank INTEGER NOT NULL CHECK (rank >= 0),
        game_start TEXT NOT NULL,
        decorations TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sim_sessions (
        session_id BIGSERIAL PRIMARY KEY,
        account_id TEXT NOT NULL REFERENCES sim_accounts (account_id),
        room_id TEXT NOT NULL,
        field TEXT NOT NULL,
        ticks INTEGER NOT NULL CHECK (ticks >= 0),
        rank INTEGER NOT NULL CHECK (rank >= 0),
        kills INTEGER NOT NULL CHECK (kills >= 0),
        deaths INTEGER NOT NULL CHECK (deaths >= 0),
        decorations TEXT NOT NULL,
        recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
)
"""The two tables, each created only if it is not there yet."""

_ACCOUNT_COLUMNS = "account_id, token_sha256, name, rank, game_start, decorations"
_TOKEN_BYTES = 24


class StoreError(ValueError):
    """A row the store cannot read as an account (``SIM_STORE_*`` codes)."""


def encode_levels(levels: Sequence[int]) -> str:
    """Decoration levels as the text a row stores.

    Args:
        levels: The nine levels.

    Returns:
        The levels joined by commas.
    """
    return ",".join(str(level) for level in levels)


def decode_levels(text: str, account_id: str) -> tuple[int, ...]:
    """Read a row's decoration levels back.

    Args:
        text: The stored text.
        account_id: The account the row belongs to, for the message.

    Returns:
        The nine levels.

    Raises:
        StoreError: If a level is not a number (``SIM_STORE_LEVELS``).
        LobbyError: If there are not nine levels of 0 to 3.
    """
    parts = text.split(",")
    if not all(part.isdigit() for part in parts):
        raise StoreError(f"SIM_STORE_LEVELS: account {account_id!r} stores levels {text!r}")
    return require_decorations(tuple(int(part) for part in parts), account_id)


def _text(row: Sequence[str | int], index: int) -> str:
    """One text column of a row.

    Args:
        row: The row.
        index: The column.

    Returns:
        The column's text.

    Raises:
        StoreError: If the column holds a number (``SIM_STORE_ROW``).
    """
    value = row[index]
    if not isinstance(value, str):
        raise StoreError(f"SIM_STORE_ROW: column {index} holds {value!r}, not text")
    return value


def _number(row: Sequence[str | int], index: int) -> int:
    """One integer column of a row.

    Args:
        row: The row.
        index: The column.

    Returns:
        The column's integer.

    Raises:
        StoreError: If the column holds text (``SIM_STORE_ROW``).
    """
    value = row[index]
    if not isinstance(value, int):
        raise StoreError(f"SIM_STORE_ROW: column {index} holds {value!r}, not an integer")
    return value


def account_from_row(row: Sequence[str | int]) -> NetAccountDict:
    """Read one ``sim_accounts`` row, columns in :data:`_ACCOUNT_COLUMNS` order.

    Args:
        row: The row.

    Returns:
        The account record.

    Raises:
        StoreError: If a column holds the wrong type or the levels do not parse.
        LobbyError: If the levels are not nine of 0 to 3.
    """
    account_id = _text(row, 0)
    return NetAccountDict(
        account_id=account_id,
        token_sha256=_text(row, 1),
        name=_text(row, 2),
        rank=_number(row, 3),
        game_start=_text(row, 4),
        decorations=list(decode_levels(_text(row, 5), account_id)),
    )


def connect_store(variable: str) -> _test_hooks.DbConnectionProtocol:
    """Connect to the database whose connection string a variable holds.

    The string carries a password, so it is named by the variable that
    holds it rather than written on a command line.

    Args:
        variable: The environment variable's name.

    Returns:
        The open connection.

    Raises:
        StoreError: If the variable is unset or empty (``SIM_STORE_DSN``).
    """
    dsn = _test_hooks.get_env(variable)
    if not dsn:
        raise StoreError(f"SIM_STORE_DSN: ${variable} holds no tankpit_sim connection string")
    return _test_hooks.connect_database(dsn)


def ensure_schema(connection: _test_hooks.DbConnectionProtocol) -> None:
    """Create the sim's tables where they are missing, and commit.

    Args:
        connection: A connection to the ``tankpit_sim`` database.
    """
    cursor = connection.cursor()
    for statement in SCHEMA:
        cursor.execute(statement)
    connection.commit()


def issue_account(
    account_id: str, name: str, rank: int, game_start: str
) -> tuple[NetAccountDict, str]:
    """Make a new account and the token that admits it.

    The token is drawn here and returned once; the record holds only its
    digest, so the token cannot be read back from the store.

    Args:
        account_id: The account's id.
        name: The tank's name.
        rank: The rank it starts at.
        game_start: The date its join confirms report.

    Returns:
        The record, with nothing earned, and the token.
    """
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    account = NetAccountDict(
        account_id=account_id,
        token_sha256=token_digest(token),
        name=name,
        rank=rank,
        game_start=game_start,
        decorations=list(FRESH_DECORATIONS),
    )
    return account, token


class PostgresAccountBook:
    """An account book kept in the ``tankpit_sim`` database."""

    def __init__(self, connection: _test_hooks.DbConnectionProtocol) -> None:
        """Bind the book to its connection.

        Args:
            connection: A connection to the database, its schema in place
                (:func:`ensure_schema`).
        """
        self._connection = connection

    def verify(self, account_id: str, token: str) -> AdmittedAccount:
        """The account a client's AUTH frame names, once its token checks.

        Args:
            account_id: The account the client claims.
            token: The token it presents.

        Returns:
            The account as the room seats it.

        Raises:
            LobbyError: If the account is unknown or the token does not
                match (``SIM_LOBBY_DENIED``), alike, so a refusal tells a
                guesser nothing.
            StoreError: If the account's row does not read.
        """
        cursor = self._connection.cursor()
        cursor.execute(
            f"SELECT {_ACCOUNT_COLUMNS} FROM sim_accounts WHERE account_id = %s", (account_id,)
        )
        row = cursor.fetchone()
        account = None if row is None else account_from_row(row)
        if account is None or not hmac.compare_digest(account["token_sha256"], token_digest(token)):
            raise LobbyError(f"SIM_LOBBY_DENIED: account {account_id!r} not admitted")
        return admitted(account)

    def record(self, account_id: str, result: SeatResult) -> None:
        """Keep a seat's rank and levels on the account, and the seat as a session row.

        One transaction: the account and its session change together.

        Args:
            account_id: The account that sat.
            result: The seat's result.
        """
        levels = encode_levels(result.decorations)
        cursor = self._connection.cursor()
        cursor.execute(
            "UPDATE sim_accounts SET rank = %s, decorations = %s WHERE account_id = %s",
            (result.rank, levels, account_id),
        )
        cursor.execute(
            "INSERT INTO sim_sessions (account_id, room_id, field, ticks, rank, kills, deaths,"
            " decorations) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                account_id,
                result.room_id,
                result.field,
                result.ticks,
                result.rank,
                result.kills,
                result.deaths,
                levels,
            ),
        )
        self._connection.commit()

    def add(self, account: NetAccountDict) -> None:
        """Store a new account.

        Args:
            account: The record (:func:`issue_account`).
        """
        cursor = self._connection.cursor()
        cursor.execute(
            f"INSERT INTO sim_accounts ({_ACCOUNT_COLUMNS}) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                account["account_id"],
                account["token_sha256"],
                account["name"],
                account["rank"],
                account["game_start"],
                encode_levels(account["decorations"]),
            ),
        )
        self._connection.commit()

    def accounts(self) -> tuple[NetAccountDict, ...]:
        """Every stored account, by id.

        Returns:
            The records.

        Raises:
            StoreError: If a row does not read.
        """
        cursor = self._connection.cursor()
        cursor.execute(f"SELECT {_ACCOUNT_COLUMNS} FROM sim_accounts ORDER BY account_id")
        return tuple(account_from_row(row) for row in cursor.fetchall())


__all__ = [
    "DATABASE",
    "SCHEMA",
    "PostgresAccountBook",
    "StoreError",
    "account_from_row",
    "connect_store",
    "decode_levels",
    "encode_levels",
    "ensure_schema",
    "issue_account",
]
