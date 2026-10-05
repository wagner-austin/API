"""The networked sim's Postgres store: rows, statements, and the real connector."""

from __future__ import annotations

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.sim.lobby import LobbyError
from tankpit_bot.sim.net_accounts import FRESH_DECORATIONS, SeatResult, admitted, token_digest
from tankpit_bot.sim.net_store import (
    SCHEMA,
    PostgresAccountBook,
    StoreError,
    account_from_row,
    connect_store,
    decode_levels,
    encode_levels,
    ensure_schema,
    issue_account,
)
from tests.sim._fake_db import DSN, DSN_VARIABLE, FakeDatabase
from tests.sim._net_client import ACCOUNT, TOKEN

_ROW: tuple[str | int, ...] = (
    "1001",
    ACCOUNT["token_sha256"],
    "austin",
    3,
    "Oct. 05, 2026",
    "1,2,0,0,0,0,0,0,0",
)
"""ACCOUNT as its sim_accounts row."""

_RESULT = SeatResult(
    room_id="1",
    field="field05_r.gif",
    ticks=40,
    rank=4,
    kills=2,
    deaths=1,
    decorations=(1, 2, 1, 0, 0, 0, 0, 0, 0),
)


def _book(fake_db: FakeDatabase) -> PostgresAccountBook:
    """A book over a connection to the fake database."""
    return PostgresAccountBook(fake_db.connect(DSN))


def test_levels_are_stored_as_their_digits_and_read_back() -> None:
    """Nine levels, comma-joined, round-trip."""
    assert encode_levels((1, 2, 0, 0, 0, 0, 0, 0, 3)) == "1,2,0,0,0,0,0,0,3"
    assert decode_levels("1,2,0,0,0,0,0,0,3", "1001") == (1, 2, 0, 0, 0, 0, 0, 0, 3)


def test_stored_levels_that_do_not_read_are_refused() -> None:
    """Text that is not digits, or digits that are not nine levels of 0 to 3."""
    with pytest.raises(StoreError, match="SIM_STORE_LEVELS: account '1001' stores levels 'x,1'"):
        decode_levels("x,1", "1001")
    with pytest.raises(LobbyError, match="SIM_ACCOUNT_INVALID: account '1001' needs 9"):
        decode_levels("1,2", "1001")


def test_an_account_row_reads_as_its_record() -> None:
    """Columns in order, levels parsed."""
    assert account_from_row(_ROW) == ACCOUNT


@pytest.mark.parametrize(
    ("column", "value", "says"),
    [(0, 1001, "column 0 holds 1001, not text"), (3, "3", "column 3 holds '3', not an integer")],
)
def test_a_row_with_a_column_of_the_wrong_type_is_refused(
    column: int, value: str | int, says: str
) -> None:
    """The cursor seam reads text and integers; each column must be the one it should."""
    row = list(_ROW)
    row[column] = value
    with pytest.raises(StoreError, match=f"SIM_STORE_ROW: {says}"):
        account_from_row(row)


def test_the_schema_is_both_tables_then_a_commit(fake_db: FakeDatabase) -> None:
    """Each table only if it is missing, in one transaction."""
    ensure_schema(fake_db.connect(DSN))
    assert [sql for sql, _ in fake_db.executed] == list(SCHEMA)
    assert fake_db.commits == 1


def test_an_issued_account_admits_only_its_own_token() -> None:
    """The record keeps the digest of the token it returns, and nothing earned."""
    account, token = issue_account("2001", "orion", 2, "Oct. 05, 2026")
    assert account["token_sha256"] == token_digest(token)
    assert (account["name"], account["rank"], account["decorations"]) == (
        "orion",
        2,
        list(FRESH_DECORATIONS),
    )
    assert issue_account("2001", "orion", 2, "Oct. 05, 2026")[1] != token


def test_the_store_connects_with_the_string_its_variable_holds(fake_db: FakeDatabase) -> None:
    """The variable names the string; the string reaches the driver."""
    connect_store(DSN_VARIABLE)
    assert fake_db.dsns == [DSN]
    with pytest.raises(StoreError, match="SIM_STORE_DSN: \\$UNSET_VARIABLE holds no tankpit_sim"):
        connect_store("UNSET_VARIABLE")


def test_the_book_admits_an_account_by_its_token(fake_db: FakeDatabase) -> None:
    """One query by id; a matching token yields the account as a room seats it."""
    fake_db.answers.append([_ROW])
    assert _book(fake_db).verify("1001", TOKEN) == admitted(ACCOUNT)
    assert fake_db.executed[0][1] == ("1001",)
    assert fake_db.statements()[0].startswith("SELECT account_id, token_sha256, name")


@pytest.mark.parametrize(("rows", "token"), [([], TOKEN), ([_ROW], "guessed")])
def test_the_book_denies_an_unknown_account_and_a_wrong_token_alike(
    fake_db: FakeDatabase, rows: list[tuple[str | int, ...]], token: str
) -> None:
    """The refusal does not say which."""
    fake_db.answers.append(list(rows))
    with pytest.raises(LobbyError, match="SIM_LOBBY_DENIED: account '1001' not admitted"):
        _book(fake_db).verify("1001", token)


def test_a_seat_updates_the_account_and_adds_a_session_in_one_commit(
    fake_db: FakeDatabase,
) -> None:
    """Rank and levels onto the account, the seat as a session row, then one commit."""
    _book(fake_db).record("1001", _RESULT)

    assert fake_db.statements()[0] == (
        "UPDATE sim_accounts SET rank = %s, decorations = %s WHERE account_id = %s"
    )
    assert fake_db.executed[0][1] == (4, "1,2,1,0,0,0,0,0,0", "1001")
    assert fake_db.statements()[1].startswith("INSERT INTO sim_sessions")
    assert fake_db.executed[1][1] == (
        "1001",
        "1",
        "field05_r.gif",
        40,
        4,
        2,
        1,
        "1,2,1,0,0,0,0,0,0",
    )
    assert fake_db.commits == 1


def test_an_added_account_is_inserted_and_listed_back(fake_db: FakeDatabase) -> None:
    """Add writes the row; accounts reads every row back as records."""
    book = _book(fake_db)
    book.add(ACCOUNT)
    assert fake_db.executed[0][1] == _ROW
    assert fake_db.commits == 1

    fake_db.answers.append([_ROW])
    assert book.accounts() == (ACCOUNT,)
    assert fake_db.statements()[1].endswith("FROM sim_accounts ORDER BY account_id")


def test_the_real_connector_fails_honestly_when_nothing_listens() -> None:
    """The connect body runs: import, dial, and the driver's own refusal."""
    operational_error: type[Exception] = __import__("psycopg").OperationalError
    with pytest.raises(operational_error):
        _test_hooks._real_connect_database("host=127.0.0.1 port=1 dbname=nothing connect_timeout=1")
