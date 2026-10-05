"""``tankpit-sim-accounts``: init, add and list against the sim's database."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.sim.net_accounts import token_digest
from tankpit_bot.sim.net_accounts_cli import AccountsUsageError, main
from tankpit_bot.sim.net_store import SCHEMA
from tests.sim._fake_db import DSN_VARIABLE, FakeDatabase

_OCT_5_2026_MS = 1_791_200_000_000


@pytest.fixture()
def october_fifth() -> Generator[None, None, None]:
    """The clock reads 2026-10-05.

    Yields:
        Nothing; the clock is set for the test.
    """
    real = _test_hooks.get_current_time_ms

    def clock() -> int:
        """2026-10-05, in milliseconds."""
        return _OCT_5_2026_MS

    _test_hooks.get_current_time_ms = clock
    yield
    _test_hooks.get_current_time_ms = real


def test_init_makes_the_tables_and_closes(
    fake_db: FakeDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    """The schema runs, commits, and the connection is closed."""
    assert main(["init", "--database-env", DSN_VARIABLE]) == 0
    assert [sql for sql, _ in fake_db.executed] == list(SCHEMA)
    assert fake_db.closed == 1
    assert capsys.readouterr().out == "tankpit_sim tables are in place\n"


def test_add_issues_an_account_and_prints_its_token_once(
    fake_db: FakeDatabase, october_fifth: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The row holds the digest of the printed token, dated today, at the named rank."""
    del october_fifth
    assert main(["add", "--database-env", DSN_VARIABLE, "--id", "2001", "--name", "orion"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert (
        lines[0]
        == "account 2001 (orion) issued; its token, shown once and stored only as a digest:"
    )
    inserted = fake_db.executed[len(SCHEMA)][1]
    assert inserted == (
        "2001",
        token_digest(lines[1]),
        "orion",
        0,
        "Oct. 05, 2026",
        "0,0,0,0,0,0,0,0,0",
    )
    assert fake_db.closed == 1


def test_add_takes_a_starting_rank(fake_db: FakeDatabase, october_fifth: None) -> None:
    """``--rank`` names the rank the account starts at."""
    del october_fifth
    main(["add", "--database-env", DSN_VARIABLE, "--id", "2002", "--name", "vega", "--rank", "5"])
    assert fake_db.executed[len(SCHEMA)][1][3] == 5


def test_list_prints_every_account(
    fake_db: FakeDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    """Id, name, rank and levels, one line each."""
    fake_db.answers.append([("1001", "a" * 64, "austin", 3, "Oct. 05, 2026", "1,2,0,0,0,0,0,0,0")])
    assert main(["list", "--database-env", DSN_VARIABLE]) == 0
    assert capsys.readouterr().out == "1001\taustin\trank 3\tdecorations 1,2,0,0,0,0,0,0,0\n"


@pytest.mark.parametrize(
    ("argv", "says"),
    [
        ([], "the first argument is one of init, add, list"),
        (["drop"], "the first argument is one of init, add, list"),
        (["init"], "--database-env NAME names the variable"),
        (["init", "--database-env"], "unknown flag or missing value at '--database-env'"),
        (["list", "--host", "x"], "unknown flag or missing value at '--host'"),
    ],
)
def test_a_command_line_it_cannot_run_is_refused_before_connecting(
    fake_db: FakeDatabase, argv: list[str], says: str
) -> None:
    """Usage errors name the problem, and nothing is opened."""
    with pytest.raises(AccountsUsageError, match=f"SIM_ACCOUNTS_USAGE: {says}"):
        main(argv)
    assert fake_db.dsns == []


def test_add_without_an_id_and_a_name_is_refused_and_still_closes(fake_db: FakeDatabase) -> None:
    """The connection opened for it is closed on the way out."""
    with pytest.raises(AccountsUsageError, match="add needs --id ID and --name NAME"):
        main(["add", "--database-env", DSN_VARIABLE, "--name", "orion"])
    assert fake_db.closed == 1
