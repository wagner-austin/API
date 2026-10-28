"""Real SQLite replay, contention, corruption and transactional rollback."""

import sqlite3
from contextlib import closing
from hashlib import sha256
from pathlib import Path

import pytest

from platform_core.event_journal import EventJournal
from platform_core.json_utils import JSONObject, dump_json_str, require_int


def _apply(state: int, event: JSONObject) -> int:
    value = require_int(event, "value")
    if value < 0:
        raise ValueError("negative event")
    return state + value


def _journal() -> EventJournal[int]:
    return EventJournal({"schema": 1}, lambda: 0, _apply)


def test_restart_lock_and_invalid_transition(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite"
    journal = _journal()
    assert journal.initialize(path) == 0
    assert journal.transact(path, lambda state: {"value": state + 2}) == 2
    assert _journal().read(path) == 2
    with pytest.raises(FileExistsError):
        journal.initialize(path)
    with pytest.raises(ValueError, match="negative event"):
        journal.transact(path, lambda _: {"value": -1})
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            journal.transact(path, lambda _: {"value": 3})
        assert journal.read(path) == 2
    assert journal.transact(path, lambda state: {"value": state}) == 4


def test_absent_database_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        _journal().read(path)
    with pytest.raises(sqlite3.OperationalError):
        _journal().transact(path, lambda _: {"value": 1})
    assert path.exists() is False


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM metadata",
        "INSERT INTO metadata SELECT payload FROM metadata",
        "UPDATE metadata SET payload = '{}'",
        "UPDATE events SET sequence = 2",
        "UPDATE events SET digest = 'bad'",
        "UPDATE events SET payload = '{}'",
    ],
)
def test_corruption(tmp_path: Path, statement: str) -> None:
    path = tmp_path / "corrupt.sqlite"
    journal = _journal()
    journal.initialize(path)
    journal.transact(path, lambda _: {"value": 1})
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(statement)
    with pytest.raises(ValueError, match="JOURNAL-001"):
        journal.read(path)
    with pytest.raises(ValueError, match="JOURNAL-001"):
        journal.transact(path, lambda _: {"value": 2})


def test_trigger_readback_failure_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "trigger.sqlite"
    journal = _journal()
    journal.initialize(path)
    payload = dump_json_str({"value": 9})
    seed = sha256(dump_json_str({"schema": 1}).encode()).hexdigest()
    digest = sha256((seed + payload).encode()).hexdigest()
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE replacement (payload TEXT, digest TEXT)")
        conn.execute("INSERT INTO replacement VALUES (?, ?)", (payload, digest))
        conn.execute(
            "CREATE TRIGGER replace_insert AFTER INSERT ON events BEGIN "
            "UPDATE events SET payload = (SELECT payload FROM replacement), "
            "digest = (SELECT digest FROM replacement) WHERE sequence = NEW.sequence; END"
        )
    with pytest.raises(ValueError, match="JOURNAL-002"):
        journal.transact(path, lambda _: {"value": 1})
    assert journal.read(path) == 0
