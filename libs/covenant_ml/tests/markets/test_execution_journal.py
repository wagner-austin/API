"""Real SQLite persistence, corruption, transaction rollback and writer contention."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, dump_json_str

from covenant_ml.markets.execution_journal import (
    append_execution_event,
    initialize_execution_journal,
    read_execution_journal,
)
from covenant_ml.markets.execution_records import encode_intent
from covenant_ml.markets.execution_state import execution_report

from ._execution_fixtures import empty_state, intent
from .test_execution_policy import receipt_event, settle_event


def test_restart_preserves_reservation_and_reconciliation(tmp_path: Path) -> None:
    path = tmp_path / "risk.sqlite"
    assert initialize_execution_journal(path) == empty_state()
    row = intent()
    event: JSONObject = {"action": "reserve", "observed_ms": 2000, "data": encode_intent(row)}
    reserved = append_execution_event(path, event)
    assert read_execution_journal(path) == reserved
    assert execution_report(reserved)["exposure"] == 10000
    with pytest.raises(ValueError, match="EXECUTION-004"):
        append_execution_event(path, event)
    assert read_execution_journal(path) == reserved
    filled = append_execution_event(path, receipt_event(row))
    assert read_execution_journal(path) == filled
    final = append_execution_event(path, settle_event(row))
    assert read_execution_journal(path) == final
    assert execution_report(final)["gross_realized_loss"] == 10000
    with pytest.raises(FileExistsError):
        initialize_execution_journal(path)
    assert read_execution_journal(path) == final
    with closing(sqlite3.connect(path)) as conn:
        rows: list[tuple[int]] = conn.execute(
            "SELECT sequence FROM events ORDER BY sequence"
        ).fetchall()
    assert rows == [(1,), (2,), (3,)]


def test_missing_database_is_never_created_by_read_or_write(tmp_path: Path) -> None:
    path = tmp_path / "absent.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        read_execution_journal(path)
    with pytest.raises(sqlite3.OperationalError):
        append_execution_event(path, {"action": "halt", "reason": "stop", "observed_ms": 2000})
    assert path.exists() is False


def test_real_writer_lock_fails_without_overbooking(tmp_path: Path) -> None:
    path = tmp_path / "risk.sqlite"
    initialize_execution_journal(path)
    event: JSONObject = {"action": "reserve", "observed_ms": 2000, "data": encode_intent(intent())}
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            append_execution_event(path, event)
        assert read_execution_journal(path) == empty_state()
    reserved = append_execution_event(path, event)
    assert execution_report(reserved)["exposure"] == 10000


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE metadata SET payload = '{}'",
        "DELETE FROM metadata",
        "INSERT INTO metadata SELECT payload FROM metadata",
        "UPDATE events SET digest = 'corrupt'",
        "UPDATE events SET sequence = 2",
        "UPDATE events SET payload = '{}'",
    ],
)
def test_corruption_is_refused_before_new_reservation(tmp_path: Path, statement: str) -> None:
    path = tmp_path / "risk.sqlite"
    initialize_execution_journal(path)
    append_execution_event(
        path, {"action": "reserve", "observed_ms": 2000, "data": encode_intent(intent())}
    )
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(statement)
    with pytest.raises(ValueError, match="EXECUTION-008"):
        read_execution_journal(path)
    with pytest.raises(ValueError, match="EXECUTION-008"):
        append_execution_event(path, receipt_event(intent()))


def test_readback_detects_a_real_database_trigger_and_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "risk.sqlite"
    initialize_execution_journal(path)
    replacement = dump_json_str({"action": "halt", "observed_ms": 2000, "reason": "changed"})
    with closing(sqlite3.connect(path)) as conn, conn:
        rows: list[tuple[str]] = conn.execute("SELECT payload FROM metadata").fetchall()
        seed = sha256(rows[0][0].encode()).hexdigest()
        digest = sha256((seed + replacement).encode()).hexdigest()
        conn.execute("CREATE TABLE replacement (payload TEXT, digest TEXT)")
        conn.execute("INSERT INTO replacement VALUES (?, ?)", (replacement, digest))
        conn.execute(
            "CREATE TRIGGER change_insert AFTER INSERT ON events BEGIN "
            "UPDATE events SET payload = (SELECT payload FROM replacement), "
            "digest = (SELECT digest FROM replacement) WHERE sequence = NEW.sequence; END"
        )
    with pytest.raises(ValueError, match="EXECUTION-009"):
        append_execution_event(path, {"action": "halt", "observed_ms": 2000, "reason": "original"})
    assert read_execution_journal(path) == empty_state()
