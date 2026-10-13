"""Transactional, hash-chained SQLite journal for local execution rehearsals."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str, load_json_str, narrow_json_to_dict

from .execution_policy import apply_execution_event
from .execution_records import CAPITAL, EXPOSURE_LIMIT, LOSS_LIMIT, ORDER_LIMIT
from .execution_state import ExecutionState


def _metadata() -> str:
    """Encode the immutable journal format and experiment ceilings.

    Returns:
        Canonical JSON text identifying this local rehearsal format.
    """
    return dump_json_str(
        {
            "schema": 1,
            "mode": "local_rehearsal",
            "capital": CAPITAL,
            "order_limit": ORDER_LIMIT,
            "exposure_limit": EXPOSURE_LIMIT,
            "loss_limit": LOSS_LIMIT,
        }
    )


def _replay(connection: sqlite3.Connection) -> tuple[ExecutionState, str, int]:
    """Validate metadata, continuity and every transition in committed history.

    Args:
        connection: Open SQLite connection inside a transaction.

    Returns:
        ExecutionState, terminal hash and last sequence number.

    Raises:
        ValueError: EXECUTION-008 for metadata, sequence or digest corruption.
        DatabaseError: The database or expected schema is invalid.
    """
    metadata: list[tuple[str]] = connection.execute("SELECT payload FROM metadata").fetchall()
    if metadata != [(_metadata(),)]:
        raise ValueError("[EXECUTION-008] execution journal metadata or fixed limits changed")
    previous = sha256(_metadata().encode()).hexdigest()
    state = ExecutionState(orders=(), halt_reason=None, observed_ms=0)
    rows: list[tuple[int, str, str]] = connection.execute(
        "SELECT sequence, payload, digest FROM events ORDER BY sequence"
    ).fetchall()
    sequence = 0
    for position, payload, digest in rows:
        sequence += 1
        expected = sha256((previous + payload).encode()).hexdigest()
        if position != sequence or digest != expected:
            raise ValueError("[EXECUTION-008] execution journal sequence or hash chain changed")
        state = apply_execution_event(state, narrow_json_to_dict(load_json_str(payload)))
        previous = digest
    return state, previous, sequence


def initialize_execution_journal(path: Path) -> ExecutionState:
    """Create a new rehearsal journal exclusively and verify its initial state.

    Args:
        path: New SQLite file; its parent directory must exist.

    Returns:
        Empty ExecutionState read back from the committed database.

    Raises:
        FileExistsError: The path already exists; balances are never reset.
        OSError: The file cannot be created.
        DatabaseError: Schema initialization or verification failed.
    """
    path.touch(exist_ok=False)
    with closing(sqlite3.connect(path, timeout=0)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE metadata (payload TEXT NOT NULL) STRICT")
        connection.execute(
            "CREATE TABLE events (sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL, "
            "digest TEXT NOT NULL) STRICT"
        )
        connection.execute("INSERT INTO metadata VALUES (?)", (_metadata(),))
    return read_execution_journal(path)


def read_execution_journal(path: Path) -> ExecutionState:
    """Read an existing journal without creating an empty replacement.

    Args:
        path: Required existing SQLite file.

    Returns:
        ExecutionState rebuilt from validated events.

    Raises:
        OperationalError: The file is missing, inaccessible or locked.
        ValueError: Persisted evidence or policy is inconsistent.
    """
    with (
        closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0)) as conn,
        conn,
    ):
        conn.execute("BEGIN")
        state, _, _ = _replay(conn)
    return state


def append_execution_event(path: Path, event: JSONObject) -> ExecutionState:
    """Atomically validate, reserve or reconcile, append and verify a transition.

    Args:
        path: Existing rehearsal SQLite file.
        event: JSON event, including its actual local observation timestamp.

    Returns:
        ExecutionState after the committed transition.

    Raises:
        OperationalError: Missing journal or another writer owns the lock.
        ValueError: Event violates policy or journal integrity.
    """
    with (
        closing(sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=0)) as conn,
        conn,
    ):
        conn.execute("BEGIN IMMEDIATE")
        state, previous, sequence = _replay(conn)
        updated = apply_execution_event(state, event)
        payload = dump_json_str(event)
        digest = sha256((previous + payload).encode()).hexdigest()
        conn.execute("INSERT INTO events VALUES (?, ?, ?)", (sequence + 1, payload, digest))
        verified, _, _ = _replay(conn)
        if verified != updated:
            raise ValueError("[EXECUTION-009] journal readback differs from validated transition")
    return updated
