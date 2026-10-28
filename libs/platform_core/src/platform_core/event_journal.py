"""Transactional hash-chain persistence lifted from execution rehearsals.

Callers own policy and state. Hash chaining detects damage, not malicious
rewriting by someone with database write access.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from typing import Generic, TypeVar

from .json_utils import JSONObject, dump_json_str, load_json_str, narrow_json_to_dict

State = TypeVar("State")


class EventJournal(Generic[State]):
    """Serialize domain decisions against the latest committed state."""

    def __init__(
        self,
        metadata: JSONObject,
        initial: Callable[[], State],
        apply: Callable[[State, JSONObject], State],
    ) -> None:
        """Bind exact format metadata and a pure domain reducer.

        Args:
            metadata: Immutable format and policy identity.
            initial: Fresh empty state factory for each replay.
            apply: Pure transition validator; invalid events raise.
        """
        self._metadata = dump_json_str(metadata)
        self._initial = initial
        self._apply = apply

    def _replay(self, connection: sqlite3.Connection) -> tuple[State, str, int]:
        """Validate metadata, continuity and transitions in a transaction.

        Args:
            connection: Connection with an open transaction.

        Returns:
            Rebuilt state, terminal digest and sequence number.

        Raises:
            ValueError: JOURNAL-001 for changed metadata or event chain.
        """
        metadata: list[tuple[str]] = connection.execute("SELECT payload FROM metadata").fetchall()
        if metadata != [(self._metadata,)]:
            raise ValueError("[JOURNAL-001] journal metadata or policy changed")
        previous = sha256(self._metadata.encode()).hexdigest()
        state = self._initial()
        rows: list[tuple[int, str, str]] = connection.execute(
            "SELECT sequence, payload, digest FROM events ORDER BY sequence"
        ).fetchall()
        sequence = 0
        for position, payload, digest in rows:
            sequence += 1
            expected = sha256((previous + payload).encode()).hexdigest()
            if position != sequence or digest != expected:
                raise ValueError("[JOURNAL-001] journal sequence or hash chain changed")
            state = self._apply(state, narrow_json_to_dict(load_json_str(payload)))
            previous = digest
        return state, previous, sequence

    def initialize(self, path: Path) -> State:
        """Create exclusively and verify its initial state.

        Args:
            path: New SQLite file under an existing directory.

        Returns:
            Empty state read back from disk.

        Raises:
            FileExistsError: Journal exists; never reset it.
            OSError: File creation fails.
            sqlite3.DatabaseError: Initialization or verification fails.
        """
        path.touch(exist_ok=False, mode=0o600)
        with closing(sqlite3.connect(path, timeout=0)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE metadata (payload TEXT NOT NULL) STRICT")
            connection.execute(
                "CREATE TABLE events (sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL, "
                "digest TEXT NOT NULL) STRICT"
            )
            connection.execute("INSERT INTO metadata VALUES (?)", (self._metadata,))
        return self.read(path)

    def read(self, path: Path) -> State:
        """Replay without creating a missing journal.

        Args:
            path: Existing journal.

        Returns:
            Validated domain state.

        Raises:
            sqlite3.OperationalError: Missing, inaccessible or locked database.
            ValueError: Invalid history or changed policy.
        """
        with (
            closing(
                sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0)
            ) as connection,
            connection,
        ):
            connection.execute("BEGIN")
            state, _, _ = self._replay(connection)
        return state

    def transact(self, path: Path, decide: Callable[[State], JSONObject]) -> State:
        """Decide against locked current state, append and verify atomically.

        Args:
            path: Existing journal.
            decide: Pure event producer evaluated under the writer lock.

        Returns:
            State after the committed event.

        Raises:
            sqlite3.OperationalError: Missing journal or competing writer.
            ValueError: Invalid event or JOURNAL-002 for altered readback.
        """
        with (
            closing(
                sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=0)
            ) as connection,
            connection,
        ):
            connection.execute("BEGIN IMMEDIATE")
            state, previous, sequence = self._replay(connection)
            event = decide(state)
            updated = self._apply(state, event)
            payload = dump_json_str(event)
            digest = sha256((previous + payload).encode()).hexdigest()
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?)", (sequence + 1, payload, digest)
            )
            verified, _, _ = self._replay(connection)
            if verified != updated:
                raise ValueError("[JOURNAL-002] journal readback differs from validated transition")
        return updated


__all__ = ["EventJournal"]
