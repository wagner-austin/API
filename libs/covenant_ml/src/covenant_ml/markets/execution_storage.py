"""Durable execution evidence, exclusive batch ownership and crash-released locks."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import closing, contextmanager
from hashlib import sha256
from os import fsync
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str, require_str

from .validation import read_document


def persist_evidence(path: Path, document: JSONObject) -> str:
    """Exclusively write and flush exact JSON bytes, returning their digest.

    Args:
        path: New evidence path whose parent exists.
        document: Validated JSON evidence.

    Returns:
        SHA-256 of the exact bytes flushed to disk.

    Raises:
        OSError: Exclusive creation or durable persistence failed.
    """
    payload = dump_json_str(document).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        fsync(stream.fileno())
    return sha256(payload).hexdigest()


def initialize_batch_storage(root: Path, predecessor: Path) -> None:
    """Create exclusive batch storage and atomically snapshot the predecessor journal.

    Args:
        root: New batch directory.
        predecessor: Validated existing journal to copy through SQLite backup.

    Raises:
        OSError: The directory or database cannot be created exclusively.
        DatabaseError: Source backup or lock database initialization fails.
    """
    root.mkdir()
    (root / "attempts").mkdir()
    target = root / "execution.sqlite"
    target.touch(exist_ok=False)
    with (
        closing(
            sqlite3.connect(predecessor.resolve().as_uri() + "?mode=ro", uri=True, timeout=0)
        ) as source,
        closing(sqlite3.connect(target, timeout=0)) as destination,
    ):
        source.backup(destination)
    lock_path = root / "operation.sqlite"
    lock_path.touch(exist_ok=False)
    with closing(sqlite3.connect(lock_path, timeout=0)) as connection, connection:
        connection.execute("CREATE TABLE ownership (mode TEXT NOT NULL) STRICT")
        connection.execute("INSERT INTO ownership VALUES ('demo_execution_batch')")


@contextmanager
def batch_operation(root: Path) -> Generator[None, None, None]:
    """Serialize account operations with a real lock released on process exit.

    Args:
        root: Initialized batch directory.

    Yields:
        None while this process owns the batch operation lock.

    Raises:
        OperationalError: Another process owns the lock or its database is absent.
        ValueError: DEMO-BATCH-008 if the lock database identity changed.
    """
    path = root / "operation.sqlite"
    with (
        closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=0)
        ) as connection,
        connection,
    ):
        connection.execute("BEGIN IMMEDIATE")
        rows: list[tuple[str]] = connection.execute("SELECT mode FROM ownership").fetchall()
        if rows != [("demo_execution_batch",)]:
            raise ValueError("[DEMO-BATCH-008] operation lock identity changed")
        yield


def claim_batch_owner(guard_root: Path, root: Path, fingerprint: str) -> None:
    """Permanently bind one credential identity to exactly one batch root and plan.

    Args:
        guard_root: Shared account-ownership directory, independent of batch roots.
        root: Fully prepared batch directory.
        fingerprint: Loaded signer identity.

    Raises:
        FileExistsError: This credential already owns a batch; no second batch starts.
        OSError: Durable ownership evidence cannot be stored.
    """
    guard_root.mkdir(parents=True, exist_ok=True)
    persist_evidence(
        guard_root / (fingerprint + ".json"),
        {
            "root": str(root.resolve()),
            "credential_fingerprint": fingerprint,
            "plan_sha256": sha256((root / "plan.json").read_bytes()).hexdigest(),
        },
    )


def require_batch_owner(guard_root: Path, root: Path, fingerprint: str) -> None:
    """Reject copied roots, changed plans and a missing account ownership record.

    Args:
        guard_root: Shared credential-ownership directory.
        root: Existing batch root.
        fingerprint: Loaded signer identity.

    Raises:
        ValueError: DEMO-BATCH-005 for changed ownership or plan bytes.
        OSError: Required ownership evidence is missing.
    """
    document = read_document(guard_root / (fingerprint + ".json"))
    if (
        require_str(document, "root") != str(root.resolve())
        or require_str(document, "credential_fingerprint") != fingerprint
        or require_str(document, "plan_sha256")
        != sha256((root / "plan.json").read_bytes()).hexdigest()
    ):
        raise ValueError("[DEMO-BATCH-005] batch ownership or immutable plan changed")
