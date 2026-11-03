"""Shared snapshot evidence and query preparation for study and diagnostic extraction."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import NamedTuple

from platform_core.json_utils import dump_json_str
from platform_core.stage_manifest import StagedFile

from . import _test_hooks as hooks
from .crsp_extract import crsp_extract_from_records, encode_crsp_extract
from .crsp_extraction_records import CRSP_RELATIONS, CrspLimits
from .crsp_protocol import CrspProtocol
from .crsp_queries import CrspScope, CrspSource, crsp_driver_sql, crsp_relation_sql
from .encrypted_storage import staged_file_record

#: psql without startup files or password prompts, stopping at the first error.
PSQL = ("psql", "-X", "-w", "-q", "-v", "ON_ERROR_STOP=1", "-f", "extract.sql")


def _snapshot(root: Path) -> tuple[str, str]:
    """Require identical snapshot readings from the transaction's first and last statements.

    Returns:
        The snapshot and the transaction start time, as the server printed them.

    Raises:
        ValueError: CRSP-011 when the readings differ or are malformed.
    """
    readings = [
        list(csv.reader(io.StringIO((root / name).read_text("utf-8"))))
        for name in ("snapshot-first.csv", "snapshot-last.csv")
    ]
    if readings[0] != readings[1] or len(readings[0]) != 1 or len(readings[0][0]) != 2:
        raise ValueError("[CRSP-011] all relations must come from one database snapshot")
    return readings[0][0][0], readings[0][0][1]


def seal_crsp_manifest(directory: Path, extracted_at: str, limits: CrspLimits) -> StagedFile:
    """Write the existing four-relation manifest after bounding the plaintext.

    Raises:
        ValueError: CRSP-012 when the CSVs exceed the partition budget, which is
            checked before the whole-file reader loads them.
    """
    records = tuple(
        staged_file_record(directory / (name + suffix))
        for name in CRSP_RELATIONS
        for suffix in (".csv", ".sql")
    )
    plaintext = sum(item["size_bytes"] for item in records if item["name"].endswith(".csv"))
    if plaintext > limits["max_partition_bytes"]:
        raise ValueError("[CRSP-012] partition plaintext exceeds its byte budget")
    manifest = crsp_extract_from_records(
        extracted_at, staged_file_record(directory / "protocol.json"), records
    )
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(encode_crsp_extract(manifest)))
    return staged_file_record(directory / "manifest.json")


def prepare_crsp_queries(
    root: Path,
    protocol_bytes: bytes,
    protocol: CrspProtocol,
    scopes: tuple[CrspScope, ...],
    source: CrspSource,
    rows: int,
    *,
    securities: tuple[int, ...] | None,
) -> None:
    """Write each partition's exact protocol copy and relation queries."""
    for scope in scopes:
        directory = root / scope.partition
        directory.mkdir(mode=0o700)
        (directory / "protocol.json").write_bytes(protocol_bytes)
        queries = crsp_relation_sql(protocol, scope, source, rows, securities=securities)
        for name, sql in queries.items():
            (directory / (name + ".sql")).write_text(sql, encoding="utf-8")


class CrspSnapshot(NamedTuple):
    """Internal evidence from the single completed read-only transaction."""

    snapshot: str
    transaction_started_at: str
    completed_at: str
    driver_sha256: str


def execute_crsp_snapshot(
    root: Path, limits: CrspLimits, scopes: tuple[CrspScope, ...], relations: tuple[str, ...]
) -> CrspSnapshot:
    """Run prepared queries with the existing bounded client and verify snapshot equality.

    Args:
        root: Private working directory containing prepared partition directories.
        limits: Frozen wall-time and per-file byte limits.
        scopes: Exact partitions rendered by the shared query owner.
        relations: Exact study or corporate-action roles rendered by that owner.

    Returns:
        Snapshot, server start time, client completion time and exact driver digest.

    Raises:
        ValueError: CRSP-011 for a failed client or inconsistent snapshots.
        TimeoutExpired: Client exceeds the frozen wall deadline.
        OSError: Client startup or private file access fails.
    """
    driver = crsp_driver_sql(
        limits["seconds"], partitions=tuple(s.partition for s in scopes), relations=relations
    )
    (root / "extract.sql").write_text(driver, encoding="utf-8")
    code = hooks.bounded_runner(
        PSQL, root, seconds=limits["seconds"], max_file_bytes=limits["max_relation_bytes"]
    )
    if code != 0:
        raise ValueError(f"[CRSP-011] psql exited {code}; no bundle was published")
    completed = hooks.clock()
    snapshot, transaction = _snapshot(root)
    return CrspSnapshot(
        snapshot, transaction, completed, staged_file_record(root / "extract.sql")["sha256"]
    )


__all__ = [
    "PSQL",
    "CrspSnapshot",
    "execute_crsp_snapshot",
    "prepare_crsp_queries",
    "seal_crsp_manifest",
]
