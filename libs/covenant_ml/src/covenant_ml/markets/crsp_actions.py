"""Capture encrypted corporate-action evidence without changing any study outcome."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str, require_int, require_str

from ..evaluation.evidence import read_evidence
from .crsp_action_records import (
    CrspActionEvidence,
    decode_crsp_action_evidence,
    encode_crsp_action_evidence,
)
from .crsp_action_schema import CRSP_ACTIONS
from .crsp_cohort_records import CrspCohort
from .crsp_extract import CrspSources
from .crsp_extraction_records import CrspLimits
from .crsp_queries import CrspScope, crsp_action_sql, crsp_scopes, synthetic_source, vendor_source
from .crsp_rows import crsp_row_key, decode_crsp_rows
from .crsp_snapshot import execute_crsp_snapshot
from .encrypted_storage import staged_file_record


def check_crsp_action_rows(
    directory: Path,
    scope: CrspScope,
    securities: tuple[int, ...],
    started_at: str,
    limits: CrspLimits,
) -> JSONObject:
    """Validate actual returned bytes before publishing any diagnostic evidence.

    All dated observations must fit the partition, even if a database or a
    faulty query supplies extra rows. Details must have a corresponding event.
    Missing fields remain missing; counts cannot establish investment value.

    Raises:
        ValueError: CRSP_ACTION_LIMIT/SCOPE/KEY for excessive or inconsistent
            evidence; canonical CSV reader errors for malformed source rows.
    """
    sizes = [(directory / (item.role + ".csv")).stat().st_size for item in CRSP_ACTIONS]
    if max(sizes) > limits["max_relation_bytes"] or sum(sizes) > limits["max_partition_bytes"]:
        raise ValueError("CRSP_ACTION_LIMIT: source bytes exceed diagnostic bounds")
    counts: JSONObject = {}
    keys: dict[str, set[tuple[str, ...]]] = {}
    allowed = set(securities)
    for relation in CRSP_ACTIONS:
        rows = decode_crsp_rows(
            (directory / (relation.role + ".csv")).read_bytes(),
            relation.role,
            relation.columns,
            started_at,
        )
        if len(rows) > limits["max_rows"]:
            raise ValueError("CRSP_ACTION_LIMIT: source rows exceed diagnostic bounds")
        for row in rows:
            if (
                require_int(row, "permno") not in allowed
                or require_str(row, relation.dates[0]) < scope.history_start
                or any(
                    row[field] is not None and require_str(row, field) > scope.observation_end
                    for field in relation.dates
                )
            ):
                raise ValueError("CRSP_ACTION_SCOPE: observations exceed the pinned cohort scope")
        keys[relation.role] = {crsp_row_key(row, relation.role) for row in rows}
        counts[relation.role] = len(rows)
    for events, details in (
        ("delist-events", "delist-details"),
        ("distribution-events", "distributions"),
    ):
        if not keys[details] <= keys[events]:
            raise ValueError("CRSP_ACTION_KEY: detail has no corresponding event identity")
        count_name = events.removesuffix("-events") + "-without-details"
        counts[count_name] = len(keys[events] - keys[details])
    return counts


def write_crsp_actions(
    root: Path,
    manifest: Path,
    cohort: CrspCohort,
    sources: CrspSources,
    cohort_digest: str,
    limits: CrspLimits,
    setup: Path | None,
    started: str,
) -> tuple[Path, JSONObject, str]:
    """Capture additional fields inside the shared diagnostic's private lifecycle.

    The caller owns authenticated cohort opening, resource bounds, encryption
    and cleanup. This writer owns only the fixed projections and their evidence.

    Raises:
        ValueError: Query, source, partition or snapshot validation fails.
        OSError: Private source files or database process cannot be accessed.
    """
    scope = crsp_scopes(sources.protocol)[0]
    securities = tuple(sorted({require_int(row, "permno") for row in sources.returns}))
    definitions = None if setup is None else setup.read_bytes()
    source = (
        vendor_source(sources.protocol)
        if definitions is None
        else synthetic_source(definitions.decode("utf-8"))
    )
    directory = root / scope.partition
    directory.mkdir(mode=0o700)
    protocol = cohort["subset"]["protocol"]
    (directory / "protocol.json").write_bytes(
        read_evidence(manifest.parent, protocol["name"], protocol["sha256"])
    )
    for role, statement in crsp_action_sql(
        sources.protocol, scope, source, limits["max_rows"], securities
    ).items():
        (directory / (role + ".sql")).write_text(statement, encoding="utf-8")
    snapshot = execute_crsp_snapshot(
        root, limits, (scope,), tuple(item.role for item in CRSP_ACTIONS)
    )
    counts = check_crsp_action_rows(directory, scope, securities, started, limits)
    for name in ("extract.sql", "snapshot-first.csv", "snapshot-last.csv"):
        (root / name).rename(directory / name)
    if definitions is not None:
        (directory / "setup.sql").write_bytes(definitions)
    record = decode_crsp_action_evidence(
        encode_crsp_action_evidence(
            CrspActionEvidence(
                kind="crsp-corporate-action-diagnostic-v1",
                cohort_bundle_sha256=cohort_digest,
                cohort=cohort,
                source=source.name,
                setup_sha256=None if definitions is None else sha256(definitions).hexdigest(),
                started_at=started,
                completed_at=snapshot.completed_at,
                snapshot=snapshot.snapshot,
                transaction_started_at=snapshot.transaction_started_at,
                files=tuple(staged_file_record(path) for path in sorted(directory.iterdir())),
                limits=limits,
            )
        )
    )
    evidence = directory / "actions.json"
    with evidence.open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(encode_crsp_action_evidence(record)))
    return evidence, counts, snapshot.completed_at


__all__ = ["check_crsp_action_rows", "write_crsp_actions"]
