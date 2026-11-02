"""Refresh an authenticated diagnostic cohort in private storage and encrypt its differences."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    require_int,
)
from platform_core.stage_manifest import encode_staged_file

from ..evaluation.evidence import read_evidence, require_private_directory
from . import _test_hooks as hooks
from .crsp_cohort_records import (
    CrspCohort,
    CrspCohortPlan,
    decode_crsp_cohort,
    decode_crsp_cohort_plan,
    encode_crsp_cohort_plan,
)
from .crsp_comparison import compare_crsp_rows
from .crsp_extract import CrspSources, encode_crsp_extract, load_crsp_sources
from .crsp_extraction_records import (
    CRSP_RELATIONS,
    CrspLimits,
    decode_crsp_limits,
    encode_crsp_limits,
)
from .crsp_partition import check_crsp_partition
from .crsp_queries import crsp_scopes, synthetic_source, vendor_source
from .crsp_refresh_records import CrspRefresh, decode_crsp_refresh, encode_crsp_refresh
from .crsp_snapshot import (
    execute_crsp_snapshot,
    prepare_crsp_queries,
    seal_crsp_manifest,
)
from .encrypted_storage import decrypted_bundle, encrypt_bundle, staged_file_record
from .research_keys import EvaluatorKeys


def _cohort_sources(manifest: Path, plan: CrspCohortPlan) -> tuple[CrspCohort, CrspSources]:
    """Open the pinned diagnostic through the canonical reader in private storage only."""
    cohort = decode_crsp_cohort(load_json_str(manifest.read_text("utf-8")))
    if cohort["plan"] != plan:
        raise ValueError("CRSP_REFRESH_LINEAGE: cohort differs from the independently pinned plan")
    # This disposable view is never published as the diagnostic's manifest.
    view = manifest.parent / "manifest.json"
    with view.open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(encode_crsp_extract(cohort["subset"])))
    sources = load_crsp_sources(view)
    check_crsp_partition(sources, crsp_scopes(sources.protocol)[0], plan["max_source_rows"])
    return cohort, sources


def _differences(old: CrspSources, fresh: CrspSources) -> tuple[JSONObject, JSONObject]:
    """Return private row evidence separately from public aggregate counts."""
    details: JSONObject = {}
    counts: JSONObject = {}
    for role, before, after in zip(
        CRSP_RELATIONS,
        (old.membership, old.returns, old.calendar, old.delists),
        (fresh.membership, fresh.returns, fresh.calendar, fresh.delists),
        strict=True,
    ):
        delta = compare_crsp_rows(before, after, role)
        details[role] = {
            "unchanged": delta.unchanged,
            "added": list(delta.added),
            "removed": list(delta.removed),
            "changed": [{"old": left, "fresh": right} for left, right in delta.changed],
        }
        counts[role] = {
            "old": len(before),
            "fresh": len(after),
            "unchanged": delta.unchanged,
            "added": len(delta.added),
            "removed": len(delta.removed),
            "changed": len(delta.changed),
        }
    return details, counts


def _refresh(
    root: Path,
    manifest: Path,
    cohort: CrspCohort,
    old: CrspSources,
    cohort_digest: str,
    limits: CrspLimits,
    setup: Path | None,
) -> tuple[CrspRefresh, JSONObject]:
    """Capture a bounded snapshot and preserve exact private source and change evidence."""
    started = hooks.clock()
    protocol = old.protocol
    scope = crsp_scopes(protocol)[0]
    securities = tuple(sorted({require_int(row, "permno") for row in old.returns}))
    setup_bytes = None if setup is None else setup.read_bytes()
    source = (
        vendor_source(protocol)
        if setup_bytes is None
        else synthetic_source(setup_bytes.decode("utf-8"))
    )
    protocol_file = old.manifest["protocol"]
    protocol_bytes = read_evidence(manifest.parent, protocol_file["name"], protocol_file["sha256"])
    prepare_crsp_queries(
        root, protocol_bytes, protocol, (scope,), source, limits["max_rows"], securities=securities
    )
    snapshot = execute_crsp_snapshot(root, limits, (scope,))
    directory = root / scope.partition
    seal_crsp_manifest(directory, started, limits)
    fresh = load_crsp_sources(directory / "manifest.json")
    check_crsp_partition(fresh, scope, limits["max_rows"])
    allowed = set(securities)
    if any(
        require_int(row, "permno") not in allowed
        for rows in (fresh.membership, fresh.returns, fresh.delists)
        for row in rows
    ):
        raise ValueError("CRSP_REFRESH_SCOPE: fresh observations contain another security")
    details, counts = _differences(old, fresh)
    with (directory / "differences.json").open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(details))
    for name in ("extract.sql", "snapshot-first.csv", "snapshot-last.csv"):
        (root / name).rename(directory / name)
    if setup_bytes is not None:
        (directory / "setup.sql").write_bytes(setup_bytes)
    record = decode_crsp_refresh(
        encode_crsp_refresh(
            CrspRefresh(
                kind="crsp-diagnostic-refresh-v1",
                cohort_bundle_sha256=cohort_digest,
                cohort=cohort,
                fresh=fresh.manifest,
                source=source.name,
                setup_sha256=None if setup_bytes is None else sha256(setup_bytes).hexdigest(),
                completed_at=snapshot.completed_at,
                snapshot=snapshot.snapshot,
                transaction_started_at=snapshot.transaction_started_at,
                driver=staged_file_record(directory / "extract.sql"),
                differences=staged_file_record(directory / "differences.json"),
                limits=limits,
            )
        )
    )
    with (directory / "comparison.json").open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(encode_crsp_refresh(record)))
    return record, counts


def compare_crsp_vintages(
    bundle: Path,
    bundle_sha256: str,
    plan: CrspCohortPlan,
    keys: EvaluatorKeys,
    *,
    scratch: Path,
    destination: Path,
    limits: CrspLimits,
    setup: Path | None,
) -> JSONObject:
    """Query only the pinned validation cohort and publish an encrypted diagnostic.

    Args:
        bundle: Authenticated frozen-cohort bundle, never the full study bundle.
        bundle_sha256: Independently pinned cohort bundle identity.
        plan: Independently pinned original bundle, protocol and source-path audit.
        keys: Trusted evaluator custody; only train-validation is acquired.
        scratch: Private scheduler storage for decrypted and queried evidence.
        destination: New exclusive directory receiving ciphertext and bundle metadata.
        limits: Explicit query row, file, total-byte and wall-time bounds.
        setup: Synthetic relation definitions for a rehearsal, or None for vendor data.

    Returns:
        Identities, extraction timestamps, row counts and normal-unwind cleanup.
        Identifiers, observation dates, values, exact SQL and differences stay encrypted.

    Raises:
        ValueError: Invalid lineage, partition, scope, rows, limits or snapshot.
        FileExistsError: Destination exists; prior attempts are not replaced.
        TimeoutExpired: Database client exceeded its time budget.
        OSError: Storage or client startup fails; private directories unwind.
        Exception: Authenticated cryptography refuses an input or recipient.
    """
    plan = decode_crsp_cohort_plan(encode_crsp_cohort_plan(plan))
    limits = decode_crsp_limits(encode_crsp_limits(limits))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("CRSP_REFRESH_OUTPUT: refusing to replace a comparison")
    with (
        decrypted_bundle(
            bundle,
            bundle_sha256,
            partition="train-validation",
            identity=keys.acquire("train-validation"),
            scratch=scratch,
            max_plaintext_bytes=min(plan["max_cohort_bytes"], keys.access["max_plaintext_bytes"]),
        ) as manifest,
        TemporaryDirectory(prefix="research-plaintext-", dir=scratch) as temporary,
    ):
        root = Path(temporary)
        require_private_directory(root)
        cohort, old = _cohort_sources(manifest, plan)
        record, counts = _refresh(root, manifest, cohort, old, bundle_sha256, limits, setup)
        directory = root / "train-validation"
        files = tuple(staged_file_record(path) for path in sorted(directory.iterdir()))
        if sum(item["size_bytes"] for item in files) > limits["max_partition_bytes"]:
            raise ValueError("CRSP_REFRESH_LIMIT: comparison evidence exceeds its byte budget")
        encrypt_bundle(
            directory,
            destination,
            files=files,
            manifest=staged_file_record(directory / "comparison.json"),
            recipient=keys.access["validation_recipient"],
            partition="train-validation",
        )
    return {
        "kind": "CRSP_DIAGNOSTIC_REFRESH",
        "cohort_bundle_sha256": bundle_sha256,
        "bundle": encode_staged_file(staged_file_record(destination / "bundle.json")),
        "source": record["source"],
        "started_at": record["fresh"]["extracted_at"],
        "completed_at": record["completed_at"],
        "rows": counts,
        "plaintext_removed": not root.exists() and not manifest.parent.exists(),
    }


__all__ = ["compare_crsp_vintages"]
