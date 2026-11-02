"""Derive an encrypted diagnostic cohort from authenticated frozen CRSP observations.

The trusted evaluator runs this on HPC3, where the full frozen bundle already
resides. Security identifiers are resolved only inside private job storage.
The output is a derived diagnostic, never a fresh vendor extraction or study
dataset. No final identity is acquired and no economic outcome is evaluated.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_dict,
    require_int,
    require_list,
    require_str,
)
from platform_core.stage_manifest import encode_staged_file

from ..evaluation.evidence import read_evidence, require_private_directory
from . import _test_hooks as hooks
from .campaign_records import require_digest
from .crsp_cohort_records import (
    CrspCohort,
    CrspCohortPlan,
    decode_crsp_cohort,
    decode_crsp_cohort_plan,
    encode_crsp_cohort,
    encode_crsp_cohort_plan,
)
from .crsp_extract import CrspSources, crsp_extract_from_records, load_crsp_sources
from .crsp_extraction_records import CRSP_RELATIONS
from .crsp_partition import check_crsp_partition
from .crsp_queries import crsp_scopes
from .crsp_rows import encode_crsp_rows
from .encrypted_storage import decrypted_bundle, encrypt_bundle, staged_file_record
from .research_keys import EvaluatorKeys
from .validation import require_bounded_int


def _anchors(payload: bytes, plan: CrspCohortPlan) -> tuple[str, ...]:
    """Require every pinned refused case to identify a distinct present entry row."""
    document = narrow_json_to_dict(load_json_str(payload.decode()))
    results = require_list(document, "results")
    if len(results) != 1:
        raise ValueError("CRSP_COHORT_AUDIT: require one complete source-path audit")
    result = narrow_json_to_dict(results[0])
    if (
        require_str(result, "kind") != "licensed-v3-source-path-audit"
        or require_digest(result, "bundle_sha256") != plan["bundle_sha256"]
        or not require_bool(result, "all_paths_accounted")
    ):
        raise ValueError("CRSP_COHORT_AUDIT: audit kind, bundle or accounting differs")
    cases = require_list(result, "refused_cases")
    paths = require_int(result, "eligible_paths")
    if (
        len(cases) != plan["expected_cases"]
        or require_int(result, "refused_paths") != len(cases)
        or require_int(result, "accepted_paths") + len(cases) != paths
    ):
        raise ValueError("CRSP_COHORT_AUDIT: audit case counts differ from the plan")
    ordinals: list[int] = []
    anchors: list[str] = []
    for value in cases:
        case = narrow_json_to_dict(value)
        entry = require_dict(case, "entry")
        period = require_str(case, "period")
        if period not in {"training", "validation"} or not require_bool(entry, "present"):
            raise ValueError("CRSP_COHORT_AUDIT: require a present non-final entry anchor")
        ordinals.append(require_bounded_int(case, "path_ordinal", 1, paths))
        anchors.append(require_digest(entry, "row_sha256"))
    if ordinals != sorted(set(ordinals)) or len(set(anchors)) != len(anchors):
        raise ValueError("CRSP_COHORT_AUDIT: require distinct ordered cases and entry anchors")
    return tuple(anchors)


def _securities(sources: CrspSources, anchors: tuple[str, ...]) -> frozenset[int]:
    """Resolve pinned row hashes in authenticated sources without exposing identifiers."""
    expected = set(anchors)
    matched: dict[str, int] = {}
    for row in sources.returns:
        digest = sha256(dump_json_str(row).encode()).hexdigest()
        if digest in expected:
            matched[digest] = require_int(row, "permno")
    if set(matched) != expected:
        raise ValueError("CRSP_COHORT_ANCHOR: an audited entry row is absent or changed")
    return frozenset(matched.values())


def _write_subset(
    original: Path,
    private: Path,
    sources: CrspSources,
    securities: frozenset[int],
    audit_bytes: bytes,
    plan: CrspCohortPlan,
) -> tuple[CrspCohort, dict[str, int]]:
    """Preserve exact query/protocol evidence and encode the selected source rows.

    The queries describe the original full extraction. The diagnostic envelope
    records that its CSVs are a derived subset, retaining the source identities
    and source extraction time rather than claiming that SQL produced a new vintage.
    """
    source = sources.manifest
    protocol = source["protocol"]
    payloads = {
        "protocol.json": read_evidence(original, protocol["name"], protocol["sha256"]),
        "audit.json": audit_bytes,
    }
    relations = (source["membership"], source["returns"], source["calendar"], source["delists"])
    counts: dict[str, int] = {}
    for name, rows, columns, relation in zip(
        CRSP_RELATIONS,
        (sources.membership, sources.returns, sources.calendar, sources.delists),
        (
            sources.protocol["membership_columns"],
            sources.protocol["returns_columns"],
            sources.protocol["calendar_columns"],
            sources.protocol["delists_columns"],
        ),
        relations,
        strict=True,
    ):
        selected = tuple(
            row for row in rows if name == "calendar" or require_int(row, "permno") in securities
        )
        counts[name] = len(selected)
        payloads[name + ".csv"] = encode_crsp_rows(selected, columns)
        query = relation["query"]
        payloads[name + ".sql"] = read_evidence(original, query["name"], query["sha256"])
    if sum(len(payload) for payload in payloads.values()) > plan["max_cohort_bytes"]:
        raise ValueError("CRSP_COHORT_LIMIT: derived evidence exceeds its byte budget")
    for name, payload in payloads.items():
        with (private / name).open("xb") as stream:
            stream.write(payload)
    records = tuple(
        staged_file_record(private / (name + suffix))
        for name in CRSP_RELATIONS
        for suffix in (".csv", ".sql")
    )
    subset = crsp_extract_from_records(
        source["extracted_at"], staged_file_record(private / "protocol.json"), records
    )
    cohort = decode_crsp_cohort(
        encode_crsp_cohort(
            CrspCohort(
                kind="crsp-diagnostic-cohort-v1",
                derived_at=hooks.clock(),
                plan=plan,
                source=source,
                subset=subset,
            )
        )
    )
    with (private / "cohort.json").open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(encode_crsp_cohort(cohort)))
    return cohort, counts


def derive_crsp_cohort(
    bundle_path: Path,
    audit_path: Path,
    plan: CrspCohortPlan,
    keys: EvaluatorKeys,
    *,
    scratch: Path,
    destination: Path,
) -> JSONObject:
    """Publish an encrypted diagnostic subset and return metadata-only evidence.

    Args:
        bundle_path: Frozen train-validation encrypted bundle, already staged on HPC3.
        audit_path: The pinned metadata-only native source-path audit.
        plan: Independent input hashes and explicit case, row and byte limits.
        keys: Trusted evaluator custody; only train-validation is acquired.
        scratch: Scheduler-owned private job storage, reclaimed by the native launcher.
        destination: New exclusive directory receiving ciphertext and bundle metadata.

    Returns:
        Public counts, identities and normal-unwind cleanup evidence, without
        security identifiers, observation dates, prices, returns or keys.

    Raises:
        ValueError: Changed inputs, inconsistent audit, missing anchor, partition
            violation or excessive plaintext; no diagnostic metadata is published.
        OSError: Existing output or failed I/O; plaintext is removed on unwinding.
        Exception: Authenticated decryption or encryption fails without recovery.
    """
    plan = decode_crsp_cohort_plan(encode_crsp_cohort_plan(plan))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("CRSP_COHORT_OUTPUT: refusing to replace a diagnostic")
    audit_bytes = read_evidence(audit_path.parent, audit_path.name, plan["audit_sha256"])
    anchors = _anchors(audit_bytes, plan)
    with (
        decrypted_bundle(
            bundle_path,
            plan["bundle_sha256"],
            partition="train-validation",
            identity=keys.acquire("train-validation"),
            scratch=scratch,
            max_plaintext_bytes=keys.access["max_plaintext_bytes"],
        ) as manifest,
        TemporaryDirectory(prefix="research-plaintext-", dir=scratch) as temporary,
    ):
        private = Path(temporary)
        require_private_directory(private)
        sources = load_crsp_sources(manifest)
        if sources.manifest["protocol"]["sha256"] != plan["protocol_sha256"]:
            raise ValueError("CRSP_COHORT_PROTOCOL: source protocol differs from the plan")
        check_crsp_partition(sources, crsp_scopes(sources.protocol)[0], plan["max_source_rows"])
        securities = _securities(sources, anchors)
        cohort, counts = _write_subset(
            manifest.parent, private, sources, securities, audit_bytes, plan
        )
        files = tuple(staged_file_record(path) for path in sorted(private.iterdir()))
        if sum(item["size_bytes"] for item in files) > plan["max_cohort_bytes"]:
            raise ValueError("CRSP_COHORT_LIMIT: diagnostic metadata exceeds its byte budget")
        encrypt_bundle(
            private,
            destination,
            files=files,
            manifest=staged_file_record(private / "cohort.json"),
            recipient=keys.access["validation_recipient"],
            partition="train-validation",
        )
    return {
        "kind": "CRSP_DIAGNOSTIC_COHORT",
        "bundle": encode_staged_file(staged_file_record(destination / "bundle.json")),
        "plan": encode_crsp_cohort_plan(plan),
        "derived_at": cohort["derived_at"],
        "cases": len(anchors),
        "securities": len(securities),
        "rows": {name: count for name, count in counts.items()},
        "plaintext_removed": not private.exists() and not manifest.parent.exists(),
        "fresh_vendor_extraction": False,
    }


__all__ = ["derive_crsp_cohort"]
