"""Authenticated lineage for corporate-action evidence that cannot admit a study."""

from __future__ import annotations

from datetime import UTC, datetime

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_list,
    require_str,
)
from platform_core.stage_manifest import StagedFile, decode_staged_file, encode_staged_file
from typing_extensions import ReadOnly, TypedDict

from .campaign_records import require_digest
from .crsp_action_schema import CRSP_ACTIONS
from .crsp_cohort_records import CrspCohort, decode_crsp_cohort, encode_crsp_cohort
from .crsp_extraction_records import CrspLimits, decode_crsp_limits, encode_crsp_limits


class CrspActionEvidence(TypedDict):
    """Private source identities and exact query artifacts, not evaluated returns."""

    kind: ReadOnly[str]
    cohort_bundle_sha256: ReadOnly[str]
    cohort: ReadOnly[CrspCohort]
    source: ReadOnly[str]
    setup_sha256: ReadOnly[str | None]
    started_at: ReadOnly[str]
    completed_at: ReadOnly[str]
    snapshot: ReadOnly[str]
    transaction_started_at: ReadOnly[str]
    files: ReadOnly[tuple[StagedFile, ...]]
    limits: ReadOnly[CrspLimits]


def encode_crsp_action_evidence(value: CrspActionEvidence) -> JSONObject:
    """Encode every diagnostic field through the existing lineage/file codecs."""
    return {
        "kind": value["kind"],
        "cohort_bundle_sha256": value["cohort_bundle_sha256"],
        "cohort": encode_crsp_cohort(value["cohort"]),
        "source": value["source"],
        "setup_sha256": value["setup_sha256"],
        "started_at": value["started_at"],
        "completed_at": value["completed_at"],
        "snapshot": value["snapshot"],
        "transaction_started_at": value["transaction_started_at"],
        "files": [encode_staged_file(item) for item in value["files"]],
        "limits": encode_crsp_limits(value["limits"]),
    }


def decode_crsp_action_evidence(value: JSONValue) -> CrspActionEvidence:
    """Require exact lineage, validation diagnostic kind and complete file identities.

    Raises:
        ValueError: CRSP_ACTION_RECORD for inconsistent identity, time or inventory.
        TypeError: A required field has the wrong type.
    """
    obj = narrow_json_to_dict(value)
    if set(obj) != set(CrspActionEvidence.__annotations__):
        raise ValueError("CRSP_ACTION_RECORD: require exact diagnostic fields")
    kind, source = require_str(obj, "kind"), require_str(obj, "source")
    setup = None if obj["setup_sha256"] is None else require_digest(obj, "setup_sha256")
    if (
        kind != "crsp-corporate-action-diagnostic-v1"
        or source not in {"crsp_a_stock", "synthetic"}
        or (source == "synthetic") != (setup is not None)
    ):
        raise ValueError("CRSP_ACTION_RECORD: require diagnostic kind and exact source identity")
    cohort = decode_crsp_cohort(obj["cohort"])
    started, completed = require_str(obj, "started_at"), require_str(obj, "completed_at")
    transaction, snapshot = require_str(obj, "transaction_started_at"), require_str(obj, "snapshot")
    times = tuple(datetime.fromisoformat(item) for item in (started, completed, transaction))
    if (
        any(item.tzinfo != UTC for item in times)
        or not datetime.fromisoformat(cohort["derived_at"]) <= times[0] <= times[1]
        or not snapshot.strip()
    ):
        raise ValueError("CRSP_ACTION_RECORD: invalid chronology or snapshot")
    files = tuple(decode_staged_file(item) for item in require_list(obj, "files"))
    expected = {
        relation.role + suffix for relation in CRSP_ACTIONS for suffix in (".csv", ".sql")
    } | {"protocol.json", "extract.sql", "snapshot-first.csv", "snapshot-last.csv"}
    if setup is not None:
        expected.add("setup.sql")
    if len(files) != len(expected) or {item["name"] for item in files} != expected:
        raise ValueError("CRSP_ACTION_RECORD: require every exact query and source file once")
    by_name = {item["name"]: item for item in files}
    if by_name["protocol.json"] != cohort["subset"]["protocol"] or (
        setup is not None and by_name["setup.sql"]["sha256"] != setup
    ):
        raise ValueError("CRSP_ACTION_RECORD: protocol or synthetic definitions differ")
    return CrspActionEvidence(
        kind=kind,
        cohort_bundle_sha256=require_digest(obj, "cohort_bundle_sha256"),
        cohort=cohort,
        source=source,
        setup_sha256=setup,
        started_at=started,
        completed_at=completed,
        snapshot=snapshot,
        transaction_started_at=transaction,
        files=files,
        limits=decode_crsp_limits(obj["limits"]),
    )


__all__ = ["CrspActionEvidence", "decode_crsp_action_evidence", "encode_crsp_action_evidence"]
