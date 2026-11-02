"""Diagnostic-only encrypted metadata for a bounded source-vintage comparison."""

from __future__ import annotations

from datetime import UTC, datetime

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_str,
)
from platform_core.stage_manifest import (
    StagedFile,
    decode_staged_file,
    encode_staged_file,
)
from typing_extensions import ReadOnly, TypedDict

from .campaign_records import require_digest
from .crsp_cohort_records import CrspCohort, decode_crsp_cohort, encode_crsp_cohort
from .crsp_extract import CrspExtract, decode_crsp_extract, encode_crsp_extract
from .crsp_extraction_records import CrspLimits, decode_crsp_limits, encode_crsp_limits


class CrspRefresh(TypedDict):
    """Private lineage and transaction evidence; never a study extraction envelope."""

    kind: ReadOnly[str]
    cohort_bundle_sha256: ReadOnly[str]
    cohort: ReadOnly[CrspCohort]
    fresh: ReadOnly[CrspExtract]
    source: ReadOnly[str]
    setup_sha256: ReadOnly[str | None]
    completed_at: ReadOnly[str]
    snapshot: ReadOnly[str]
    transaction_started_at: ReadOnly[str]
    driver: ReadOnly[StagedFile]
    differences: ReadOnly[StagedFile]
    limits: ReadOnly[CrspLimits]


def encode_crsp_refresh(value: CrspRefresh) -> JSONObject:
    """Encode the diagnostic using the canonical cohort, extraction and staging codecs."""
    return {
        "kind": value["kind"],
        "cohort_bundle_sha256": value["cohort_bundle_sha256"],
        "cohort": encode_crsp_cohort(value["cohort"]),
        "fresh": encode_crsp_extract(value["fresh"]),
        "source": value["source"],
        "setup_sha256": value["setup_sha256"],
        "completed_at": value["completed_at"],
        "snapshot": value["snapshot"],
        "transaction_started_at": value["transaction_started_at"],
        "driver": encode_staged_file(value["driver"]),
        "differences": encode_staged_file(value["differences"]),
        "limits": encode_crsp_limits(value["limits"]),
    }


def decode_crsp_refresh(value: JSONValue) -> CrspRefresh:
    """Refuse ambiguous lineage, chronology and synthetic/vendor identity.

    Args:
        value: Untrusted diagnostic metadata.

    Returns:
        Exact typed fields. File identities bind bytes, not vendor authorship.

    Raises:
        ValueError: CRSP_REFRESH_RECORD for unknown fields or inconsistent metadata.
        TypeError: A required field has the wrong type.
    """
    obj = narrow_json_to_dict(value)
    if set(obj) != set(CrspRefresh.__annotations__):
        raise ValueError("CRSP_REFRESH_RECORD: require exact diagnostic fields")
    kind = require_str(obj, "kind")
    source = require_str(obj, "source")
    setup = None if obj["setup_sha256"] is None else require_digest(obj, "setup_sha256")
    if (
        kind != "crsp-diagnostic-refresh-v1"
        or source not in {"crsp_a_stock", "synthetic"}
        or (source == "synthetic") != (setup is not None)
    ):
        raise ValueError("CRSP_REFRESH_RECORD: require diagnostic kind and exact source identity")
    cohort = decode_crsp_cohort(obj["cohort"])
    fresh = decode_crsp_extract(obj["fresh"])
    completed = require_str(obj, "completed_at")
    transaction = require_str(obj, "transaction_started_at")
    snapshot = require_str(obj, "snapshot")
    if (
        fresh["protocol"] != cohort["subset"]["protocol"]
        or datetime.fromisoformat(completed).tzinfo != UTC
        or datetime.fromisoformat(transaction).tzinfo != UTC
        or not datetime.fromisoformat(cohort["derived_at"])
        <= datetime.fromisoformat(fresh["extracted_at"])
        <= datetime.fromisoformat(completed)
        or not snapshot.strip()
    ):
        raise ValueError("CRSP_REFRESH_RECORD: protocol, chronology or snapshot differs")
    return CrspRefresh(
        kind=kind,
        cohort_bundle_sha256=require_digest(obj, "cohort_bundle_sha256"),
        cohort=cohort,
        fresh=fresh,
        source=source,
        setup_sha256=setup,
        completed_at=completed,
        snapshot=snapshot,
        transaction_started_at=transaction,
        driver=decode_staged_file(obj["driver"]),
        differences=decode_staged_file(obj["differences"]),
        limits=decode_crsp_limits(obj["limits"]),
    )


__all__ = ["CrspRefresh", "decode_crsp_refresh", "encode_crsp_refresh"]
