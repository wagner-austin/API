"""Pinned inputs and diagnostic-only lineage for an encrypted CRSP source cohort."""

from __future__ import annotations

from datetime import UTC, datetime

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from .campaign_records import require_digest
from .crsp_extract import CrspExtract, decode_crsp_extract, encode_crsp_extract
from .validation import require_bounded_int


class CrspCohortPlan(TypedDict):
    """Independent input identities and explicit derivation bounds, without identifiers."""

    bundle_sha256: ReadOnly[str]
    protocol_sha256: ReadOnly[str]
    audit_sha256: ReadOnly[str]
    expected_cases: ReadOnly[int]
    max_source_rows: ReadOnly[int]
    max_cohort_bytes: ReadOnly[int]


def encode_crsp_cohort_plan(value: CrspCohortPlan) -> JSONObject:
    """Encode the three pinned identities and three explicit bounds."""
    return {
        "bundle_sha256": value["bundle_sha256"],
        "protocol_sha256": value["protocol_sha256"],
        "audit_sha256": value["audit_sha256"],
        "expected_cases": value["expected_cases"],
        "max_source_rows": value["max_source_rows"],
        "max_cohort_bytes": value["max_cohort_bytes"],
    }


def decode_crsp_cohort_plan(value: JSONValue) -> CrspCohortPlan:
    """Require exact fields, SHA-256 identities and a maximum of 64 anchor cases.

    Raises:
        ValueError: CRSP_COHORT_PLAN for unknown fields; shared digest and bound
            validators refuse malformed identities or out-of-range limits.
        TypeError: A required field has the wrong JSON type.
    """
    obj = narrow_json_to_dict(value)
    if set(obj) != set(CrspCohortPlan.__annotations__):
        raise ValueError("CRSP_COHORT_PLAN: require exact plan fields")
    return CrspCohortPlan(
        bundle_sha256=require_digest(obj, "bundle_sha256"),
        protocol_sha256=require_digest(obj, "protocol_sha256"),
        audit_sha256=require_digest(obj, "audit_sha256"),
        expected_cases=require_bounded_int(obj, "expected_cases", 1, 64),
        max_source_rows=require_bounded_int(obj, "max_source_rows", 1, 100_000_000),
        max_cohort_bytes=require_bounded_int(obj, "max_cohort_bytes", 1, 1_000_000_000),
    )


class CrspCohort(TypedDict):
    """Encrypted transformation lineage, deliberately not a study extraction manifest."""

    kind: ReadOnly[str]
    derived_at: ReadOnly[str]
    plan: ReadOnly[CrspCohortPlan]
    source: ReadOnly[CrspExtract]
    subset: ReadOnly[CrspExtract]


def encode_crsp_cohort(value: CrspCohort) -> JSONObject:
    """Encode the diagnostic envelope using the existing source manifest codec."""
    return {
        "kind": value["kind"],
        "derived_at": value["derived_at"],
        "plan": encode_crsp_cohort_plan(value["plan"]),
        "source": encode_crsp_extract(value["source"]),
        "subset": encode_crsp_extract(value["subset"]),
    }


def decode_crsp_cohort(value: JSONValue) -> CrspCohort:
    """Refuse unknown envelopes, protocol drift or a claimed new extraction vintage.

    A filtered cohort retains its original source extraction time. Its distinct
    envelope cannot pass ``decode_crsp_extract`` or a study source loader.

    Raises:
        ValueError: CRSP_COHORT_MANIFEST for malformed or inconsistent lineage.
        TypeError: A required field has the wrong JSON type.
    """
    obj = narrow_json_to_dict(value)
    if set(obj) != set(CrspCohort.__annotations__):
        raise ValueError("CRSP_COHORT_MANIFEST: require exact diagnostic fields")
    kind = require_str(obj, "kind")
    derived = require_str(obj, "derived_at")
    if kind != "crsp-diagnostic-cohort-v1" or datetime.fromisoformat(derived).tzinfo != UTC:
        raise ValueError("CRSP_COHORT_MANIFEST: require diagnostic kind and UTC derivation time")
    plan = decode_crsp_cohort_plan(obj["plan"])
    source = decode_crsp_extract(obj["source"])
    subset = decode_crsp_extract(obj["subset"])
    if (
        source["protocol"]["sha256"] != plan["protocol_sha256"]
        or subset["protocol"] != source["protocol"]
        or subset["extracted_at"] != source["extracted_at"]
        or datetime.fromisoformat(derived) < datetime.fromisoformat(source["extracted_at"])
    ):
        raise ValueError("CRSP_COHORT_MANIFEST: protocol or extraction lineage differs")
    return CrspCohort(kind=kind, derived_at=derived, plan=plan, source=source, subset=subset)


__all__ = [
    "CrspCohort",
    "CrspCohortPlan",
    "decode_crsp_cohort",
    "decode_crsp_cohort_plan",
    "encode_crsp_cohort",
    "encode_crsp_cohort_plan",
]
