"""Explicit lineage for a protocol revision of an already encrypted extraction."""

from datetime import UTC, datetime

from platform_core.json_utils import JSONObject, JSONValue, narrow_json_to_dict, require_str
from platform_core.stage_manifest import StagedFile, decode_staged_file, encode_staged_file
from typing_extensions import ReadOnly, TypedDict

from .campaign_records import require_digest


class CrspBundleRevision(TypedDict):
    """Old and new bundle/manifest identities for one unchanged partition."""

    previous: ReadOnly[StagedFile]
    current: ReadOnly[StagedFile]
    previous_manifest: ReadOnly[StagedFile]
    manifest: ReadOnly[StagedFile]


class CrspProtocolRevision(TypedDict):
    """Transformation provenance; this is not a new extraction receipt."""

    extraction_sha256: ReadOnly[str]
    extracted_at: ReadOnly[str]
    previous_protocol: ReadOnly[StagedFile]
    protocol: ReadOnly[StagedFile]
    train_validation: ReadOnly[CrspBundleRevision]
    final: ReadOnly[CrspBundleRevision]


def encode_crsp_bundle_revision(value: CrspBundleRevision) -> JSONObject:
    """Encode the four exact transport identities."""
    return {
        "previous": encode_staged_file(value["previous"]),
        "current": encode_staged_file(value["current"]),
        "previous_manifest": encode_staged_file(value["previous_manifest"]),
        "manifest": encode_staged_file(value["manifest"]),
    }


def decode_crsp_bundle_revision(value: JSONValue) -> CrspBundleRevision:
    """Refuse missing or extra lineage fields."""
    obj = narrow_json_to_dict(value)
    if set(obj) != set(CrspBundleRevision.__annotations__):
        raise ValueError("CRSP_REVISION_FIELDS: exact bundle lineage fields required")
    return CrspBundleRevision(
        previous=decode_staged_file(obj["previous"]),
        current=decode_staged_file(obj["current"]),
        previous_manifest=decode_staged_file(obj["previous_manifest"]),
        manifest=decode_staged_file(obj["manifest"]),
    )


def encode_crsp_protocol_revision(value: CrspProtocolRevision) -> JSONObject:
    """Encode lineage without observations, private keys or machine paths."""
    return {
        "extraction_sha256": value["extraction_sha256"],
        "extracted_at": value["extracted_at"],
        "previous_protocol": encode_staged_file(value["previous_protocol"]),
        "protocol": encode_staged_file(value["protocol"]),
        "train_validation": encode_crsp_bundle_revision(value["train_validation"]),
        "final": encode_crsp_bundle_revision(value["final"]),
    }


def decode_crsp_protocol_revision(value: JSONValue) -> CrspProtocolRevision:
    """Require exact lineage fields and the original UTC extraction vintage."""
    obj = narrow_json_to_dict(value)
    if set(obj) != set(CrspProtocolRevision.__annotations__):
        raise ValueError("CRSP_REVISION_FIELDS: exact protocol lineage fields required")
    extracted_at = require_str(obj, "extracted_at")
    if datetime.fromisoformat(extracted_at).tzinfo != UTC:
        raise ValueError("CRSP_REVISION_TIME: extraction vintage must carry UTC")
    return CrspProtocolRevision(
        extraction_sha256=require_digest(obj, "extraction_sha256"),
        extracted_at=extracted_at,
        previous_protocol=decode_staged_file(obj["previous_protocol"]),
        protocol=decode_staged_file(obj["protocol"]),
        train_validation=decode_crsp_bundle_revision(obj["train_validation"]),
        final=decode_crsp_bundle_revision(obj["final"]),
    )


__all__ = [
    "CrspBundleRevision",
    "CrspProtocolRevision",
    "decode_crsp_bundle_revision",
    "decode_crsp_protocol_revision",
    "encode_crsp_bundle_revision",
    "encode_crsp_protocol_revision",
]
