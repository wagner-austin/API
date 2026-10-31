"""Extraction limits and the metadata receipt published beside the encrypted bundles.

The receipt carries timestamps, the database snapshot, file identities and
row counts. It never carries a source row, a key or a credential.
"""

from __future__ import annotations

from datetime import UTC, datetime

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_bool,
    require_dict,
    require_str,
)
from platform_core.stage_manifest import StagedFile, decode_staged_file, encode_staged_file
from typing_extensions import ReadOnly, TypedDict

from .campaign_records import require_digest
from .validation import require_bounded_int, require_day

#: The four relation roles of a manifest, in the order every record lists them.
CRSP_RELATIONS = ("membership", "returns", "calendar", "delists")


class CrspLimits(TypedDict):
    """Explicit bounds applied before and during extraction."""

    max_rows: ReadOnly[int]
    max_relation_bytes: ReadOnly[int]
    max_partition_bytes: ReadOnly[int]
    seconds: ReadOnly[int]


class CrspPartitionReceipt(TypedDict):
    """One sealed partition: its bounds, plaintext manifest and bundle identities."""

    partition: ReadOnly[str]
    observation_end: ReadOnly[str]
    calendar_end: ReadOnly[str]
    manifest: ReadOnly[StagedFile]
    bundle: ReadOnly[StagedFile]
    membership_rows: ReadOnly[int]
    returns_rows: ReadOnly[int]
    calendar_rows: ReadOnly[int]
    delists_rows: ReadOnly[int]
    plaintext_bytes: ReadOnly[int]


class CrspExtractionReceipt(TypedDict):
    """Where, when and from which snapshot both partitions were produced."""

    schema: ReadOnly[str]
    setup_sha256: ReadOnly[str | None]
    protocol: ReadOnly[StagedFile]
    driver_sha256: ReadOnly[str]
    started_at: ReadOnly[str]
    completed_at: ReadOnly[str]
    snapshot: ReadOnly[str]
    transaction_started_at: ReadOnly[str]
    limits: ReadOnly[CrspLimits]
    train_validation: ReadOnly[CrspPartitionReceipt]
    final: ReadOnly[CrspPartitionReceipt]
    plaintext_removed: ReadOnly[bool]


def _exact(obj: JSONObject, fields: set[str], what: str) -> None:
    """Refuse missing or unknown fields with CRSP-010."""
    if set(obj) != fields:
        raise ValueError(f"[CRSP-010] {what} requires exactly its declared fields")


def _utc(obj: JSONObject, key: str) -> str:
    """Require an aware UTC ISO timestamp."""
    value = require_str(obj, key)
    if datetime.fromisoformat(value).tzinfo != UTC:
        raise ValueError(f"[CRSP-010] {key} must carry UTC")
    return value


def encode_crsp_limits(value: CrspLimits) -> JSONObject:
    """Encode every bound."""
    return {
        "max_rows": value["max_rows"],
        "max_relation_bytes": value["max_relation_bytes"],
        "max_partition_bytes": value["max_partition_bytes"],
        "seconds": value["seconds"],
    }


def decode_crsp_limits(value: JSONValue) -> CrspLimits:
    """Require positive bounds, with the partition budget covering one relation.

    Args:
        value: Untrusted limits document.

    Returns:
        Validated limits.

    Raises:
        ValueError: CRSP-010 for unknown fields or inconsistent budgets;
            MARKET-001 for a bound outside its interval.
        TypeError: Missing or wrongly typed fields.
    """
    obj = narrow_json_to_dict(value)
    _exact(obj, set(CrspLimits.__annotations__), "limits")
    limits = CrspLimits(
        max_rows=require_bounded_int(obj, "max_rows", 1, 10**9),
        max_relation_bytes=require_bounded_int(obj, "max_relation_bytes", 1, 2**40),
        max_partition_bytes=require_bounded_int(obj, "max_partition_bytes", 1, 2**42),
        seconds=require_bounded_int(obj, "seconds", 1, 86400),
    )
    if limits["max_partition_bytes"] < limits["max_relation_bytes"]:
        raise ValueError("[CRSP-010] the partition budget must cover one relation")
    return limits


def encode_crsp_partition_receipt(value: CrspPartitionReceipt) -> JSONObject:
    """Encode one partition's identities and counts."""
    return {
        "partition": value["partition"],
        "observation_end": value["observation_end"],
        "calendar_end": value["calendar_end"],
        "manifest": encode_staged_file(value["manifest"]),
        "bundle": encode_staged_file(value["bundle"]),
        "membership_rows": value["membership_rows"],
        "returns_rows": value["returns_rows"],
        "calendar_rows": value["calendar_rows"],
        "delists_rows": value["delists_rows"],
        "plaintext_bytes": value["plaintext_bytes"],
    }


def decode_crsp_partition_receipt(value: JSONValue, partition: str) -> CrspPartitionReceipt:
    """Require the expected partition name, canonical days and nonnegative counts.

    Args:
        value: Untrusted partition document.
        partition: The name its position in the receipt requires.

    Returns:
        Validated partition receipt.

    Raises:
        ValueError: CRSP-010 for unknown fields or a different partition name;
            MARKET-001/004 for invalid counts or days.
        TypeError: Missing or wrongly typed fields.
    """
    obj = narrow_json_to_dict(value)
    _exact(obj, set(CrspPartitionReceipt.__annotations__), "a partition receipt")
    if require_str(obj, "partition") != partition:
        raise ValueError(f"[CRSP-010] expected the {partition} partition")
    counts = {
        key: require_bounded_int(obj, key, 0, 10**9)
        for key in ("membership_rows", "returns_rows", "calendar_rows", "delists_rows")
    }
    return CrspPartitionReceipt(
        partition=partition,
        observation_end=require_day(obj, "observation_end"),
        calendar_end=require_day(obj, "calendar_end"),
        manifest=decode_staged_file(obj["manifest"]),
        bundle=decode_staged_file(obj["bundle"]),
        membership_rows=counts["membership_rows"],
        returns_rows=counts["returns_rows"],
        calendar_rows=counts["calendar_rows"],
        delists_rows=counts["delists_rows"],
        plaintext_bytes=require_bounded_int(obj, "plaintext_bytes", 0, 2**42),
    )


def encode_crsp_extraction_receipt(value: CrspExtractionReceipt) -> JSONObject:
    """Encode the complete receipt without machine paths."""
    return {
        "schema": value["schema"],
        "setup_sha256": value["setup_sha256"],
        "protocol": encode_staged_file(value["protocol"]),
        "driver_sha256": value["driver_sha256"],
        "started_at": value["started_at"],
        "completed_at": value["completed_at"],
        "snapshot": value["snapshot"],
        "transaction_started_at": value["transaction_started_at"],
        "limits": encode_crsp_limits(value["limits"]),
        "train_validation": encode_crsp_partition_receipt(value["train_validation"]),
        "final": encode_crsp_partition_receipt(value["final"]),
        "plaintext_removed": value["plaintext_removed"],
    }


def decode_crsp_extraction_receipt(value: JSONValue) -> CrspExtractionReceipt:
    """Require exact fields, UTC ordering and a recorded plaintext removal.

    Args:
        value: Untrusted receipt document.

    Returns:
        Validated receipt.

    Raises:
        ValueError: CRSP-010 for unknown fields, unordered timestamps, a blank
            snapshot or plaintext that was not removed.
        TypeError: Missing or wrongly typed fields.
    """
    obj = narrow_json_to_dict(value)
    _exact(obj, set(CrspExtractionReceipt.__annotations__), "an extraction receipt")
    setup = obj["setup_sha256"]
    started, completed = _utc(obj, "started_at"), _utc(obj, "completed_at")
    if datetime.fromisoformat(completed) < datetime.fromisoformat(started):
        raise ValueError("[CRSP-010] completion must not precede the start")
    snapshot = require_str(obj, "snapshot")
    if not snapshot.strip():
        raise ValueError("[CRSP-010] the database snapshot must be recorded")
    if not require_bool(obj, "plaintext_removed"):
        raise ValueError("[CRSP-010] private plaintext must have been removed")
    return CrspExtractionReceipt(
        schema=require_str(obj, "schema"),
        setup_sha256=None if setup is None else require_digest(obj, "setup_sha256"),
        protocol=decode_staged_file(obj["protocol"]),
        driver_sha256=require_digest(obj, "driver_sha256"),
        started_at=started,
        completed_at=completed,
        snapshot=snapshot,
        transaction_started_at=require_str(obj, "transaction_started_at"),
        limits=decode_crsp_limits(require_dict(obj, "limits")),
        train_validation=decode_crsp_partition_receipt(obj["train_validation"], "train-validation"),
        final=decode_crsp_partition_receipt(obj["final"], "final"),
        plaintext_removed=True,
    )


__all__ = [
    "CRSP_RELATIONS",
    "CrspExtractionReceipt",
    "CrspLimits",
    "CrspPartitionReceipt",
    "decode_crsp_extraction_receipt",
    "decode_crsp_limits",
    "decode_crsp_partition_receipt",
    "encode_crsp_extraction_receipt",
    "encode_crsp_limits",
    "encode_crsp_partition_receipt",
]
