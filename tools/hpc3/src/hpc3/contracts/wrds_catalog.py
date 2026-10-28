"""Validated metadata records from the standalone WRDS filesystem probe."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    require_bool,
    require_int,
    require_str,
)
from typing_extensions import TypedDict


class CatalogEntry(TypedDict):
    """One observed path; readable does not imply redistribution permission."""

    kind: str
    path: str
    bytes: int
    readable: bool
    target: str


def encode_catalog_entry(entry: CatalogEntry) -> list[JSONValue]:
    """Encode a catalog entry to the probe's ordered wire format.

    Args:
        entry: Validated record.

    Returns:
        The five wire fields.
    """
    return [entry["kind"], entry["path"], entry["bytes"], entry["readable"], entry["target"]]


def decode_catalog_entry(value: JSONValue) -> CatalogEntry:
    """Validate a probe record without accepting truncated or unknown records.

    Args:
        value: Parsed JSON value.

    Returns:
        A typed record.

    Raises:
        JSONTypeError: For a malformed record.
    """
    if not isinstance(value, list) or len(value) != 5:
        raise JSONTypeError("WRDS_CATALOG_RECORD: expected five fields")
    fields: dict[str, JSONValue] = dict(
        zip(("kind", "path", "bytes", "readable", "target"), value, strict=True)
    )
    kind = require_str(fields, "kind")
    path = require_str(fields, "path")
    size = require_int(fields, "bytes")
    readable = require_bool(fields, "readable")
    target = require_str(fields, "target")
    if kind not in {
        "file",
        "directory",
        "alias",
        "denied_directory",
        "unresolved_link",
        "special",
        "complete",
    }:
        raise JSONTypeError(f"WRDS_CATALOG_KIND: {kind}")
    if path == "" or size < 0:
        raise JSONTypeError("WRDS_CATALOG_VALUE: empty path or negative size")
    return CatalogEntry(kind=kind, path=path, bytes=size, readable=readable, target=target)


class CatalogReceipt(TypedDict):
    """Where and how one filesystem catalog was produced.

    ``job_id`` is the WRDS Grid Engine job's id, not a Slurm ledger row.
    """

    host: str
    scheduler: str
    job_id: str
    source_sha256: str
    catalog_sha256: str
    remote_catalog: str
    scope: str


class CatalogRun(TypedDict):
    """A verified catalog's records and the receipt saved beside it."""

    entries: list[CatalogEntry]
    receipt: CatalogReceipt


def encode_catalog_receipt(receipt: CatalogReceipt) -> JSONObject:
    """Encode a receipt for the JSON file saved beside its catalog.

    Args:
        receipt: Validated receipt.

    Returns:
        The JSON object.
    """
    return {
        "host": receipt["host"],
        "scheduler": receipt["scheduler"],
        "job_id": receipt["job_id"],
        "source_sha256": receipt["source_sha256"],
        "catalog_sha256": receipt["catalog_sha256"],
        "remote_catalog": receipt["remote_catalog"],
        "scope": receipt["scope"],
    }


def decode_catalog_receipt(value: JSONValue) -> CatalogReceipt:
    """Validate a saved receipt.

    Args:
        value: Parsed JSON value.

    Returns:
        The typed receipt.

    Raises:
        JSONTypeError: When the value is not an object or a field is missing
            or not a string.
    """
    fields = narrow_json_to_dict(value)
    return CatalogReceipt(
        host=require_str(fields, "host"),
        scheduler=require_str(fields, "scheduler"),
        job_id=require_str(fields, "job_id"),
        source_sha256=require_str(fields, "source_sha256"),
        catalog_sha256=require_str(fields, "catalog_sha256"),
        remote_catalog=require_str(fields, "remote_catalog"),
        scope=require_str(fields, "scope"),
    )


__all__ = [
    "CatalogEntry",
    "CatalogReceipt",
    "CatalogRun",
    "decode_catalog_entry",
    "decode_catalog_receipt",
    "encode_catalog_entry",
    "encode_catalog_receipt",
]
