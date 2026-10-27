"""Validated metadata records from the standalone WRDS filesystem probe."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
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


__all__ = ["CatalogEntry", "decode_catalog_entry", "encode_catalog_entry"]
