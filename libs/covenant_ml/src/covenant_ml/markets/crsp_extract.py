"""Four-relation CRSP evidence, bound to exact source and protocol bytes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_str,
)
from platform_core.stage_manifest import StagedFile, decode_staged_file, encode_staged_file
from typing_extensions import ReadOnly, TypedDict

from ..evaluation.evidence import read_evidence, require_evidence_filename
from .crsp_protocol import CrspProtocol, decode_crsp_protocol
from .crsp_rows import decode_crsp_rows


class CrspRelation(TypedDict):
    """Exact CSV and extraction query files; staging owns the file record shape."""

    csv: ReadOnly[StagedFile]
    query: ReadOnly[StagedFile]


class CrspExtract(TypedDict):
    """A single extraction vintage with its exact frozen protocol."""

    extracted_at: ReadOnly[str]
    protocol: ReadOnly[StagedFile]
    membership: ReadOnly[CrspRelation]
    returns: ReadOnly[CrspRelation]
    calendar: ReadOnly[CrspRelation]
    delists: ReadOnly[CrspRelation]


class CrspSources(NamedTuple):
    """Validated source rows; no inference about historical publication timing."""

    manifest: CrspExtract
    protocol: CrspProtocol
    membership: tuple[JSONObject, ...]
    returns: tuple[JSONObject, ...]
    calendar: tuple[JSONObject, ...]
    delists: tuple[JSONObject, ...]


def decode_crsp_relation(value: JSONValue) -> CrspRelation:
    """Decode exactly two staged files, refusing unknown fields with CRSP-006."""
    obj = narrow_json_to_dict(value)
    if set(obj) != {"csv", "query"}:
        raise ValueError("[CRSP-006] a relation requires exactly csv and query evidence")
    return CrspRelation(csv=decode_staged_file(obj["csv"]), query=decode_staged_file(obj["query"]))


def encode_crsp_relation(value: CrspRelation) -> JSONObject:
    """Encode both file identities using the shared staging codec."""
    return {"csv": encode_staged_file(value["csv"]), "query": encode_staged_file(value["query"])}


def decode_crsp_extract(value: JSONValue) -> CrspExtract:
    """Require UTC vintage, exact fields and distinct safe evidence filenames.

    Args:
        value: Untrusted JSON manifest.

    Returns:
        Validated source identities. Checksums do not attest vendor authorship.

    Raises:
        ValueError: CRSP-006 for ambiguous identity or extraction time.
        TypeError: Missing or wrongly typed fields.
    """
    obj = narrow_json_to_dict(value)
    if set(obj) != set(CrspExtract.__annotations__):
        raise ValueError("[CRSP-006] require exact extract fields")
    extracted = require_str(obj, "extracted_at")
    if datetime.fromisoformat(extracted).tzinfo != UTC:
        raise ValueError("[CRSP-006] extraction timestamp must carry UTC")
    record = CrspExtract(
        extracted_at=extracted,
        protocol=decode_staged_file(obj["protocol"]),
        membership=decode_crsp_relation(obj["membership"]),
        returns=decode_crsp_relation(obj["returns"]),
        calendar=decode_crsp_relation(obj["calendar"]),
        delists=decode_crsp_relation(obj["delists"]),
    )
    files = [record["protocol"]]
    for relation in (
        record["membership"],
        record["returns"],
        record["calendar"],
        record["delists"],
    ):
        files.extend((relation["csv"], relation["query"]))
    names = [require_evidence_filename(file["name"]) for file in files]
    if len(set(names)) != len(names):
        raise ValueError("[CRSP-006] evidence filenames must be distinct")
    return record


def encode_crsp_extract(value: CrspExtract) -> JSONObject:
    """Encode the vintage and every nested file identity without machine paths."""
    return {
        "extracted_at": value["extracted_at"],
        "protocol": encode_staged_file(value["protocol"]),
        "membership": encode_crsp_relation(value["membership"]),
        "returns": encode_crsp_relation(value["returns"]),
        "calendar": encode_crsp_relation(value["calendar"]),
        "delists": encode_crsp_relation(value["delists"]),
    }


def _read_staged(directory: Path, record: StagedFile) -> bytes:
    """Verify shared staging size in addition to the lifted reader's digest."""
    payload = read_evidence(directory, record["name"], record["sha256"])
    if len(payload) != record["size_bytes"]:
        raise ValueError("[CRSP-006] staged byte count differs from verified evidence")
    return payload


def load_crsp_sources(path: Path) -> CrspSources:
    """Read each source once, then decode only those verified bytes.

    Query bytes are preserved, not executed or claimed as proof of extraction.
    Empty terminal-event CSVs are valid when their exact header is present.
    Calendar completeness and decision eligibility are checked by the panel.

    Args:
        path: Manifest alongside its nine evidence files.

    Returns:
        Typed manifest, protocol and four validated row collections.

    Raises:
        ValueError: CRSP-006/007 for changed, malformed or duplicate evidence.
        OSError: An evidence file cannot be read.
    """
    manifest = decode_crsp_extract(load_json_str(path.read_text("utf-8")))
    protocol = decode_crsp_protocol(
        narrow_json_to_dict(
            load_json_str(_read_staged(path.parent, manifest["protocol"]).decode("utf-8"))
        )
    )
    records: list[tuple[JSONObject, ...]] = []
    for name, relation, columns in (
        ("membership", manifest["membership"], protocol["membership_columns"]),
        ("returns", manifest["returns"], protocol["returns_columns"]),
        ("calendar", manifest["calendar"], protocol["calendar_columns"]),
        ("delists", manifest["delists"], protocol["delists_columns"]),
    ):
        query = _read_staged(path.parent, relation["query"])
        if not query.strip():
            raise ValueError("[CRSP-006] extraction SQL must not be blank")
        payload = _read_staged(path.parent, relation["csv"])
        records.append(decode_crsp_rows(payload, name, columns, manifest["extracted_at"]))
    return CrspSources(manifest, protocol, records[0], records[1], records[2], records[3])
