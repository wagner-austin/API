"""Evaluator-side check of a collected extraction: ciphertext only, then the existing reader.

The collected tree must hold exactly the receipt, the metrics and two bundles
of age ciphertext. The train-validation bundle is opened with its own key into
private scratch, reread by ``load_crsp_sources`` and checked against its
partition. A final bundle is opened only for synthetic evidence: licensed
final-period evidence is opened by the one-shot final release and nothing else.
The result is metadata: identities, counts and the latest date per column.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, JSONValue, load_json_str, require_str

from .crsp_extract import CrspSources, load_crsp_sources
from .crsp_extraction_records import (
    CrspExtractionReceipt,
    CrspPartitionReceipt,
    decode_crsp_extraction_receipt,
)
from .crsp_partition import check_crsp_partition
from .crsp_queries import crsp_scopes
from .crsp_writer import SYNTHETIC_SCHEMA
from .encrypted_storage import decrypted_bundle, read_encrypted_bundle
from .research_keys import EvaluatorKeys

#: The first line of every age v1 file.
AGE_HEADER = b"age-encryption.org/v1\n"


def require_ciphertext_only(outputs: Path) -> int:
    """Refuse any collected file other than the receipt, metrics and age bundles.

    Args:
        outputs: The collected ``outputs`` directory.

    Returns:
        The number of age ciphertext files.

    Raises:
        ValueError: CRSP-013 for an unexpected name or a file without the age header.
    """
    ciphertext = 0
    for path in sorted(item for item in outputs.rglob("*") if item.is_file()):
        name = path.relative_to(outputs).as_posix()
        partition, _, leaf = name.partition("/")
        if name in {"extraction.json", "metrics.json"}:
            continue
        if partition not in {"train-validation", "final"} or "/" in leaf or not leaf:
            raise ValueError(f"[CRSP-013] unexpected collected file {name}")
        if leaf == "bundle.json":
            continue
        with path.open("rb") as stream:
            if not leaf.endswith(".age") or stream.read(len(AGE_HEADER)) != AGE_HEADER:
                raise ValueError(f"[CRSP-013] collected file {name} is not age ciphertext")
        ciphertext += 1
    return ciphertext


def _latest(rows: tuple[JSONObject, ...], field: str) -> JSONValue:
    """Return the latest present date in one column, or None when there is none."""
    days = [require_str(row, field) for row in rows if row[field] is not None]
    return max(days) if days else None


def _summary(sources: CrspSources, item: CrspPartitionReceipt, max_rows: int) -> JSONObject:
    """Check one opened partition against its receipt and describe it without rows.

    Raises:
        ValueError: CRSP-013 when counts differ from the receipt; CRSP-012 and
            calendar refusals propagate from the partition check.
    """
    scope = crsp_scopes(sources.protocol)[0 if item["partition"] == "train-validation" else 1]
    check_crsp_partition(sources, scope, max_rows)
    counts = (
        len(sources.membership),
        len(sources.returns),
        len(sources.calendar),
        len(sources.delists),
    )
    expected = (
        item["membership_rows"],
        item["returns_rows"],
        item["calendar_rows"],
        item["delists_rows"],
    )
    if counts != expected:
        raise ValueError("[CRSP-013] decrypted row counts differ from the receipt")
    return {
        "partition": item["partition"],
        "opened": True,
        "observation_end": scope.observation_end,
        "latest_return": _latest(sources.returns, "dlycaldt"),
        "latest_membership_end": _latest(sources.membership, "nameendt"),
        "latest_delisting": _latest(sources.delists, "delistingdt"),
        "latest_calendar": _latest(sources.calendar, "caldt"),
        "rows": list(counts),
    }


def _open(
    bundle: Path,
    item: CrspPartitionReceipt,
    keys: EvaluatorKeys,
    scratch: Path,
    receipt: CrspExtractionReceipt,
) -> JSONObject:
    """Decrypt one partition with its own identity and run the existing reader.

    Raises:
        ValueError: CRSP-013 for a manifest differing from the receipt; bundle,
            identity and reader refusals propagate.
    """
    with decrypted_bundle(
        bundle,
        item["bundle"]["sha256"],
        partition=item["partition"],
        identity=keys.acquire(item["partition"]),
        scratch=scratch,
        max_plaintext_bytes=item["plaintext_bytes"],
    ) as manifest:
        if sha256(manifest.read_bytes()).hexdigest() != item["manifest"]["sha256"]:
            raise ValueError("[CRSP-013] decrypted manifest differs from the receipt")
        summary = _summary(load_crsp_sources(manifest), item, receipt["limits"]["max_rows"])
        private = manifest.parent
    summary["plaintext_removed"] = not private.exists()
    return summary


def verify_crsp_extraction(
    collected: Path, keys: EvaluatorKeys, *, scratch: Path, open_final: bool
) -> JSONObject:
    """Verify a collected extraction through decryption and the existing reader.

    Args:
        collected: The native collector's output directory.
        keys: Evaluator identities; the final key is read only when opening final.
        scratch: Evaluator-private scratch for transient plaintext.
        open_final: Also open the final bundle; permitted only for synthetic evidence.

    Returns:
        Metadata for both partitions and the ciphertext file count.

    Raises:
        ValueError: CRSP-013 for plaintext in the collected tree, a final open of
            licensed evidence or a recipient differing from the evaluator's;
            receipt, bundle, identity, reader and partition refusals propagate.
        OSError: A collected file or key cannot be read.
    """
    outputs = collected / "outputs"
    ciphertext = require_ciphertext_only(outputs)
    receipt = decode_crsp_extraction_receipt(
        load_json_str((outputs / "extraction.json").read_text("utf-8"))
    )
    if open_final and receipt["schema"] != SYNTHETIC_SCHEMA:
        raise ValueError("[CRSP-013] licensed final evidence opens only through its release")
    items = (receipt["train_validation"], receipt["final"])
    recipients = (keys.access["validation_recipient"], keys.access["final_recipient"])
    partitions: list[JSONValue] = []
    for item, recipient in zip(items, recipients, strict=True):
        path = outputs / item["partition"] / "bundle.json"
        bundle = read_encrypted_bundle(path, item["bundle"]["sha256"])
        if bundle["partition"] != item["partition"] or bundle["recipient"] != recipient:
            raise ValueError("[CRSP-013] a bundle is not sealed to its partition's recipient")
        if item["partition"] == "final" and not open_final:
            partitions.append({"partition": "final", "opened": False})
            continue
        partitions.append(_open(path, item, keys, scratch, receipt))
    return {
        "schema": receipt["schema"],
        "snapshot": receipt["snapshot"],
        "ciphertext_files": ciphertext,
        "partitions": partitions,
    }


__all__ = ["AGE_HEADER", "require_ciphertext_only", "verify_crsp_extraction"]
