"""Evaluator-side check of a collected extraction: ciphertext only, then the existing reader.

The collected tree must hold exactly the receipt, the metrics and two bundles
of age ciphertext. The train-validation bundle is opened with its own key into
private scratch, reread by ``load_crsp_sources`` and checked against its
partition. A final bundle is opened only for synthetic evidence: licensed
final-period evidence is opened by the one-shot final release and nothing else.
The result is metadata: identities, counts and the latest date per column, with
the sha256 of the collected receipt and of each partition's ``bundle.json`` so
the reading names the files the native run's own load report digested.
"""

from __future__ import annotations

from collections.abc import Generator
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
from .crsp_writer import SYNTHETIC_SOURCE
from .encrypted_storage import decrypted_bundle, verify_encrypted_bundle
from .research_keys import EvaluatorKeys

#: The first line of every age v1 file.
AGE_HEADER = b"age-encryption.org/v1\n"


def _collected_files(outputs: Path) -> Generator[Path, None, None]:
    """Yield only regular files in the exact two-partition collection layout.

    Raises:
        ValueError: CRSP-013 for links, special files, nested directories or
            missing collection metadata; dangling links are never ignored.
    """
    if outputs.is_symlink() or not outputs.is_dir():
        raise ValueError("[CRSP-013] unsafe collected directory")
    expected = {"extraction.json", "metrics.json", "train-validation", "final"}
    if {path.name for path in outputs.iterdir()} != expected:
        raise ValueError("[CRSP-013] unexpected or incomplete collected root inventory")
    for path in sorted(outputs.rglob("*")):
        name = path.relative_to(outputs).as_posix()
        if path.is_symlink():
            raise ValueError(f"[CRSP-013] symlink in collected evidence {name}")
        if path.is_dir():
            if name not in {"train-validation", "final"}:
                raise ValueError(f"[CRSP-013] unexpected collected directory {name}")
        elif path.is_file():
            if name in {"train-validation", "final"}:
                raise ValueError(f"[CRSP-013] expected collected partition directory {name}")
            yield path
        else:
            raise ValueError(f"[CRSP-013] nonregular collected file {name}")


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
    for path in _collected_files(outputs):
        name = path.relative_to(outputs).as_posix()
        leaf = path.name
        if name in {"extraction.json", "metrics.json"}:
            continue
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
        max_plaintext_bytes=min(item["plaintext_bytes"], keys.access["max_plaintext_bytes"]),
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
        Metadata for both partitions, each with its verified ``bundle.json``
        sha256, the collected ``extraction.json`` sha256 and the ciphertext
        file count.

    Raises:
        ValueError: CRSP-013 for plaintext in the collected tree, a final open of
            licensed evidence or a recipient differing from the evaluator's;
            receipt, bundle, identity, reader and partition refusals propagate.
        OSError: A collected file or key cannot be read.
    """
    outputs = collected / "outputs"
    ciphertext = require_ciphertext_only(outputs)
    receipt_bytes = (outputs / "extraction.json").read_bytes()
    receipt = decode_crsp_extraction_receipt(load_json_str(receipt_bytes.decode("utf-8")))
    if open_final and receipt["source"] != SYNTHETIC_SOURCE:
        raise ValueError("[CRSP-013] licensed final evidence opens only through its release")
    items = (receipt["train_validation"], receipt["final"])
    recipients = (keys.access["validation_recipient"], keys.access["final_recipient"])
    partitions: list[JSONValue] = []
    for item, recipient in zip(items, recipients, strict=True):
        path = outputs / item["partition"] / "bundle.json"
        bundle = verify_encrypted_bundle(path, item["bundle"]["sha256"])
        if bundle["partition"] != item["partition"] or bundle["recipient"] != recipient:
            raise ValueError("[CRSP-013] a bundle is not sealed to its partition's recipient")
    # Verify both sealed inventories before opening even the validation partition.
    for item in items:
        path = outputs / item["partition"] / "bundle.json"
        if item["partition"] == "final" and not open_final:
            summary: JSONObject = {"partition": "final", "opened": False}
        else:
            summary = _open(path, item, keys, scratch, receipt)
        summary["bundle_sha256"] = item["bundle"]["sha256"]
        partitions.append(summary)
    return {
        "source": receipt["source"],
        "snapshot": receipt["snapshot"],
        "extraction_sha256": sha256(receipt_bytes).hexdigest(),
        "ciphertext_files": ciphertext,
        "partitions": partitions,
    }


__all__ = ["AGE_HEADER", "require_ciphertext_only", "verify_crsp_extraction"]
