"""Revise compatible CRSP protocol metadata without opening observation ciphertext."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

from platform_core.json_utils import JSONObject, dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.stage_manifest import StagedFile

from .crsp_extract import crsp_extract_from_records, encode_crsp_extract
from .crsp_extraction_records import CrspExtractionReceipt, decode_crsp_extraction_receipt
from .crsp_protocol import decode_crsp_protocol
from .crsp_revision_records import (
    CrspBundleRevision,
    CrspProtocolRevision,
    decode_crsp_protocol_revision,
    encode_crsp_protocol_revision,
)
from .crsp_verify import require_ciphertext_only
from .encrypted_records import EncryptedBundle
from .encrypted_storage import revise_encrypted_bundle, staged_record, verify_encrypted_bundle
from .research_keys import ResearchAccess


class _RevisionInputs(NamedTuple):
    """Verified original metadata and compatible replacement bytes."""

    receipt: CrspExtractionReceipt
    protocol: StagedFile
    protocol_bytes: bytes
    bundles: tuple[EncryptedBundle, ...]
    manifests: tuple[bytes, ...]


def _inputs(
    outputs: Path,
    receipt_digest: str,
    previous_protocol: Path,
    protocol: Path,
    access: ResearchAccess,
) -> _RevisionInputs:
    """Verify the pinned extraction and reconstruct its exact original manifests.

    Only version, entry execution and cost-accounting rules may differ. Every
    extraction setting must remain identical, including dates and columns.
    """
    require_ciphertext_only(outputs)
    payload = (outputs / "extraction.json").read_bytes()
    if sha256(payload).hexdigest() != receipt_digest:
        raise ValueError("CRSP_REVISION_RECEIPT: extraction receipt differs from its pin")
    receipt = decode_crsp_extraction_receipt(load_json_str(payload.decode()))
    previous = previous_protocol.read_bytes()
    if staged_record(BytesIO(previous), "protocol.json") != receipt["protocol"]:
        raise ValueError("CRSP_REVISION_PROTOCOL: original protocol differs from extraction")
    current = protocol.read_bytes()
    current_document = narrow_json_to_dict(load_json_str(current.decode()))
    decode_crsp_protocol(current_document)
    old_document = narrow_json_to_dict(load_json_str(previous.decode()))
    allowed = {"version", "entry_rule", "cost_rule"}
    if (
        current_document == old_document
        or set(current_document) - allowed != set(old_document) - allowed
        or any(
            current_document[key] != old_document[key]
            for key in current_document
            if key not in allowed
        )
    ):
        raise ValueError(
            "CRSP_REVISION_RULES: require changed accounting with identical extraction settings"
        )
    replacement = staged_record(BytesIO(current), "protocol.json")
    bundles: list[EncryptedBundle] = []
    manifests: list[bytes] = []
    for item, recipient in zip(
        (receipt["train_validation"], receipt["final"]),
        (access["validation_recipient"], access["final_recipient"]),
        strict=True,
    ):
        path = outputs / item["partition"] / "bundle.json"
        bundle = verify_encrypted_bundle(path, item["bundle"]["sha256"])
        with path.open("rb") as source:
            identity = staged_record(source, "bundle.json")
        if (
            identity != item["bundle"]
            or bundle["partition"] != item["partition"]
            or bundle["recipient"] != recipient
            or bundle["manifest"] != item["manifest"]
        ):
            raise ValueError(
                "CRSP_REVISION_PARTITION: original bundle differs from its receipt or recipient"
            )
        plain = {file["plaintext"]["name"]: file["plaintext"] for file in bundle["files"]}
        if (
            plain.get("protocol.json") != receipt["protocol"]
            or bundle["manifest"]["name"] != "manifest.json"
        ):
            raise ValueError(
                "CRSP_REVISION_MANIFEST: canonical protocol and manifest identities required"
            )
        records = tuple(
            value for name, value in plain.items() if name not in {"protocol.json", "manifest.json"}
        )
        original_manifest = crsp_extract_from_records(
            receipt["started_at"], receipt["protocol"], records
        )
        original_bytes = dump_json_str(encode_crsp_extract(original_manifest)).encode()
        if staged_record(BytesIO(original_bytes), "manifest.json") != bundle["manifest"]:
            raise ValueError(
                "CRSP_REVISION_MANIFEST: reconstructed original manifest differs from its pin"
            )
        revised = crsp_extract_from_records(receipt["started_at"], replacement, records)
        manifests.append(dump_json_str(encode_crsp_extract(revised)).encode())
        bundles.append(bundle)
    return _RevisionInputs(receipt, replacement, current, tuple(bundles), tuple(manifests))


def revise_crsp_protocol(
    outputs: Path,
    receipt_digest: str,
    previous_protocol: Path,
    protocol: Path,
    access: ResearchAccess,
    destination: Path,
) -> CrspProtocolRevision:
    """Publish two compatible metadata revisions with the original extraction vintage.

    Args:
        outputs: Original collected extraction output, retained unchanged.
        receipt_digest: Independently frozen extraction receipt SHA-256.
        previous_protocol: Exact public protocol bytes used for extraction.
        protocol: Current supported protocol, changing accounting rules only.
        access: Public recipients; no key is acquired.
        destination: New exclusive revision root.

    Returns:
        Published transformation lineage; no new extraction is claimed.

    Raises:
        ValueError: Changed or incompatible source metadata or ciphertext.
        OSError: Existing destination or failed storage operation.
        Exception: Encryption failure; the complete revision receipt is absent.
    """
    prepared = _inputs(outputs, receipt_digest, previous_protocol, protocol, access)
    destination.mkdir(mode=0o700, exist_ok=False)
    partitions: list[CrspBundleRevision] = []
    with TemporaryDirectory(prefix="crsp-revision-metadata-", dir=destination.parent) as temporary:
        metadata = Path(temporary)
        (metadata / "protocol.json").write_bytes(prepared.protocol_bytes)
        for item, manifest_bytes in zip(
            (prepared.receipt["train_validation"], prepared.receipt["final"]),
            prepared.manifests,
            strict=True,
        ):
            (metadata / "manifest.json").write_bytes(manifest_bytes)
            manifest = staged_record(BytesIO(manifest_bytes), "manifest.json")
            revise_encrypted_bundle(
                outputs / item["partition"] / "bundle.json",
                item["bundle"]["sha256"],
                metadata,
                destination / item["partition"],
                files=(prepared.protocol, manifest),
                manifest=manifest,
            )
            with (destination / item["partition"] / "bundle.json").open("rb") as stream:
                current = staged_record(stream, "bundle.json")
            partitions.append(
                CrspBundleRevision(
                    previous=item["bundle"],
                    current=current,
                    previous_manifest=item["manifest"],
                    manifest=manifest,
                )
            )
    revision = CrspProtocolRevision(
        extraction_sha256=receipt_digest,
        extracted_at=prepared.receipt["started_at"],
        previous_protocol=prepared.receipt["protocol"],
        protocol=prepared.protocol,
        train_validation=partitions[0],
        final=partitions[1],
    )
    with (destination / "revision.json").open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(encode_crsp_protocol_revision(revision)))
    return revision


def verify_crsp_revision(
    outputs: Path,
    receipt_digest: str,
    previous_protocol: Path,
    protocol: Path,
    access: ResearchAccess,
    destination: Path,
) -> JSONObject:
    """Reread original and revised ciphertext and independently reconstruct lineage.

    Args:
        outputs: Original extraction output.
        receipt_digest: Independently pinned original receipt.
        previous_protocol: Original frozen protocol bytes.
        protocol: Expected current protocol bytes.
        access: Public partition recipients, without acquiring keys.
        destination: Complete revision root to verify.

    Returns:
        Metadata-only evidence suitable for the existing service load reporter.

    Raises:
        ValueError: Incomplete revision, altered identities or relation ciphertext.
        OSError: Unreadable original or revised files.
    """
    prepared = _inputs(outputs, receipt_digest, previous_protocol, protocol, access)
    if destination.is_symlink() or {p.name for p in destination.iterdir()} != {
        "revision.json",
        "train-validation",
        "final",
    }:
        raise ValueError("CRSP_REVISION_INVENTORY: exact regular revision root required")
    receipt_path = destination / "revision.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("CRSP_REVISION_INVENTORY: require regular revision receipt")
    payload = receipt_path.read_bytes()
    revision = decode_crsp_protocol_revision(load_json_str(payload.decode()))
    if (
        revision["extraction_sha256"] != receipt_digest
        or revision["extracted_at"] != prepared.receipt["started_at"]
        or revision["previous_protocol"] != prepared.receipt["protocol"]
        or revision["protocol"] != prepared.protocol
    ):
        raise ValueError("CRSP_REVISION_LINEAGE: receipt differs from expected transformation")
    for item, changed, old, manifest_bytes in zip(
        (prepared.receipt["train_validation"], prepared.receipt["final"]),
        (revision["train_validation"], revision["final"]),
        prepared.bundles,
        prepared.manifests,
        strict=True,
    ):
        path = destination / item["partition"] / "bundle.json"
        new = verify_encrypted_bundle(path, changed["current"]["sha256"])
        with path.open("rb") as source:
            current = staged_record(source, "bundle.json")
        manifest = staged_record(BytesIO(manifest_bytes), "manifest.json")
        expected = CrspBundleRevision(
            previous=item["bundle"],
            current=current,
            previous_manifest=item["manifest"],
            manifest=manifest,
        )
        if (
            changed != expected
            or new["manifest"] != manifest
            or (new["partition"], new["recipient"]) != (old["partition"], old["recipient"])
        ):
            raise ValueError(
                "CRSP_REVISION_LINEAGE: revised partition differs from expected identities"
            )
        retained = {file["plaintext"]["name"]: file for file in old["files"]}
        revised = {file["plaintext"]["name"]: file for file in new["files"]}
        if set(retained) != set(revised):
            raise ValueError("CRSP_REVISION_FILES: revision changed the file inventory")
        for name, file in revised.items():
            if name in {"protocol.json", "manifest.json"}:
                expected_plain = prepared.protocol if name == "protocol.json" else manifest
                if file["plaintext"] != expected_plain:
                    raise ValueError("CRSP_REVISION_FILES: replacement metadata differs")
            elif file != retained[name]:
                raise ValueError("CRSP_REVISION_FILES: relation ciphertext must remain unchanged")
    return {
        "kind": "CRSP_PROTOCOL_REVISION_EVIDENCE",
        "revision_sha256": sha256(payload).hexdigest(),
        "revision": encode_crsp_protocol_revision(revision),
        "retained_relation_ciphertexts": 16,
        "opened_observations": False,
    }


__all__ = ["revise_crsp_protocol", "verify_crsp_revision"]
