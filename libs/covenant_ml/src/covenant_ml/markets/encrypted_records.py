"""Encrypted transport identities alongside the existing plaintext staging records.

The admission policy must pin the bundle manifest digest. Checksums alone do
not authenticate a manifest supplied by an adversary. No secret belongs here.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_list,
    require_str,
)
from platform_core.stage_manifest import StagedFile, decode_staged_file, encode_staged_file
from typing_extensions import ReadOnly, TypedDict

from ..evaluation.evidence import require_evidence_filename


class EncryptedFile(TypedDict):
    """One age file and the exact plaintext it is authorized to produce."""

    plaintext: ReadOnly[StagedFile]
    ciphertext: ReadOnly[StagedFile]


class EncryptedBundle(TypedDict):
    """Public metadata; the recipient is a public key, never an age identity."""

    format: ReadOnly[str]
    partition: ReadOnly[str]
    recipient: ReadOnly[str]
    manifest: ReadOnly[StagedFile]
    files: ReadOnly[tuple[EncryptedFile, ...]]


def decode_encrypted_file(value: JSONValue) -> EncryptedFile:
    """Require two existing staging records with safe, distinct names."""
    obj = narrow_json_to_dict(value)
    if set(obj) != set(EncryptedFile.__annotations__):
        raise ValueError("ENCRYPTED_FIELDS: exact encrypted file fields required")
    plain = decode_staged_file(obj["plaintext"])
    cipher = decode_staged_file(obj["ciphertext"])
    require_evidence_filename(plain["name"])
    require_evidence_filename(cipher["name"])
    if cipher["name"] != plain["name"] + ".age":
        raise ValueError("ENCRYPTED_NAME: ciphertext must use the plaintext name plus .age")
    return EncryptedFile(plaintext=plain, ciphertext=cipher)


def encode_encrypted_file(value: EncryptedFile) -> JSONObject:
    """Encode both transport and consumed-byte identities."""
    return {
        "plaintext": encode_staged_file(value["plaintext"]),
        "ciphertext": encode_staged_file(value["ciphertext"]),
    }


def decode_encrypted_bundle(value: JSONValue) -> EncryptedBundle:
    """Refuse ambiguous partitions, private keys, repeated files and absent manifests.

    Args:
        value: Untrusted decoded metadata.

    Returns:
        Exact immutable bundle metadata, still requiring a trusted digest pin.

    Raises:
        ValueError: ENCRYPTED_* for unsupported format or ambiguous evidence.
        TypeError: Missing or wrongly typed fields.
    """
    obj = narrow_json_to_dict(value)
    if set(obj) != set(EncryptedBundle.__annotations__):
        raise ValueError("ENCRYPTED_FIELDS: exact bundle fields required")
    format_name = require_str(obj, "format")
    partition = require_str(obj, "partition")
    recipient = require_str(obj, "recipient")
    if format_name != "age-v1" or partition not in {"train-validation", "final"}:
        raise ValueError("ENCRYPTED_FORMAT: require age-v1 and an explicit data partition")
    if not recipient.startswith("age1") or len(recipient) != 62:
        raise ValueError("ENCRYPTED_RECIPIENT: require an X25519 public age recipient")
    manifest = decode_staged_file(obj["manifest"])
    files = tuple(decode_encrypted_file(item) for item in require_list(obj, "files"))
    plain_names = [item["plaintext"]["name"] for item in files]
    cipher_names = [item["ciphertext"]["name"] for item in files]
    if not files or len(set(plain_names)) != len(files):
        raise ValueError("ENCRYPTED_FILES: require distinct nonempty files")
    if set(plain_names) & set(cipher_names):
        raise ValueError("ENCRYPTED_FILES: plaintext and ciphertext names must not overlap")
    if manifest not in [item["plaintext"] for item in files]:
        raise ValueError("ENCRYPTED_MANIFEST: the exact plaintext manifest must be included")
    return EncryptedBundle(
        format=format_name,
        partition=partition,
        recipient=recipient,
        manifest=manifest,
        files=files,
    )


def encode_encrypted_bundle(value: EncryptedBundle) -> JSONObject:
    """Encode reproducibility metadata without key material or observation rows."""
    return {
        "format": value["format"],
        "partition": value["partition"],
        "recipient": value["recipient"],
        "manifest": encode_staged_file(value["manifest"]),
        "files": [encode_encrypted_file(item) for item in value["files"]],
    }


__all__ = [
    "EncryptedBundle",
    "EncryptedFile",
    "decode_encrypted_bundle",
    "decode_encrypted_file",
    "encode_encrypted_bundle",
    "encode_encrypted_file",
]
