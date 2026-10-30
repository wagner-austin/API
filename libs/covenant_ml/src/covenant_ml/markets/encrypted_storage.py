"""Authenticated bundles and temporary plaintext for trusted evaluator processes.

This module does not grant access or deliver keys. Its caller must obtain the
proper partition's identity through evaluator-owned authorization. A private
directory does not isolate processes sharing the evaluator's OS identity.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from hashlib import file_digest, sha256
from io import BufferedIOBase
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.stage_manifest import StagedFile, StageManifest

from ..evaluation.evidence import require_evidence_path
from . import _test_hooks_encryption as hooks
from .encrypted_records import (
    EncryptedBundle,
    EncryptedFile,
    decode_encrypted_bundle,
    encode_encrypted_bundle,
)


def _file_record(stream: BufferedIOBase, name: str) -> StagedFile:
    """Hash an open file without materializing its contents in memory."""
    size = stream.seek(0, 2)
    stream.seek(0)
    digest = file_digest(stream, "sha256")
    stream.seek(0)
    return StagedFile(name=name, sha256=digest.hexdigest(), size_bytes=size)


class _BoundedWriter:
    """Stop even authenticated ciphertext exceeding its declared plaintext size."""

    def __init__(self, stream: BinaryIO, limit: int) -> None:
        self.stream = stream
        self.remaining = limit

    def write(self, data: bytes) -> int:
        """Refuse the oversized chunk before writing any of it."""
        if len(data) > self.remaining:
            raise ValueError("ENCRYPTED_LIMIT: decrypted output exceeds frozen size")
        written = self.stream.write(data)
        self.remaining -= written
        return written

    def flush(self) -> None:
        """Flush verified writes through the open descriptor."""
        self.stream.flush()


def _verify(stream: BufferedIOBase, expected: StagedFile) -> None:
    """Require both the length and SHA-256 of the same open descriptor."""
    if _file_record(stream, expected["name"]) != expected:
        raise ValueError("ENCRYPTED_DIGEST: file differs from its frozen identity")


def encrypt_bundle(
    directory: Path,
    destination: Path,
    *,
    files: tuple[StagedFile, ...],
    manifest: StagedFile,
    recipient: str,
    partition: str,
) -> EncryptedBundle:
    """Encrypt frozen source files and publish metadata only after every file succeeds.

    Args:
        directory: Evaluator-owned extraction directory.
        destination: New exclusive ciphertext directory.
        files: Exact source identities, including the extraction manifest.
        manifest: Extraction manifest identity.
        recipient: Public X25519 recipient; extraction needs no decryption key.
        partition: Explicit train-validation or final partition.

    Returns:
        Metadata whose encoded digest must be frozen before staging or admission.

    Raises:
        ValueError: Invalid identities, paths, partition or changed source bytes.
        OSError: File access fails or destination already exists.
        Exception: pyrage rejects the recipient or encryption fails; no metadata
            is published, and no plaintext is written into the destination.
    """
    public = hooks.x25519.Recipient.from_str(recipient)
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    encrypted: list[EncryptedFile] = []
    for plain in files:
        cipher_name = plain["name"] + ".age"
        with require_evidence_path(directory, plain["name"]).open("rb") as source:
            _verify(source, plain)
            with (destination / cipher_name).open("x+b") as target:
                hooks.backend.encrypt_io(source, target, [public])
                target.flush()
                encrypted.append(
                    EncryptedFile(plaintext=plain, ciphertext=_file_record(target, cipher_name))
                )
    bundle = EncryptedBundle(
        format="age-v1",
        partition=partition,
        recipient=recipient,
        manifest=manifest,
        files=tuple(encrypted),
    )
    decode_encrypted_bundle(encode_encrypted_bundle(bundle))
    with (destination / "bundle.json").open("x", encoding="utf-8") as output:
        output.write(dump_json_str(encode_encrypted_bundle(bundle)))
    return bundle


def encrypted_stage_manifest(
    bundle: EncryptedBundle, path: Path, destination: str
) -> StageManifest:
    """Use the existing staging contract for ciphertext and pinned bundle metadata.

    Args:
        bundle: Completed encrypted bundle.
        path: Its published metadata file.
        destination: Cluster destination for encrypted files only.

    Returns:
        Existing transport contract with partition and plaintext-manifest lineage.

    Raises:
        ValueError: Metadata bytes differ from the supplied bundle.
        OSError: Metadata cannot be read.
    """
    payload = path.read_bytes()
    if payload != dump_json_str(encode_encrypted_bundle(bundle)).encode():
        raise ValueError("ENCRYPTED_MANIFEST: supplied bundle differs from published metadata")
    record = StagedFile(name=path.name, sha256=sha256(payload).hexdigest(), size_bytes=len(payload))
    return StageManifest(
        destination=destination,
        files=[record, *(item["ciphertext"] for item in bundle["files"])],
        provenance={
            "encrypted_bundle": record["sha256"],
            "plaintext_manifest": bundle["manifest"]["sha256"],
            "partition": bundle["partition"],
        },
    )


def read_encrypted_bundle(path: Path, expected_digest: str) -> EncryptedBundle:
    """Decode only metadata bytes matching the evaluator's frozen digest."""
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != expected_digest:
        raise ValueError("ENCRYPTED_MANIFEST: bundle differs from admission")
    return decode_encrypted_bundle(load_json_str(payload.decode("utf-8")))


@contextmanager
def decrypted_bundle(
    path: Path,
    expected_digest: str,
    *,
    partition: str,
    identity: hooks.AgeIdentity,
    scratch: Path,
    max_plaintext_bytes: int,
) -> Generator[Path, None, None]:
    """Yield verified plaintext only after all files authenticate; remove it on exit.

    Args:
        path: Staged encrypted metadata.
        expected_digest: Bundle SHA-256 from trusted admission, not from this file.
        partition: Partition authorized by the evaluator's access controls.
        identity: In-memory identity obtained outside agent-visible files.
        scratch: Trusted job-local scratch, never a shared staging directory.
        max_plaintext_bytes: Explicit per-job limit before any decryption starts.

    Yields:
        Verified extraction manifest inside a new mode-0700 directory.

    Raises:
        ValueError: Changed metadata, wrong partition/identity, size limit or checksum.
        OSError: Storage failure; temporary plaintext is removed on unwinding.
        Exception: Authentication failure propagates; no plaintext is yielded.

    SIGKILL cannot run Python cleanup. The cluster must reclaim job-local scratch
    after allocation termination; this API does not claim secure physical erasure.
    """
    bundle = read_encrypted_bundle(path, expected_digest)
    if bundle["partition"] != partition or str(identity.to_public()) != bundle["recipient"]:
        raise ValueError("ENCRYPTED_ACCESS: partition or identity is not authorized")
    if sum(item["plaintext"]["size_bytes"] for item in bundle["files"]) > max_plaintext_bytes:
        raise ValueError("ENCRYPTED_LIMIT: plaintext exceeds the job's storage limit")
    with TemporaryDirectory(prefix="research-plaintext-", dir=scratch) as temporary:
        private = Path(temporary)
        for item in bundle["files"]:
            plain, cipher = item["plaintext"], item["ciphertext"]
            with require_evidence_path(path.parent, cipher["name"]).open("rb") as source:
                _verify(source, cipher)
                with (private / plain["name"]).open("x+b") as target:
                    bounded = _BoundedWriter(target, plain["size_bytes"])
                    hooks.backend.decrypt_io(source, bounded, [identity])
                    bounded.flush()
                    _verify(target, plain)
        yield private / bundle["manifest"]["name"]


__all__ = [
    "decrypted_bundle",
    "encrypt_bundle",
    "encrypted_stage_manifest",
    "read_encrypted_bundle",
]
