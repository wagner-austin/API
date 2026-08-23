"""Digest computation and the verification that makes staging meaningful.

The digest is not a transfer checksum. It is the identity of the bytes a run
is entitled to read, carried from the manifest that recorded them through to
the file the job opens. Verifying it locally proves the emitter produced the
right file; verifying it again on the cluster proves that file is what
arrived. Only the second one answers the question the run actually depends on.
"""

from __future__ import annotations

import hashlib
import pathlib

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.stage import StagedFile
from hpc3.core import _test_hooks


def sha256_hex(payload: bytes) -> str:
    """Digest bytes.

    Args:
        payload: Bytes to digest.

    Returns:
        Lowercase hex sha256, the form every manifest in this package records.
    """
    return hashlib.sha256(payload).hexdigest()


def read_and_verify(source_dir: pathlib.Path, staged: StagedFile) -> bytes:
    """Read a file and prove it is the one the manifest describes.

    Size is checked before the digest so a truncated transfer reports its
    actual length rather than an opaque hash mismatch, which tells the reader
    whether the file was cut short or corrupted in place.

    Args:
        source_dir: Directory holding the file.
        staged: Manifest record naming the file and the bytes expected.

    Returns:
        The file's bytes, verified against the record.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.MANIFEST_FILE_MISSING` if the
            manifest names a file that is not present, or with
            :attr:`~platform_core.errors.Hpc3ErrorCode.DIGEST_MISMATCH` if the file's
            length or digest differs from the record. A mismatch is never
            recoverable here: the caller asked for specific bytes and these
            are not them.
    """
    path = source_dir / staged["name"]
    if not _test_hooks.file_exists(path):
        raise AppError(
            Hpc3ErrorCode.MANIFEST_FILE_MISSING,
            f"Manifest names {staged['name']!r} but no such file exists under {source_dir}.",
        )

    payload = _test_hooks.read_bytes(path)
    if len(payload) != staged["size_bytes"]:
        raise AppError(
            Hpc3ErrorCode.DIGEST_MISMATCH,
            f"{staged['name']!r} is {len(payload)} bytes, manifest says "
            f"{staged['size_bytes']}. The file is truncated or is a different file.",
        )

    actual = sha256_hex(payload)
    if actual != staged["sha256"]:
        raise AppError(
            Hpc3ErrorCode.DIGEST_MISMATCH,
            f"{staged['name']!r} digests to {actual}, manifest says {staged['sha256']}. "
            "Same length, different contents: this is the wrong file, not a truncated one.",
        )
    return payload


def parse_remote_digest(output: str, name: str) -> str:
    """Read a digest out of ``sha256sum`` output.

    Args:
        output: The command's standard output, in ``sha256sum`` format:
            digest, whitespace, filename.
        name: Filename the digest was requested for, used in error messages.

    Returns:
        The lowercase hex digest.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.REMOTE_COMMAND_FAILED` if the
            output holds no digest-shaped first token. Treating unparsable
            output as a mismatch would blame the file for a broken command.
    """
    head = output.strip().split(maxsplit=1)
    if head == [] or len(head[0]) != 64:
        raise AppError(
            Hpc3ErrorCode.REMOTE_COMMAND_FAILED,
            f"sha256sum for {name!r} produced no digest; got {output.strip()!r}.",
        )
    return head[0]


def check_remote_digest(name: str, expected: str, actual: str) -> str:
    """Compare a digest computed on the cluster against the manifest.

    Returns the digest rather than None so the verified value is what flows
    onward, matching the rest of this package: ``read_and_verify`` returns the
    bytes it verified and ``parse_remote_digest`` returns the digest it read.
    A validator that returns nothing can only be tested by asserting it did
    not raise, which asserts almost nothing.

    Args:
        name: Filename being verified, used in the error message.
        expected: Digest the manifest records.
        actual: Digest computed on the cluster.

    Returns:
        The verified digest, equal to both arguments.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.DIGEST_MISMATCH` if they
            differ. Reaching here means the bytes changed in transit or the
            destination already held a different file of the same name; both
            make the run void, so neither is retried.
    """
    if actual != expected:
        raise AppError(
            Hpc3ErrorCode.DIGEST_MISMATCH,
            f"{name!r} digests to {actual} on the cluster but the manifest says {expected}. "
            "The bytes that arrived are not the bytes the run is entitled to.",
        )
    return actual


__all__ = ["check_remote_digest", "parse_remote_digest", "read_and_verify", "sha256_hex"]
