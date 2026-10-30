"""Content-bound local evidence shared by WRDS research adapters."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path


def require_private_directory(path: Path) -> None:
    """Require the Linux evaluator's actual private-directory permissions.

    Args:
        path: Evaluator-owned directory to check before opening keys or plaintext.

    Raises:
        ValueError: WRDS_PRIVATE_STORAGE for symlinks or permissions other than 0700.
        OSError: The directory cannot be inspected.

    Windows ACLs are not POSIX modes; this contract does not certify them.
    """
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("WRDS_PRIVATE_STORAGE: require a non-symlink mode-0700 directory")


def require_evidence_digest(value: str) -> str:
    """Require a complete lowercase SHA256, raising WRDS_DATA_DIGEST otherwise."""
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("WRDS_DATA_DIGEST: expected a lowercase SHA256")
    return value


def require_evidence_filename(value: str) -> str:
    """Require a plain sibling name, raising WRDS_DATA_PATH for traversal."""
    if not value or value in {".", ".."} or any(c in value for c in "/\\\n\r\x00"):
        raise ValueError("WRDS_DATA_PATH: evidence must name a sibling file")
    return value


def require_evidence_path(directory: Path, name: str) -> Path:
    """Resolve a safe sibling path, refusing symlinks outside its evidence directory."""
    path = directory / require_evidence_filename(name)
    if path.resolve().parent != directory.resolve():
        raise ValueError("WRDS_DATA_DRIFT: evidence symlink leaves the manifest directory")
    return path


def read_evidence(directory: Path, name: str, expected: str) -> bytes:
    """Read once and verify precisely the bytes consumed.

    Args:
        directory: Manifest directory.
        name: Validated sibling file name.
        expected: Validated lowercase SHA256.

    Returns:
        Verified original bytes, without a second read after verification.

    Raises:
        ValueError: WRDS_DATA_DRIFT for changed or escaping evidence.
        OSError: Missing or unreadable evidence.
    """
    path = require_evidence_path(directory, name)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"WRDS_DATA_DRIFT: {name} differs from the extract manifest")
    return payload
