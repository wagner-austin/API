"""Content-bound local evidence shared by WRDS research adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path


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
    path = directory / name
    if path.resolve().parent != directory.resolve():
        raise ValueError("WRDS_DATA_DRIFT: evidence symlink leaves the manifest directory")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"WRDS_DATA_DRIFT: {name} differs from the extract manifest")
    return payload
