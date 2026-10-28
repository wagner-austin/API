"""Verified remote text artifacts shared by WRDS and HPC3 evidence readers."""

from __future__ import annotations

import hashlib
import shlex

from platform_core.json_utils import JSONObject, require_str
from typing_extensions import ReadOnly, TypedDict

from hpc3.core.digest import check_remote_digest, parse_remote_digest
from hpc3.core.remote import remote_digest, run_remote


class Artifact(TypedDict):
    """A remote text file copied back with its verified digest."""

    text: ReadOnly[str]
    sha256: ReadOnly[str]


def encode_artifact(value: Artifact) -> JSONObject:
    """Encode verified content and its identity.

    Args:
        value: Verified remote text.

    Returns:
        Artifact JSON document.
    """
    return {"text": value["text"], "sha256": value["sha256"]}


def decode_artifact(value: JSONObject) -> Artifact:
    """Validate an artifact's content against its declared digest.

    Args:
        value: Untrusted artifact document.

    Returns:
        Immutable verified content.

    Raises:
        ValueError: HPC_ARTIFACT_FIELDS for unexpected or missing fields.
        AppError: Declared digest does not identify the text.
    """
    if set(value) != {"text", "sha256"}:
        raise ValueError("HPC_ARTIFACT_FIELDS: exact text and sha256 fields required")
    text, digest = require_str(value, "text"), require_str(value, "sha256")
    check_remote_digest("artifact", digest, hashlib.sha256(text.encode()).hexdigest())
    return Artifact(text=text, sha256=digest)


def download(host: str, path: str) -> Artifact:
    """Copy a job's text artifact back and confirm it matches the remote file.

    Args:
        host: Cluster SSH alias.
        path: Absolute artifact path.

    Returns:
        The text and its sha256.

    Raises:
        AppError: When the read fails or the remote digest differs.
    """
    text = run_remote(host, shlex.join(["cat", "--", path]))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    check_remote_digest(path, digest, parse_remote_digest(remote_digest(host, path), path))
    return Artifact(text=text, sha256=digest)


__all__ = ["Artifact", "decode_artifact", "download", "encode_artifact"]
