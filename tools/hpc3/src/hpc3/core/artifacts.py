"""Verified remote text artifacts shared by WRDS and HPC3 evidence readers."""

from __future__ import annotations

import hashlib
import shlex
from collections.abc import Sequence
from pathlib import PurePosixPath

from platform_core.json_utils import JSONObject, require_str
from typing_extensions import ReadOnly, TypedDict

from hpc3.core.digest import check_remote_digest, parse_remote_digest
from hpc3.core.remote import make_directory, remote_digest, run_remote


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


def prepare_artifact_directories(host: str, artifacts: Sequence[str | None]) -> tuple[str, ...]:
    """Create the directory every job about to be queued will write its artifact into.

    A job that computes for its whole allocation and then cannot open its
    result for writing has spent the allocation for nothing. Measured
    2026-11-14 (board task dcdec7d2): shards 2 and 3 of array 59589599 ran
    ten minutes each and died on FileNotFoundError for
    ``/pub/wagnera3/coax-drone/uncertainty/893459e/``, which nothing had
    created; earlier sweeps worked only because it was made by hand.

    Args:
        host: Cluster SSH alias.
        artifacts: Each job's declared artifact, None for one that writes no
            file of its own.

    Returns:
        The directories created, each once, in first-seen order.

    Raises:
        AppError: When a directory cannot be created.
    """
    directories: list[str] = []
    for path in artifacts:
        if path is None:
            continue
        directory = str(PurePosixPath(path).parent)
        if directory not in directories:
            directories.append(directory)
    for directory in directories:
        make_directory(host, directory)
    return tuple(directories)


__all__ = [
    "Artifact",
    "decode_artifact",
    "download",
    "encode_artifact",
    "prepare_artifact_directories",
]
