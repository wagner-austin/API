"""Verified remote text artifacts shared by WRDS and HPC3 evidence readers."""

from __future__ import annotations

import hashlib
import shlex

from typing_extensions import TypedDict

from hpc3.core.digest import check_remote_digest, parse_remote_digest
from hpc3.core.remote import remote_digest, run_remote


class Artifact(TypedDict):
    """A remote text file copied back with its verified digest."""

    text: str
    sha256: str


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


__all__ = ["Artifact", "download"]
