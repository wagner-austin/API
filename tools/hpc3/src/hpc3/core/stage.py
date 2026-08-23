"""Placing manifest-described files on the cluster, proven on both sides.

The sequence is fixed and every step of it is load-bearing:

1. Read each file locally and verify its length and digest against the
   manifest. This catches an emitter that produced the wrong corpus -- the
   733-page-versus-773-page case -- before any bytes cross the network.
2. Send the exact bytes.
3. Digest the file again ON THE CLUSTER and compare. This is the only step
   that proves what the job will read, and it is the reason a local-only
   check is insufficient.

There is no partial success. If any file fails, the operation raises and the
caller learns which file and why; it does not report how many succeeded,
because a corpus staged in part is not a corpus.
"""

from __future__ import annotations

import pathlib

from hpc3.contracts.provenance import format_provenance
from hpc3.contracts.stage import StagedFile, StageManifest
from hpc3.core import audit, digest, remote


def stage_one(host: str, source_dir: pathlib.Path, destination: str, staged: StagedFile) -> str:
    """Place one manifest-described file and prove it arrived intact.

    Args:
        host: SSH destination.
        source_dir: Local directory holding the file.
        destination: Absolute directory on the cluster receiving it.
        staged: Manifest record naming the file and the bytes expected.

    Returns:
        The file's absolute path on the cluster.

    Raises:
        AppError: With ``MANIFEST_FILE_MISSING`` if the local file is
            absent, ``DIGEST_MISMATCH`` if the local bytes or the arrived
            bytes differ from the manifest, or ``REMOTE_COMMAND_FAILED`` if
            the write or the remote digest could not run.
    """
    payload = digest.read_and_verify(source_dir, staged)
    remote_path = f"{destination}/{staged['name']}"
    remote.put_bytes(host, remote_path, payload)
    output = remote.remote_digest(host, remote_path)
    arrived = digest.parse_remote_digest(output, staged["name"])
    digest.check_remote_digest(staged["name"], staged["sha256"], arrived)
    return remote_path


def stage_manifest(host: str, source_dir: pathlib.Path, manifest: StageManifest) -> list[str]:
    """Place every file a manifest describes, verifying each on both sides.

    Args:
        host: SSH destination.
        source_dir: Local directory holding the files.
        manifest: What to place and where.

    Returns:
        Absolute cluster paths of the placed files, in manifest order.

    Raises:
        AppError: On the first file that cannot be verified or transferred.
            Earlier files remain on the cluster; they are individually
            correct, and the caller is told which file stopped the run rather
            than being handed a partial success to interpret.
    """
    remote.make_directory(host, manifest["destination"])
    placed = [
        stage_one(host, source_dir, manifest["destination"], staged) for staged in manifest["files"]
    ]
    # Logged only after every file verified on the cluster: an event emitted
    # per file would record a partial stage as a sequence of successes.
    audit.files_staged(
        host=host,
        destination=manifest["destination"],
        count=len(placed),
        provenance=format_provenance(manifest["provenance"]),
    )
    return placed


__all__ = ["stage_manifest", "stage_one"]
