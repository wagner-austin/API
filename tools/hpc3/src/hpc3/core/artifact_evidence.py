"""Bind every member of a recorded job array to its scheduler record and its artifact's bytes.

A sweep's results are usually copied off the cluster and committed somewhere
else, and nothing then ties those copies to the jobs that wrote them: the
run document says what was submitted and the ledger what was recorded, but
neither proves the committed bytes are the ones a completed allocation
produced. This measures that tie for one array. Each member must have
completed with exit 0 in native accounting, its artifact must hash on the
cluster to exactly the bytes of the same-named local file, and its
modification instant must fall inside that member's allocation.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from pathlib import PurePosixPath

from platform_core.json_utils import JSONObject, JSONValue

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.execution_evidence import encode_execution_evidence
from hpc3.contracts.experiment import encode_experiment
from hpc3.contracts.ledger import LedgerEntry
from hpc3.core import _test_hooks, remote
from hpc3.core.digest import check_remote_digest, parse_remote_digest, sha256_hex
from hpc3.core.execution_evidence import accounting, artifact_time


def array_members(entries: Sequence[LedgerEntry], array: str) -> list[LedgerEntry]:
    """Every recorded member of one array, in member order.

    Args:
        entries: Decoded ledger entries.
        array: The array's numeric job id, without a member suffix.

    Returns:
        Each entry whose id is ``<array>_<member>``, ordered by member number.

    Raises:
        ValueError: HPC_EVIDENCE_ARRAY for a non-numeric array id, an array the
            ledger records no member of, or a member recorded twice.
    """
    if not array.isdecimal():
        raise ValueError("HPC_EVIDENCE_ARRAY: numeric array id required")
    members = [
        entry
        for entry in entries
        if entry["job_id"].startswith(f"{array}_") and entry["job_id"][len(array) + 1 :].isdecimal()
    ]
    if not members:
        raise ValueError(f"HPC_EVIDENCE_ARRAY: the ledger records no member of array {array}")
    ids = [entry["job_id"] for entry in members]
    if len(set(ids)) != len(ids):
        raise ValueError(f"HPC_EVIDENCE_ARRAY: a member of array {array} is recorded twice")

    def member_number(entry: LedgerEntry) -> int:
        return int(entry["job_id"][len(array) + 1 :])

    return sorted(members, key=member_number)


def require_artifact(entry: LedgerEntry) -> str:
    """The artifact a member was told to write.

    Args:
        entry: The member's ledger entry.

    Returns:
        Its artifact's absolute cluster path.

    Raises:
        ValueError: HPC_EVIDENCE_IDENTITY for a member that names no artifact.
    """
    artifact = entry["artifact"]
    if artifact is None:
        raise ValueError(f"HPC_EVIDENCE_IDENTITY: member {entry['job_id']} names no artifact")
    return artifact


def measure_member(
    entry: LedgerEntry, artifact: str, cluster: ClusterFacts, local: pathlib.Path
) -> JSONObject:
    """Bind one member's completed allocation to its artifact and a local copy.

    Args:
        entry: The member's ledger entry.
        artifact: Its artifact, from :func:`require_artifact`.
        cluster: Declared cluster.
        local: The local file that claims to be the artifact.

    Returns:
        The member's id, name, experiment, image, artifact path, digest and
        modification instant, the local path, and its native accounting.

    Raises:
        ValueError: HPC_EVIDENCE_ACCOUNTING for an allocation that did not
            complete cleanly, HPC_EVIDENCE_TIME for an artifact written outside it.
        AppError: A remote command failed, or the local bytes differ from the
            artifact's.
        OSError: The local file cannot be read.
    """
    evidence = accounting(entry, cluster)
    remote_sha = parse_remote_digest(remote.remote_digest(entry["host"], artifact), artifact)
    local_sha = sha256_hex(_test_hooks.read_bytes(local))
    check_remote_digest(str(local), local_sha, remote_sha)
    return {
        "job_id": entry["job_id"],
        "name": entry["name"],
        "experiment": encode_experiment(entry["experiment"]),
        "image_digest": entry["image_digest"],
        "artifact": artifact,
        "artifact_sha256": remote_sha,
        "artifact_mtime": artifact_time(entry["host"], artifact, evidence),
        "local_path": str(local),
        "accounting": encode_execution_evidence(evidence),
    }


def artifact_report(
    entries: Sequence[LedgerEntry], array: str, cluster: ClusterFacts, local_dir: pathlib.Path
) -> JSONObject:
    """Measure every member of one array against same-named files in a local directory.

    Args:
        entries: Decoded ledger entries.
        array: The array's numeric job id.
        cluster: Declared cluster.
        local_dir: Directory holding one file per member, named as its artifact.

    Returns:
        The array id, the local directory, and each member's measurement.

    Raises:
        ValueError: HPC_EVIDENCE_ARRAY for an unknown array, or two members
            whose artifacts share a file name, or a member's own refusal.
        AppError: A remote command failed or a local copy differs.
        OSError: A local file cannot be read.
    """
    members = array_members(entries, array)
    artifacts = [require_artifact(entry) for entry in members]
    names = [PurePosixPath(artifact).name for artifact in artifacts]
    if len(set(names)) != len(names):
        raise ValueError(f"HPC_EVIDENCE_ARRAY: two members of array {array} share an artifact name")
    measured: list[JSONValue] = [
        measure_member(entry, artifact, cluster, local_dir / name)
        for entry, artifact, name in zip(members, artifacts, names, strict=True)
    ]
    return {"array": array, "local_dir": str(local_dir), "members": measured}


__all__ = ["array_members", "artifact_report", "measure_member", "require_artifact"]
