"""Bind every artifact of a recorded sweep to the job that completed it and to its bytes.

A sweep's results are usually copied off the cluster and committed somewhere
else, and nothing then ties those copies to the jobs that wrote them: the
run document says what was submitted and the ledger what was recorded, but
neither proves the committed bytes are the ones a completed allocation
produced. This measures that tie across every array a sweep ran in, the
first submission and any hpc3-campaign resubmissions. Each artifact must
have exactly one declaring member that completed with exit 0 in native
accounting and none still running, its bytes must hash on the cluster to
exactly the same-named local file, and its modification instant must fall
inside that member's allocation.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from pathlib import PurePosixPath

from platform_core.json_utils import JSONObject, JSONValue, require_dict, require_str

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.execution_evidence import (
    ExecutionEvidence,
    decode_execution_evidence,
    encode_execution_evidence,
)
from hpc3.contracts.experiment import encode_experiment
from hpc3.contracts.ledger import LedgerEntry
from hpc3.core import _test_hooks, remote
from hpc3.core.digest import check_remote_digest, parse_remote_digest, sha256_hex
from hpc3.core.execution_evidence import allocation_document, artifact_time


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


#: States in which a job may still write its artifact. A group holding one is
#: not settled, whatever another member of it did.
LIVE_STATES = frozenset({"PENDING", "RUNNING", "SUSPENDED", "COMPLETING"})


def measure_member(
    entry: LedgerEntry, artifact: str, evidence: ExecutionEvidence, local: pathlib.Path
) -> JSONObject:
    """Bind one member's completed allocation to its artifact and a local copy.

    Args:
        entry: The member's ledger entry.
        artifact: Its artifact, from :func:`require_artifact`.
        evidence: Its completed, zero-exit accounting.
        local: The local file that claims to be the artifact.

    Returns:
        The member's id, name, experiment, image, artifact path, digest and
        modification instant, the local path, and its native accounting.

    Raises:
        ValueError: HPC_EVIDENCE_TIME for an artifact written outside the
            allocation.
        AppError: A remote command failed, or the local bytes differ from the
            artifact's.
        OSError: The local file cannot be read.
    """
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


def settled_member(
    artifact: str, group: Sequence[tuple[LedgerEntry, JSONObject]], cluster: ClusterFacts
) -> tuple[LedgerEntry, ExecutionEvidence]:
    """The one member that completed an artifact, among every member declaring it.

    A sweep converged by hpc3-campaign declares one artifact in each array
    that ran it: a first member preempted or failed, then the resubmission
    that completed. Exactly one completed member settles it.

    Args:
        artifact: The artifact's cluster path.
        group: Every member declaring it, with its allocation document.
        cluster: Declared cluster.

    Returns:
        The completed member and its accounting.

    Raises:
        ValueError: HPC_EVIDENCE_ARRAY when a member declaring it may still be
            writing it, when none completed with exit 0, or when two did.
    """
    live = [entry["job_id"] for entry, document in group if _state(document) in LIVE_STATES]
    if live:
        raise ValueError(f"HPC_EVIDENCE_ARRAY: {artifact} is still being written by {live}")
    completed = [
        (entry, document)
        for entry, document in group
        if _state(document) == "COMPLETED" and document["exit_code"] == "0:0"
    ]
    if not completed:
        raise ValueError(f"HPC_EVIDENCE_ARRAY: no member that declares {artifact} completed")
    if len(completed) > 1:
        ids = [entry["job_id"] for entry, _ in completed]
        raise ValueError(f"HPC_EVIDENCE_ARRAY: {artifact} was completed twice, by {ids}")
    entry, document = completed[0]
    return entry, decode_execution_evidence(document, cluster)


def _state(document: JSONObject) -> str:
    """A member's scheduler state from its allocation document.

    Args:
        document: From :func:`~hpc3.core.execution_evidence.allocation_document`.

    Returns:
        The state as sacct spells it.
    """
    return require_str(require_dict(document, "job"), "state")


def artifact_report(
    entries: Sequence[LedgerEntry],
    arrays: Sequence[str],
    cluster: ClusterFacts,
    local_dir: pathlib.Path,
) -> JSONObject:
    """Measure a sweep's artifacts, across every array it ran in, against local copies.

    Args:
        entries: Decoded ledger entries.
        arrays: The numeric ids of the sweep's arrays, the first submission
            and any campaign resubmissions, each once.
        cluster: Declared cluster.
        local_dir: Directory holding one file per artifact, named as it is.

    Returns:
        The arrays, the local directory, each artifact's settled member
        measured, and each member it superseded with its state and exit code.

    Raises:
        ValueError: HPC_EVIDENCE_ARRAY for no array, a repeated or unknown
            array, two artifacts sharing a file name, or an artifact
            :func:`settled_member` refuses; or a member's own refusal.
        AppError: A remote command failed or a local copy differs.
        OSError: A local file cannot be read.
    """
    if not arrays or len(set(arrays)) != len(arrays):
        raise ValueError("HPC_EVIDENCE_ARRAY: name each of the sweep's arrays once")
    members = [entry for array in arrays for entry in array_members(entries, array)]
    groups: dict[str, list[LedgerEntry]] = {}
    for entry in members:
        groups.setdefault(require_artifact(entry), []).append(entry)
    names = [PurePosixPath(artifact).name for artifact in groups]
    if len(set(names)) != len(names):
        raise ValueError("HPC_EVIDENCE_ARRAY: two artifacts of the sweep share a file name")
    measured: list[JSONValue] = []
    superseded: list[JSONValue] = []
    for (artifact, group), name in zip(groups.items(), names, strict=True):
        documents = [(entry, allocation_document(entry, cluster)) for entry in group]
        settled, evidence = settled_member(artifact, documents, cluster)
        measured.append(measure_member(settled, artifact, evidence, local_dir / name))
        superseded.extend(
            {
                "job_id": entry["job_id"],
                "artifact": artifact,
                "state": _state(document),
                "exit_code": document["exit_code"],
            }
            for entry, document in documents
            if entry is not settled
        )
    return {
        "arrays": list(arrays),
        "local_dir": str(local_dir),
        "members": measured,
        "superseded": superseded,
    }


__all__ = [
    "LIVE_STATES",
    "array_members",
    "artifact_report",
    "measure_member",
    "require_artifact",
    "settled_member",
]
