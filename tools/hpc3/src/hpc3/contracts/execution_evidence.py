"""Strict native Slurm execution evidence, distinct from submitted job intent."""

from __future__ import annotations

from datetime import datetime

from platform_core.json_utils import JSONObject, require_dict, require_str
from typing_extensions import ReadOnly, TypedDict

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.status import JobStatus, decode_job_status, encode_job_status


class ExecutionEvidence(TypedDict):
    """Successful allocation accounting with UTC instants and a real node."""

    job: ReadOnly[JobStatus]
    exit_code: ReadOnly[str]
    started_at: ReadOnly[str]
    ended_at: ReadOnly[str]


def encode_execution_evidence(value: ExecutionEvidence) -> JSONObject:
    """Encode measured accounting.

    Args:
        value: Validated successful execution.

    Returns:
        JSON evidence object.
    """
    return {
        "job": encode_job_status(value["job"]),
        "exit_code": value["exit_code"],
        "started_at": value["started_at"],
        "ended_at": value["ended_at"],
    }


def decode_execution_evidence(value: JSONObject, cluster: ClusterFacts) -> ExecutionEvidence:
    """Require completed zero-exit accounting and ordered, known UTC instants.

    Args:
        value: Untrusted accounting document.
        cluster: Declared cluster partitions.

    Returns:
        Validated native execution evidence.

    Raises:
        ValueError: HPC_EVIDENCE_ACCOUNTING for incomplete or inconsistent accounting.
    """
    if set(value) != set(ExecutionEvidence.__annotations__):
        raise ValueError("HPC_EVIDENCE_ACCOUNTING: exact evidence fields required")
    job = decode_job_status(require_dict(value, "job"), cluster)
    code = require_str(value, "exit_code")
    start, end = require_str(value, "started_at"), require_str(value, "ended_at")
    if job["state"] != "COMPLETED" or code != "0:0" or not job["node_list"]:
        raise ValueError(
            "HPC_EVIDENCE_ACCOUNTING: require completed allocation, zero exit and node"
        )
    first, last = datetime.fromisoformat(start), datetime.fromisoformat(end)
    if first.tzinfo is not None or last.tzinfo is not None or first > last:
        raise ValueError("HPC_EVIDENCE_ACCOUNTING: require ordered scheduler UTC instants")
    return ExecutionEvidence(job=job, exit_code=code, started_at=start, ended_at=end)


__all__ = ["ExecutionEvidence", "decode_execution_evidence", "encode_execution_evidence"]
