"""Measure paired HPC3 executions from native accounting and verified artifacts."""

from __future__ import annotations

import re
import shlex
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)
from platform_core.run_record import (
    RunRecord,
    compare_run_records,
    decode_run_record,
    run_record_sidecar,
)

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.execution_evidence import (
    ExecutionEvidence,
    decode_execution_evidence,
    encode_execution_evidence,
)
from hpc3.contracts.ledger import LedgerEntry, encode_ledger_entry
from hpc3.contracts.status import encode_job_status
from hpc3.core import artifacts, remote
from hpc3.core.digest import check_remote_digest, parse_remote_digest, sha256_hex
from hpc3.core.logs import log_path
from hpc3.core.status import SACCT_FIELDS, parse_sacct_row, sacct_commands


class MeasuredRun(NamedTuple):
    """Internal shared result and its non-numerical execution evidence."""

    record: RunRecord
    report: JSONObject


def parse_checksums(text: str) -> dict[str, str]:
    """Decode an exact sha256sum manifest with absolute, unique file paths.

    Args:
        text: Frozen checksum file, acquired before evaluation.

    Returns:
        Remote path to expected content digest.

    Raises:
        ValueError: HPC_EVIDENCE_INPUTS for malformed, empty or repeated entries.
    """
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([a-f0-9]{64}) [ *](.+)", line)
        if match is None:
            raise ValueError("HPC_EVIDENCE_INPUTS: invalid checksum line")
        digest, path = match.groups()
        if (
            not PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or path in result
        ):
            raise ValueError("HPC_EVIDENCE_INPUTS: require unique absolute input paths")
        result[path] = digest
    if not result:
        raise ValueError("HPC_EVIDENCE_INPUTS: checksum manifest is empty")
    return result


def accounting(entry: LedgerEntry, cluster: ClusterFacts) -> ExecutionEvidence:
    """Read completed allocation evidence directly from Slurm in UTC.

    Args:
        entry: Unique recorded submission.
        cluster: Declared partitions.

    Returns:
        Native successful allocation evidence.

    Raises:
        ValueError: HPC_EVIDENCE_ACCOUNTING for missing/mismatched accounting.
        AppError: Remote command failure or malformed native fields.
    """
    if not entry["job_id"].isdecimal():
        raise ValueError("HPC_EVIDENCE_ACCOUNTING: numeric allocation id required")
    fields = list(SACCT_FIELDS)
    fields[1] = "JobName%200"
    fields.extend(["ExitCode", "Start", "End"])
    command = sacct_commands([entry["job_id"]], fields)[0]
    lines = remote.run_remote(entry["host"], "env TZ=UTC " + command).strip().splitlines()
    if len(lines) != 1 or len(lines[0].split("|")) != len(fields):
        raise ValueError("HPC_EVIDENCE_ACCOUNTING: require exactly one complete allocation row")
    values = lines[0].split("|")
    job = parse_sacct_row("|".join(values[: len(SACCT_FIELDS)]), cluster)
    if (job["job_id"], job["name"], job["partition"]) != (
        entry["job_id"],
        entry["name"],
        entry["partition"],
    ):
        raise ValueError("HPC_EVIDENCE_ACCOUNTING: allocation does not match ledger")
    return decode_execution_evidence(
        {
            "job": encode_job_status(job),
            "exit_code": values[-3],
            "started_at": values[-2],
            "ended_at": values[-1],
        },
        cluster,
    )


def _artifact_time(host: str, path: str, evidence: ExecutionEvidence) -> int:
    """Require the artifact's modification instant to fall inside this allocation.

    Args:
        host: Cluster SSH alias.
        path: Remote result or record.
        evidence: Measured start/end instants.

    Returns:
        Remote modification epoch seconds.

    Raises:
        ValueError: HPC_EVIDENCE_TIME for an artifact from outside this job.
    """
    instant = int(remote.run_remote(host, shlex.join(["stat", "-c", "%Y", "--", path])).strip())
    first = datetime.fromisoformat(evidence["started_at"]).replace(tzinfo=UTC).timestamp()
    last = datetime.fromisoformat(evidence["ended_at"]).replace(tzinfo=UTC).timestamp()
    if not first <= instant <= last:
        raise ValueError("HPC_EVIDENCE_TIME: artifact was not written during this allocation")
    return instant


def measure(entry: LedgerEntry, cluster: ClusterFacts, checksums: dict[str, str]) -> MeasuredRun:
    """Bind native accounting, input/runtime checks and benchmark result bytes.

    Args:
        entry: Ledger submission, including its declared artifact.
        cluster: Declared cluster.
        checksums: Frozen input, external entrypoint and image digests.

    Returns:
        Shared numerical record plus metadata-only evidence.

    Raises:
        ValueError: Missing artifact/runtime identity, checksum log or result lineage.
        AppError: Missing/altered remote file or failed remote command.
    """
    artifact = entry["artifact"]
    image = entry["image_digest"]
    if artifact is None or not image or image not in checksums.values():
        raise ValueError("HPC_EVIDENCE_IDENTITY: require artifact and checksummed runtime")
    evidence = accounting(entry, cluster)
    output = artifacts.download(entry["host"], artifact)
    record_path = str(run_record_sidecar(Path(artifact)))
    sidecar = artifacts.download(entry["host"], record_path)
    record = decode_run_record(load_json_str(sidecar["text"]))
    manifest = narrow_json_to_dict(load_json_str(output["text"]))
    payload_digest = sha256_hex(dump_json_str(require_list(manifest, "results")).encode())
    if record["fingerprint"]["image_digest"] != image or record["payload_digest"] != payload_digest:
        raise ValueError("HPC_EVIDENCE_RESULT: record does not bind result payload and runtime")
    if require_dict(manifest, "fingerprint") != require_dict(
        narrow_json_to_dict(load_json_str(sidecar["text"])),
        "fingerprint",
    ):
        raise ValueError("HPC_EVIDENCE_RESULT: manifest and record fingerprints differ")
    output_time = _artifact_time(entry["host"], artifact, evidence)
    record_time = _artifact_time(entry["host"], record_path, evidence)
    log = artifacts.download(entry["host"], log_path(entry))
    logged = set(log["text"].splitlines())
    for path, digest in checksums.items():
        actual = parse_remote_digest(remote.remote_digest(entry["host"], path), path)
        check_remote_digest(path, digest, actual)
        if f"{path}: OK" not in logged:
            raise ValueError("HPC_EVIDENCE_INPUTS: job log lacks a required compute checksum")
    return MeasuredRun(
        record,
        {
            "ledger": encode_ledger_entry(entry),
            "accounting": encode_execution_evidence(evidence),
            "result_sha256": output["sha256"],
            "record_sha256": sidecar["sha256"],
            "stdout_sha256": log["sha256"],
            "result_mtime": output_time,
            "record_mtime": record_time,
            "verified_files": dict(checksums),
            "compute_checksum_lines": len(checksums),
            "dataset": require_dict(manifest, "dataset"),
        },
    )


def replay_report(
    left: LedgerEntry, right: LedgerEntry, cluster: ClusterFacts, checksums: dict[str, str]
) -> JSONObject:
    """Measure two distinct jobs and compare their shared numerical records.

    Args:
        left: First ledger submission.
        right: Repeated ledger submission.
        cluster: Declared cluster.
        checksums: Same frozen inputs/code/image for both runs.

    Returns:
        Metadata-only report; mismatch is reported rather than labelled success.

    Raises:
        ValueError: HPC_EVIDENCE_PAIR for duplicate jobs or changed experiment identity.
    """
    first_identity = {
        key: value for key, value in left["experiment"].items() if key != "repetition"
    }
    second_identity = {
        key: value for key, value in right["experiment"].items() if key != "repetition"
    }
    if (
        left["job_id"] == right["job_id"]
        or left["host"] != right["host"]
        or first_identity != second_identity
        or not first_identity
    ):
        raise ValueError("HPC_EVIDENCE_PAIR: require distinct jobs of the same frozen experiment")
    first, second = measure(left, cluster, checksums), measure(right, cluster, checksums)
    comparison = compare_run_records(first.record, second.record, ())
    if comparison["kind"] != "compared":
        raise ValueError("HPC_EVIDENCE_PAIR: numerical environments are not comparable")
    deltas = comparison["deltas"]
    identical = (
        bool(deltas)
        and not comparison["unmatched"]
        and all(delta["difference"] == 0 for delta in deltas)
        and first.record["payload_digest"] == second.record["payload_digest"]
        and first.report["result_sha256"] == second.report["result_sha256"]
    )
    return {
        "kind": "HPC3_REPLAY_EVIDENCE",
        "left": first.report,
        "right": second.report,
        "comparison": "identical" if identical else "different",
        "shared_observations": len(deltas),
        "unmatched": list(comparison["unmatched"]),
    }


__all__ = ["MeasuredRun", "accounting", "measure", "parse_checksums", "replay_report"]
