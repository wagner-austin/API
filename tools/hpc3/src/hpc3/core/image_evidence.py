"""Verify a recorded image build against native accounting and frozen build files."""

from __future__ import annotations

from pathlib import PurePosixPath

from platform_core.json_utils import JSONObject

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.execution_evidence import encode_execution_evidence
from hpc3.contracts.ledger import LedgerEntry, encode_ledger_entry
from hpc3.core import artifacts, remote
from hpc3.core.digest import check_remote_digest, parse_remote_digest
from hpc3.core.execution_evidence import accounting, artifact_time
from hpc3.core.image_layout import (
    COMMIT_NAME,
    DEFINITION_NAME,
    REQUIREMENTS_NAME,
    SBATCH_NAME,
    SELFCHECK_NAME,
)


def measure_build(
    entry: LedgerEntry, cluster: ClusterFacts, checksums: dict[str, str]
) -> JSONObject:
    """Read the native build, image and recipe, refusing incomplete verification.

    Args:
        entry: Existing image-build ledger row.
        cluster: Declared native cluster.
        checksums: Frozen absolute paths and digests, including image and build log.

    Returns:
        Metadata-only accounting, verified file identities and verification counts.

    Raises:
        ValueError: HPC_IMAGE_EVIDENCE for a non-build, incomplete pins or log.
        AppError: Failed remote read or changed artifact.
    """
    image = entry["artifact"]
    directory = PurePosixPath(entry["log_dir"])
    if (
        entry["experiment"].get("kind") != "image-build"
        or image is None
        or PurePosixPath(image).parent != directory
    ):
        raise ValueError("HPC_IMAGE_EVIDENCE: require a recorded image build and its artifact")
    log_path = str(directory / f"build-{entry['job_id']}.out")
    required = {
        str(directory / name)
        for name in (
            SBATCH_NAME,
            "build.sh",
            DEFINITION_NAME,
            REQUIREMENTS_NAME,
            SELFCHECK_NAME,
            COMMIT_NAME,
        )
    } | {image, log_path}
    if not required.issubset(checksums):
        raise ValueError("HPC_IMAGE_EVIDENCE: pin image, log and every generated build file")
    evidence = accounting(entry, cluster)
    for path, digest in checksums.items():
        actual = parse_remote_digest(remote.remote_digest(entry["host"], path), path)
        check_remote_digest(path, digest, actual)
    log = artifacts.download(entry["host"], log_path)
    check_remote_digest(log_path, checksums[log_path], log["sha256"])
    lines = log["text"].splitlines()
    required_lines = {
        f"{checksums[image]}  {PurePosixPath(image).name}",
        "re-verify exit 0",
        "--- smoke 1/1: /opt/env/bin/python -m pip check ---",
        "No broken requirements found.",
        "smoke 1 OK",
        "exit      0",
    }
    if lines.count("image self-check OK") != 2 or not required_lines.issubset(lines):
        raise ValueError("HPC_IMAGE_EVIDENCE: require both selfchecks and dependency smoke success")
    return {
        "kind": "HPC3_IMAGE_BUILD_EVIDENCE",
        "ledger": encode_ledger_entry(entry),
        "accounting": encode_execution_evidence(evidence),
        "verified_files": dict(checksums),
        "image_mtime": artifact_time(entry["host"], image, evidence),
        "log_mtime": artifact_time(entry["host"], log_path, evidence),
        "selfcheck_passes": 2,
        "dependency_check": "passed",
    }


__all__ = ["measure_build"]
