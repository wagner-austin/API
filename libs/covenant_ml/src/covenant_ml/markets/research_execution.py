"""Durable admission-to-HPC3 execution through the maintained cluster commands."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_str,
)

from . import _test_hooks
from .encrypted_records import decode_encrypted_bundle
from .research_cluster import ResearchCluster, attempt_identity, prepare_bundle
from .research_journal import ResearchAttempt, research_journal
from .research_records import decode_research_proposal
from .research_trial_records import ResearchTrial, encode_research_trial
from .validation import parse_timestamp, read_document


def trial_event(
    path: Path, trial: ResearchTrial, action: str, extra: JSONObject
) -> ResearchAttempt:
    """Commit a lifecycle transition under the existing journal's writer lock.

    Args:
        path: Admission journal.
        trial: Frozen admitted request.
        action: Transition name.
        extra: Transition-specific measured evidence.

    Returns:
        Persisted, replay-validated attempt state.
    """
    event: JSONObject = {
        "action": action,
        "number": trial["number"],
        "observed_ms": parse_timestamp(_test_hooks.clock()),
    }
    event.update(extra)
    return research_journal(trial["policy"]).transact(path, lambda _: event)[trial["number"] - 1]


def require_trial_attempt(path: Path, trial: ResearchTrial) -> ResearchAttempt:
    """Require the admitted proposal's typed values before any external operation.

    Args:
        path: Admission journal.
        trial: Requested execution.

    Returns:
        Matching durable attempt.

    Raises:
        ValueError: RESEARCH_EXECUTION_REQUEST for missing or altered admission.
    """
    rows = research_journal(trial["policy"]).read(path)
    number = trial["number"]
    if (
        number > len(rows)
        or decode_research_proposal(narrow_json_to_dict(load_json_str(rows[number - 1]["request"])))
        != trial["proposal"]
    ):
        raise ValueError("RESEARCH_EXECUTION_REQUEST: request differs from admission")
    return rows[number - 1]


def cluster_command(
    cluster: ResearchCluster, directory: Path, module: str, args: tuple[str, ...], log: str
) -> Path:
    """Run a maintained HPC3 CLI with durable output and real failure propagation.

    Args:
        cluster: Fixed transport settings.
        directory: Attempt directory.
        module: Maintained HPC3 entrypoint suffix.
        args: Validated operation arguments.
        log: Exclusive output filename.

    Returns:
        Successful command's durable output path.

    Raises:
        RuntimeError: RESEARCH_EXECUTION_PROCESS for a nonzero process exit.
        TimeoutExpired: Transport timeout; submitting intent remains unresolved.
    """
    output = directory / log
    command = (
        cluster["python"],
        "-m",
        "hpc3.cli." + module,
        "--config",
        cluster["workspace"],
        *args,
    )
    code = _test_hooks.campaign_runner(command, directory, output)
    if code != 0:
        raise RuntimeError(f"RESEARCH_EXECUTION_PROCESS: {module} exited {code}; inspect {output}")
    return output


def require_receipt(
    document: JSONObject,
    trial: ResearchTrial,
    directory: Path,
    *,
    final: bool = False,
) -> str:
    """Bind a returned ledger row to the frozen recipe and request.

    Args:
        document: HPC3 ledger encoding.
        trial: Expected admission.
        directory: Persisted recipe.
        final: Require the final allocation identity instead of validation.

    Returns:
        Numeric Slurm allocation id.

    Raises:
        ValueError: RESEARCH_EXECUTION_RECEIPT for altered scheduler lineage.
    """
    recipe = read_document(directory / "run.json")
    experiment = require_dict(document, "experiment")
    job = require_str(document, "job_id")
    request_digest = sha256(dump_json_str(encode_research_trial(trial)).encode()).hexdigest()
    if (
        not job.isdecimal()
        or experiment.get("research_attempt") != attempt_identity(trial, final=final)
        or experiment.get("request_sha256") != request_digest
        or document.get("artifact") != recipe["artifact"]
        or document.get("image_digest") != trial["policy"]["image_digest"]
    ):
        raise ValueError("RESEARCH_EXECUTION_RECEIPT: ledger differs from frozen attempt")
    return job


def submit_attempt(
    path: Path, trial: ResearchTrial, cluster: ResearchCluster, directory: Path
) -> ResearchAttempt:
    """Reserve submission once, stage frozen inputs, submit and bind its receipt.

    Args:
        path: Existing admission journal.
        trial: Frozen admitted request.
        cluster: Operator settings.
        directory: New attempt directory.

    Returns:
        Running attempt bound to one actual Slurm allocation.

    Raises:
        ValueError: Refused, duplicate or altered attempt; never submits it.
        RuntimeError: Failed transport; intent remains submitting for explicit recovery.
    """
    require_trial_attempt(path, trial)
    trial_event(path, trial, "submit", {})
    prepare_bundle(directory, cluster, trial)
    job = stage_and_submit(trial, cluster, directory)
    return trial_event(path, trial, "scheduled", {"job_id": job})


def stage_and_submit(
    trial: ResearchTrial,
    cluster: ResearchCluster,
    directory: Path,
    *,
    final: bool = False,
) -> str:
    """Stage a reserved immutable recipe, submit it and verify its native receipt.

    Args:
        trial: Frozen request with an already-reserved submission intent.
        cluster: Maintained transport settings.
        directory: Prepared recipe and checksum bundle.
        final: Bind a final allocation rather than validation.

    Returns:
        Numeric job identity from the existing HPC3 ledger.

    Raises:
        RuntimeError: Transport failed; caller retains its submission intent.
        ValueError: Native receipt differs from the reserved request.
    """
    cluster_command(
        cluster,
        directory,
        "stage",
        (
            "--manifest",
            str(directory / "stage.json"),
            "--source-dir",
            str(directory),
            "--expect-from",
            str(directory / "expected.txt"),
        ),
        "stage.log",
    )
    output = cluster_command(
        cluster,
        directory,
        "submit",
        ("--run", str(directory / "run.json"), "--format", "json"),
        "submit.log",
    )
    return require_receipt(read_document(output), trial, directory, final=final)


def observe_attempt(
    trial: ResearchTrial,
    cluster: ResearchCluster,
    directory: Path,
    *,
    final: bool = False,
) -> JSONObject:
    """Retain a fresh observation from the existing native ledger and evidence verifier.

    Args:
        trial: Exact frozen request.
        cluster: Original transport and report settings.
        directory: Existing reserved recipe directory.
        final: Observe the final allocation identity.

    Returns:
        Native accounting and, on completion, measured data/result evidence.

    Raises:
        RuntimeError: Observation process failed; earlier receipts remain intact.
        ValueError: Native CLI refuses missing, ambiguous or changed evidence.
    """
    log = f"observe-{len(tuple(directory.glob('observe-*.log')))}.log"
    output = cluster_command(
        cluster,
        directory,
        "execution_inspect",
        (
            "--attempt",
            attempt_identity(trial, final=final),
            "--checksums",
            str(directory / "checksums.txt"),
            "--report-task",
            cluster["report_task"],
        ),
        log,
    )
    return read_document(output)


def reconcile_attempt(
    path: Path, trial: ResearchTrial, cluster: ResearchCluster, directory: Path
) -> ResearchAttempt:
    """Recover a receipt or reconcile native accounting; never resubmit an attempt.

    Args:
        path: Durable journal.
        trial: Original frozen request.
        cluster: Operator settings.
        directory: Existing attempt directory with checksums and recipe.

    Returns:
        Running or terminal state backed by native execution evidence.

    Raises:
        ValueError: Altered lineage, missing/ambiguous job or invalid lifecycle.
        RuntimeError: Remote observation failure; prior state remains intact.
    """
    row = require_trial_attempt(path, trial)
    if row["status"] not in {"submitting", "running"}:
        raise ValueError("RESEARCH_EXECUTION_STATE: require submitting or running attempt")
    report = observe_attempt(trial, cluster, directory)
    job = require_receipt(require_dict(report, "ledger"), trial, directory)
    if row["status"] == "submitting":
        row = trial_event(path, trial, "scheduled", {"job_id": job})
    if row["job_id"] != job:
        raise ValueError("RESEARCH_EXECUTION_RECEIPT: observed job changed")
    accounting = require_dict(report, "accounting")
    native = require_dict(accounting, "job")
    state = require_str(native, "state")
    if state == "COMPLETED":
        require_measured_dataset(
            report, trial["manifest"], trial["policy"]["dataset_digest"], "train-validation"
        )
        return trial_event(
            path,
            trial,
            "complete",
            {
                "job_id": job,
                "dataset_digest": trial["policy"]["dataset_digest"],
                "code_digest": trial["policy"]["code_digest"],
                "record": require_dict(report, "record"),
            },
        )
    if state in {"FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY"}:
        action = "timeout" if state == "TIMEOUT" else "failed"
        return trial_event(path, trial, action, {"reason": dump_json_str(accounting)})
    if state not in {"PENDING", "RUNNING", "SUSPENDED", "COMPLETING", "REQUEUED"}:
        raise ValueError("RESEARCH_EXECUTION_STATE: unknown native scheduler state")
    return row


def require_measured_dataset(
    report: JSONObject,
    manifest: str,
    digest: str,
    partition: Literal["train-validation", "final"],
) -> None:
    """Verify native accounting and exact encrypted input identities for either split.

    Args:
        report: Existing native execution inspection response.
        manifest: Expected ciphertext manifest path.
        digest: Expected ciphertext manifest hash.
        partition: Authorized validation or final partition.

    Raises:
        ValueError: RESEARCH_EXECUTION_RECEIPT for mismatched measured evidence.
    """
    measurement = require_dict(report, "measurement")
    if require_dict(measurement, "accounting") != require_dict(report, "accounting"):
        raise ValueError("RESEARCH_EXECUTION_RECEIPT: measured accounting differs")
    dataset = require_dict(measurement, "dataset")
    bundle = decode_encrypted_bundle(require_dict(dataset, "bundle"))
    verified = require_dict(measurement, "verified_files")
    if (
        dataset.get("bundle_sha256") != digest
        or verified.get(manifest) != digest
        or bundle["partition"] != partition
    ):
        raise ValueError("RESEARCH_EXECUTION_RECEIPT: measured dataset differs from admission")


__all__ = [
    "cluster_command",
    "observe_attempt",
    "reconcile_attempt",
    "require_measured_dataset",
    "require_receipt",
    "require_trial_attempt",
    "stage_and_submit",
    "submit_attempt",
    "trial_event",
]
