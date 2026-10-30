"""Final native allocation reuses the released grant and existing HPC3 ledger."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str, require_dict, require_str
from platform_core.run_record import decode_run_record, encode_run_record
from platform_core.stage_manifest import decode_stage_manifest

from ..evaluation.evidence import require_private_directory
from .research_cluster import (
    FinalExecution,
    ResearchCluster,
    attempt_identity,
    decode_final_execution,
    prepare_bundle,
)
from .research_execution import (
    observe_attempt,
    require_measured_dataset,
    require_receipt,
    require_trial_attempt,
    stage_and_submit,
)
from .research_trial_records import ResearchTrial, encode_research_trial
from .validation import read_document


def _delivered(
    ledger: Path,
    trial: ResearchTrial,
    directory: Path,
    remote_manifest: str,
) -> FinalExecution:
    """Bind a successful private delivery to the original journal's released request.

    Args:
        ledger: Authoritative admission journal, retained on the controller.
        trial: Selected request.
        directory: Consumed local grant and its exclusive delivery receipt.
        remote_manifest: Final ciphertext manifest pinned in native checksums.

    Returns:
        Strict native final input paths; no private key is accessed.

    Raises:
        ValueError: RESEARCH_FINAL_NATIVE for missing or altered delivery authority.
        OSError: Missing grant metadata or journal.
    """
    row = require_trial_attempt(ledger, trial)
    require_private_directory(directory)
    manifest = decode_stage_manifest(read_document(directory / "delivery.json"))
    expected = {
        "research_attempt": attempt_identity(trial),
        "request_sha256": sha256(dump_json_str(encode_research_trial(trial)).encode()).hexdigest(),
        "final_manifest_sha256": row["final_manifest_digest"],
    }
    if (
        row["selection"] != "released"
        or manifest["provenance"] != expected
        or (directory / "consumed").read_bytes() != b"released\n"
        or (directory / "final.agekey").exists()
        or (directory / "delivery.log").read_text() != "private delivery verified\n"
        or [file["name"] for file in manifest["files"]] != ["release.json", "final.agekey"]
    ):
        raise ValueError("RESEARCH_FINAL_NATIVE: require verified delivery of the released grant")
    return decode_final_execution(
        {
            "manifest": remote_manifest,
            "manifest_digest": row["final_manifest_digest"],
            "grant_directory": manifest["destination"],
        }
    )


def submit_final_attempt(
    ledger: Path,
    trial: ResearchTrial,
    cluster: ResearchCluster,
    grant: Path,
    remote_manifest: str,
) -> JSONObject:
    """Reserve the delivered grant's native directory once, then stage and submit.

    Args:
        ledger: Original journal with its committed one-shot release.
        trial: Selected validation request, unchanged for final scoring.
        cluster: Frozen evaluator-owned runtime and ciphertext checksums.
        grant: Consumed local grant; its native subdirectory is the submission intent.
        remote_manifest: Frozen remote final ciphertext manifest path.

    Returns:
        Shared queue metadata with the real native allocation identity.

    Raises:
        FileExistsError: Submission already attempted; reconcile instead of retrying.
        ValueError: Released authority, frozen input or native receipt differs.
        RuntimeError: Transport failure; retained intent must never be resubmitted.
    """
    final = _delivered(ledger, trial, grant, remote_manifest)
    directory = grant / "native"
    prepare_bundle(directory, cluster, trial, final=final)
    job = stage_and_submit(trial, cluster, directory, final=True)
    return {"state": "running", "native_job_id": job, "record_sha256": None}


def reconcile_final_attempt(
    ledger: Path,
    trial: ResearchTrial,
    cluster: ResearchCluster,
    grant: Path,
    remote_manifest: str,
) -> JSONObject:
    """Observe the sole final allocation and verify its ciphertext and numerical record.

    Args:
        ledger: Original journal; validation evidence is retained unchanged.
        trial: Selected request.
        cluster: Same frozen native configuration used for submission.
        grant: Consumed local grant containing its native submission intent.
        remote_manifest: Exact final ciphertext manifest path.

    Returns:
        Existing shared queue state, native job and verified record digest.

    Raises:
        ValueError: Missing/ambiguous allocation or changed native evidence.
        RuntimeError: Failed observation; never submits or restores a grant.
    """
    final = _delivered(ledger, trial, grant, remote_manifest)
    directory = grant / "native"
    report = observe_attempt(trial, cluster, directory, final=True)
    job = require_receipt(require_dict(report, "ledger"), trial, directory, final=True)
    accounting = require_dict(report, "accounting")
    state = require_str(require_dict(accounting, "job"), "state")
    if state == "COMPLETED":
        require_measured_dataset(report, remote_manifest, final["manifest_digest"], "final")
        record = decode_run_record(require_dict(report, "record"))
        if (
            record["label"] != f"final-{trial['number']}"
            or record["experiment"] != trial["policy"]["protocol_digest"]
            or record["fingerprint"]["image_digest"] != trial["policy"]["image_digest"]
        ):
            raise ValueError(
                "RESEARCH_FINAL_NATIVE: final allocation evidence differs from release"
            )
        digest = sha256(dump_json_str(encode_run_record(record)).encode()).hexdigest()
        return {"state": "complete", "native_job_id": job, "record_sha256": digest}
    if state in {"FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY"}:
        return {"state": "failed", "native_job_id": job, "record_sha256": None}
    if state not in {"PENDING", "RUNNING", "SUSPENDED", "COMPLETING", "REQUEUED"}:
        raise ValueError("RESEARCH_FINAL_NATIVE: unknown scheduler state")
    return {"state": "running", "native_job_id": job, "record_sha256": None}


__all__ = ["reconcile_final_attempt", "submit_final_attempt"]
