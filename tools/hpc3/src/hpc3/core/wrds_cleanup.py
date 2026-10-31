"""Measure terminal job scratch on its original node through Grid Engine."""

from __future__ import annotations

import pathlib

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)

from hpc3.contracts.wrds_run import WrdsRun, digest, run_identity, safe_label
from hpc3.core import (
    _test_hooks,
    artifacts,
    wrds_attempt_report,
    wrds_grid,
    wrds_jobs,
    wrds_results,
    wrds_scheduler,
)
from hpc3.core.digest import check_remote_digest, parse_remote_digest, sha256_hex
from hpc3.core.remote import remote_digest
from hpc3.core.wrds_bundle import worker_bundle


def _subject(run: WrdsRun, attempt: str) -> tuple[str, str, JSONObject, str, str | None]:
    """Read a failed probe receipt or a completed job's authenticated attestation.

    Args:
        run: Frozen native experiment specification.
        attempt: Terminal attempt; failed probes require an opened.json receipt.

    Returns:
        Subject id, node, authenticated receipt, digest and optional probe path.

    Raises:
        ValueError: Nonterminal subject or inconsistent receipts.
        AppError: Transport, scheduler or verified download fails.
    """
    identifier = wrds_jobs.job_id(run, attempt)
    state = wrds_scheduler.status(run["host"], identifier)
    if state not in {"failed", "completed"}:
        raise ValueError("WRDS_SCRATCH_STATE: subject must have terminal accounting")
    accounting = wrds_scheduler.accounting(run["host"], identifier)
    host = accounting["hostname"]
    if state == "completed":
        completion = wrds_results.attested_completion(run, attempt, identifier)
        return identifier, host, completion, sha256_hex(dump_json_str(completion).encode()), None
    directory = wrds_jobs.attempt_directory(run, attempt)
    opened = artifacts.download(run["host"], directory + "/outputs/opened.json")
    receipt = narrow_json_to_dict(load_json_str(opened["text"]))
    if set(receipt) != {
        "node",
        "scratch",
        "scratch_mode",
        "private",
        "mode",
        "manifest_sha256",
        "files",
    }:
        raise ValueError("WRDS_SCRATCH_RECEIPT: unexpected opened receipt fields")
    private = require_str(receipt, "private")
    if (
        require_str(receipt, "node") != host
        or require_str(receipt, "scratch") != str(pathlib.PurePosixPath(private).parent)
        or require_int(receipt, "mode") != 0o700
        or require_int(receipt, "files") < 1
    ):
        raise ValueError("WRDS_SCRATCH_RECEIPT: opened receipt differs from private native storage")
    digest(require_str(receipt, "manifest_sha256"))
    return identifier, host, receipt, opened["sha256"], private


def submit(run: WrdsRun, attempt: str, output: pathlib.Path) -> str:
    """Submit one same-node observer and persist its receipt before accounting.

    Args:
        run: Frozen native experiment specification.
        attempt: Terminal attempt; failed probes require an opened.json receipt.
        output: Future evidence file; its stem also names the remote observation.

    Returns:
        Native observer job identifier, durably recorded beside output.

    Raises:
        ValueError: Invalid subject or unsafe output name.
        OSError: Receipt creation fails or a prior artifact exists.
        AppError: Scheduler, upload or verified download fails.
    """
    receipt_path = wrds_grid.receipt_path(output)
    wrds_grid.refuse_existing([output, receipt_path])
    label = safe_label(output.stem)
    identifier, host, _, receipt_sha, private = _subject(run, attempt)
    directory = wrds_jobs.attempt_directory(run, attempt)
    remote = wrds_grid.remote_directory(directory + "/scratch-" + label)
    wrds_grid.create_directory(run["host"], remote)
    archive = str(remote / "worker.pyz")
    source_sha = wrds_grid.upload(run["host"], archive, worker_bundle())
    target = str(remote / "observation.json")
    observer = wrds_grid.submit(
        run["host"],
        remote,
        "wrds_scratch",
        [
            run["runtime"]["python"],
            "-s",
            archive,
            "scratch",
            host,
            identifier,
            *([] if private is None else [private]),
            target,
        ],
        environment=False,
        execution_host=host,
    )
    _test_hooks.write_text(
        receipt_path,
        dump_json_str(
            {
                "identity": run_identity(run),
                "attempt": attempt,
                "subject_job_id": identifier,
                "observer_job_id": observer,
                "worker_sha256": source_sha,
                "subject_receipt_sha256": receipt_sha,
            }
        )
        + "\n",
    )
    return observer


def _submitted_observer(
    run: WrdsRun, attempt: str, output: pathlib.Path, identifier: str, receipt_sha: str
) -> tuple[str, str]:
    """Validate the persisted observer against the live subject's exact identity.

    Args:
        run: Frozen native experiment specification.
        attempt: Terminal attempt; failed probes require an opened.json receipt.
        output: Evidence destination naming the persisted submission receipt.
        identifier: Verified native subject job id.
        receipt_sha: Verified subject receipt digest.

    Returns:
        Observer job id and worker digest from the matched receipt.

    Raises:
        ValueError: Receipt fields or identity do not match the subject.
        OSError: Submission receipt cannot be read.
    """
    recorded = narrow_json_to_dict(
        load_json_str(_test_hooks.read_bytes(wrds_grid.receipt_path(output)).decode())
    )
    if set(recorded) != {
        "identity",
        "attempt",
        "subject_job_id",
        "observer_job_id",
        "worker_sha256",
        "subject_receipt_sha256",
    }:
        raise ValueError("WRDS_SCRATCH_SUBMISSION: unexpected submission receipt fields")
    if any(
        recorded.get(key) != expected
        for key, expected in (
            ("identity", run_identity(run)),
            ("attempt", attempt),
            ("subject_job_id", identifier),
            ("subject_receipt_sha256", receipt_sha),
        )
    ):
        raise ValueError("WRDS_SCRATCH_SUBMISSION: receipt does not identify this subject")
    observer = wrds_scheduler.parse_job_id(require_str(recorded, "observer_job_id"))
    source_sha = digest(require_str(recorded, "worker_sha256"))
    return observer, source_sha


def _check_observation(
    reading: JSONObject,
    host: str,
    identifier: str,
    observer: str,
    receipt: JSONObject,
    private: str | None,
) -> None:
    """Require the observer's exact identity, scope and typed directory measurements.

    Raises:
        ValueError: WRDS_SCRATCH_OBSERVER for mismatched identities or scope.
        TypeError: A required metadata field has the wrong type.
    """
    expected_fields = {
        "node",
        "job_id",
        "observer_job_id",
        "observed_at",
        "matching_scratch_roots",
    }
    if private is not None:
        expected_fields.update({"scratch", "private", "scratch_present", "private_present"})
    if set(reading) != expected_fields:
        raise ValueError("WRDS_SCRATCH_OBSERVER: unexpected observation fields")
    if any(
        reading.get(key) != expected
        for key, expected in (
            ("node", host),
            ("job_id", identifier),
            ("observer_job_id", observer),
        )
    ):
        raise ValueError("WRDS_SCRATCH_OBSERVER: observation identity differs from the subject")
    roots = require_list(reading, "matching_scratch_roots")
    if any(not isinstance(root, str) or not root.startswith(identifier + ".1.") for root in roots):
        raise ValueError("WRDS_SCRATCH_OBSERVER: scratch roots must belong to the subject")
    if private is not None:
        if reading["private"] != private or reading["scratch"] != receipt["scratch"]:
            raise ValueError("WRDS_SCRATCH_OBSERVER: observation paths differ from the subject")
        for key in ("scratch_present", "private_present"):
            if type(reading[key]) is not bool:
                raise ValueError("WRDS_SCRATCH_OBSERVER: presence observations must be booleans")
    require_str(reading, "observed_at")


def measure(run: WrdsRun, attempt: str, output: pathlib.Path) -> str | None:
    """Reconcile a recorded observer; pending accounting never publishes evidence.

    Args:
        run: Frozen native experiment specification.
        attempt: Terminal attempt; failed probes require an opened.json receipt.
        output: Evidence destination with its existing submission receipt.

    Returns:
        Complete program-authored report, or None while accounting is pending.

    Raises:
        ValueError: Receipt mismatch or unsuccessful/malformed observer evidence.
        OSError: Missing receipt or local evidence creation fails.
        AppError: Transport, scheduler or verified download fails.
    """
    wrds_grid.refuse_existing([output])
    label = safe_label(output.stem)
    identifier, host, receipt, receipt_sha, private = _subject(run, attempt)
    observer, source_sha = _submitted_observer(run, attempt, output, identifier, receipt_sha)
    state = wrds_scheduler.status(run["host"], observer)
    if state == "accounting-pending":
        return None
    if state != "completed":
        raise ValueError("WRDS_SCRATCH_OBSERVER: observer must have completed successfully")
    directory = wrds_jobs.attempt_directory(run, attempt)
    remote = wrds_grid.remote_directory(directory + "/scratch-" + label)
    archive = str(remote / "worker.pyz")
    check_remote_digest(
        archive, source_sha, parse_remote_digest(remote_digest(run["host"], archive), archive)
    )
    observed_accounting = wrds_scheduler.accounting(run["host"], observer)
    if observed_accounting != {"hostname": host, "failed": "0", "exit_status": "0"}:
        raise ValueError("WRDS_SCRATCH_OBSERVER: observer accounting differs from requested node")
    observed = artifacts.download(run["host"], str(remote / "observation.json"))
    reading = narrow_json_to_dict(load_json_str(observed["text"]))
    _check_observation(reading, host, identifier, observer, receipt, private)
    report, _ = wrds_attempt_report.measure(run, attempt)
    evidence: JSONObject = {
        "subject_report": report,
        "subject_receipt": receipt,
        "subject_receipt_sha256": receipt_sha,
        "observer_source_sha256": source_sha,
        "observer_accounting": {
            "hostname": observed_accounting["hostname"],
            "failed": observed_accounting["failed"],
            "exit_status": observed_accounting["exit_status"],
        },
        "observation": reading,
        "observation_sha256": observed["sha256"],
    }
    encoded = dump_json_str(evidence)
    _test_hooks.write_text(output, encoded + "\n")
    return "WRDS SCRATCH LOAD-REPORT\n" + encoded


__all__ = ["measure", "submit"]
