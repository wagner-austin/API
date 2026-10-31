"""Measure a failed storage probe's exact scratch path through Grid Engine."""

from __future__ import annotations

import pathlib

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_str,
)

from hpc3.contracts.wrds_run import WrdsRun, digest, safe_label
from hpc3.core import _test_hooks, artifacts, wrds_attempt_report, wrds_grid, wrds_jobs
from hpc3.core import wrds_scheduler
from hpc3.core.wrds_bundle import worker_bundle


def measure(run: WrdsRun, attempt: str, output: pathlib.Path) -> str:
    """Schedule a metadata observation on the failed job's accounting hostname.

    Args:
        run: Frozen storage-probe specification.
        attempt: Failed attempt with an opened.json receipt.
        output: New local evidence file; its stem also names the remote observation.

    Returns:
        Program-authored report binding source, native jobs and downloaded evidence.

    Raises:
        ValueError: Active/successful subject, inconsistent receipts or failed observer.
        OSError: Local evidence creation fails.
        AppError: Transport, scheduler, upload or verified download fails.
    """
    wrds_grid.refuse_existing([output])
    label = safe_label(output.stem)
    identifier = wrds_jobs.job_id(run, attempt)
    if wrds_scheduler.status(run["host"], identifier) != "failed":
        raise ValueError("WRDS_SCRATCH_STATE: subject must have failed terminal accounting")
    accounting = wrds_scheduler.accounting(run["host"], identifier)
    host = accounting["hostname"]
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
    report, _ = wrds_attempt_report.measure(run, attempt)
    remote = wrds_grid.remote_directory(directory + "/scratch-" + label)
    wrds_grid.create_directory(run["host"], remote)
    archive = str(remote / "worker.pyz")
    source_sha = wrds_grid.upload(run["host"], archive, worker_bundle())
    target = str(remote / "observation.json")
    observer = wrds_grid.submit(
        run["host"],
        remote,
        "wrds_scratch",
        [run["runtime"]["python"], "-s", archive, "scratch", host, identifier, private, target],
        environment=False,
        execution_host=host,
    )
    observed_accounting = wrds_scheduler.accounting(run["host"], observer)
    if observed_accounting != {"hostname": host, "failed": "0", "exit_status": "0"}:
        raise ValueError("WRDS_SCRATCH_OBSERVER: observer accounting differs from requested node")
    observed = artifacts.download(run["host"], target)
    reading = narrow_json_to_dict(load_json_str(observed["text"]))
    if set(reading) != {
        "node",
        "job_id",
        "observer_job_id",
        "observed_at",
        "scratch",
        "private",
        "scratch_present",
        "private_present",
    }:
        raise ValueError("WRDS_SCRATCH_OBSERVER: unexpected observation fields")
    if any(
        reading.get(key) != expected
        for key, expected in (
            ("node", host),
            ("job_id", identifier),
            ("observer_job_id", observer),
            ("private", private),
            ("scratch", require_str(receipt, "scratch")),
        )
    ):
        raise ValueError("WRDS_SCRATCH_OBSERVER: observation identity differs from the subject")
    for key in ("scratch_present", "private_present"):
        if type(reading.get(key)) is not bool:
            raise ValueError("WRDS_SCRATCH_OBSERVER: presence observations must be booleans")
    require_str(reading, "observed_at")
    evidence: JSONObject = {
        "subject_report": report,
        "opened": receipt,
        "opened_sha256": opened["sha256"],
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


__all__ = ["measure"]
