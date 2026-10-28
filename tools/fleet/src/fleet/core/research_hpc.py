"""Poll the maintained admitted-proposal controller without replacing its journal."""

from __future__ import annotations

import hashlib
import pathlib

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_int,
    require_list,
    require_str,
)
from platform_core.run_record import decode_run_record, encode_run_record

from fleet.core import _test_hooks, research_control


def _attempt(report: JSONObject, number: int) -> JSONObject:
    """Select exactly the admitted attempt named by the frozen trial.

    Args:
        report: Maintained controller's structured output.
        number: Frozen trial number.

    Returns:
        The single matching journal row.

    Raises:
        ValueError: RESEARCH_ATTEMPT_MISSING for absent or ambiguous admission.
    """
    rows = [narrow_json_to_dict(value) for value in require_list(report, "attempts")]
    selected = [row for row in rows if require_int(row, "number") == number]
    if len(selected) != 1:
        raise ValueError("RESEARCH_ATTEMPT_MISSING: expected exactly one admitted trial")
    return selected[0]


def poll(
    document: JSONObject,
    log: pathlib.Path,
    task_id: str,
    environment: tuple[tuple[str, str], ...] = (),
) -> JSONObject:
    """Submit an admitted trial once, reconcile it, or verify its completed receipt.

    Args:
        document: Tenant-authorized content-addressed private specification.
        log: Private controller transcript.
        task_id: Queue-owned board task for fresh native evidence.
        environment: Private native report credentials.

    Returns:
        Metadata only: state, native job id and verified record digest.

    Raises:
        ValueError: Frozen input drift, invalid lifecycle or mismatched receipt.
        OSError: Missing input or process failure; journal state remains authoritative.
    """
    expected = {
        "tenant",
        "backend",
        "attempt",
        "python",
        "script",
        "policy",
        "trial",
        "cluster",
        "ledger",
        "directory",
        "native_attempt",
    }
    if set(document) != expected:
        raise ValueError("RESEARCH_HPC_FIELDS: unexpected private controller fields")
    python = str(research_control.require_path(document, "python"))
    script = research_control.verified_input(document, "script")
    policy = research_control.verified_input(document, "policy")
    trial = research_control.verified_input(document, "trial")
    cluster_path = research_control.verified_input(document, "cluster")
    ledger = research_control.require_path(document, "ledger")
    directory = research_control.require_path(document, "directory")
    trial_document = narrow_json_to_dict(load_json_str(_test_hooks.read_text(trial)))
    number = require_int(trial_document, "number")
    if number < 1:
        raise ValueError("RESEARCH_ATTEMPT_MISSING: trial number must be positive")
    prefix = (python, str(script), "--policy", str(policy), "--ledger", str(ledger))
    row = _attempt(research_control.invoke((*prefix, "status"), log, 60, environment), number)
    state = require_str(row, "status")
    if state in {"admitted", "submitting", "running"}:
        action = "submit" if state == "admitted" else "reconcile"
        row = _attempt(
            research_control.invoke(
                (
                    *prefix,
                    action,
                    "--trial",
                    str(trial),
                    "--cluster",
                    str(cluster_path),
                    "--directory",
                    str(directory),
                ),
                log,
                300,
                environment,
            ),
            number,
        )
        state = require_str(row, "status")
    native = require_str(row, "job_id")
    if state == "complete":
        cluster = narrow_json_to_dict(load_json_str(_test_hooks.read_text(cluster_path)))
        measured = research_control.invoke(
            (
                python,
                "-m",
                "hpc3.cli.execution_inspect",
                "--config",
                require_str(cluster, "workspace"),
                "--attempt",
                require_str(document, "native_attempt"),
                "--checksums",
                str(directory / "checksums.txt"),
                "--report-task",
                task_id,
            ),
            log,
            300,
            environment,
        )
        recorded = decode_run_record(narrow_json_to_dict(load_json_str(require_str(row, "record"))))
        observed = decode_run_record(require_dict(measured, "record"))
        if (
            require_str(require_dict(measured, "ledger"), "job_id") != native
            or observed != recorded
        ):
            raise ValueError("RESEARCH_RECEIPT_DRIFT: native observation differs from the journal")
        digest = hashlib.sha256(dump_json_str(encode_run_record(observed)).encode()).hexdigest()
        return {"state": "complete", "native_job_id": native, "record_sha256": digest}
    if state in {"failed", "timeout", "interrupted", "refused"}:
        return {"state": "failed", "native_job_id": native, "record_sha256": None}
    if state != "running":
        raise ValueError(f"RESEARCH_CONTROLLER_STATE: unexpected admitted-controller state {state}")
    return {"state": "running", "native_job_id": native, "record_sha256": None}


__all__ = ["poll"]
