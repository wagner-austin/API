"""WRDS polling through the maintained held-submission and verified-collection CLI."""

from __future__ import annotations

import hashlib
import pathlib

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_str,
    require_bool,
    require_dict,
    require_str,
)
from platform_core.run_record import decode_run_record

from fleet.core import _test_hooks, file_turn, research_control


def _reserve(directory: pathlib.Path) -> bool:
    """Record submission intent once, before any native side effect.

    Args:
        directory: Private directory of the immutable queue specification.

    Returns:
        True only for the writer that created intent; all later calls reconcile.
    """
    with file_turn.held(
        directory / "submission.lock",
        holder="research-wrds",
        bounds=file_turn.TurnBounds(
            what="WRDS submission intent",
            waited_for="another intent writer",
            stale_seconds=30,
            poll_seconds=1,
        ),
    ):
        intent = directory / "submission.intent"
        if _test_hooks.file_exists(intent):
            return False
        _test_hooks.write_text(intent, "Native submission may have begun; reconcile its receipt.\n")
        return True


def _verify_collected(
    output: pathlib.Path, job: str, attempt: str, identity: str, completion_digest: str
) -> str:
    """Recheck collected artifacts and lineage after an interrupted queue close.

    Args:
        output: Maintained collector's output directory.
        job: Native job just observed.
        attempt: Frozen attempt identifier.
        identity: Identity just verified by the native report command.
        completion_digest: Canonical completion digest just measured on WRDS.

    Returns:
        Digest of the existing RunRecord bytes.

    Raises:
        ValueError: RESEARCH_WRDS_RECEIPT or RESEARCH_WRDS_ARTIFACT on drift.
        OSError: Partial collection; never replace it with an unverified success.
    """
    completion = narrow_json_to_dict(
        load_json_str(_test_hooks.read_text(output / "completion.json"))
    )
    if hashlib.sha256(dump_json_str(completion).encode()).hexdigest() != completion_digest:
        raise ValueError("RESEARCH_WRDS_RECEIPT: collected completion differs from native bytes")
    if (
        require_str(completion, "job_id") != job
        or require_str(completion, "attempt") != attempt
        or require_str(completion, "identity") != identity
    ):
        raise ValueError("RESEARCH_WRDS_RECEIPT: collected completion differs from native identity")
    for name, value in require_dict(completion, "outputs").items():
        path = (output / "outputs" / name).resolve()
        if not path.is_relative_to((output / "outputs").resolve()):
            raise ValueError("RESEARCH_WRDS_ARTIFACT: output escapes its collection directory")
        if hashlib.sha256(_test_hooks.read_bytes(path)).hexdigest() != narrow_json_to_str(value):
            raise ValueError(f"RESEARCH_WRDS_ARTIFACT: collected bytes changed for {name}")
    record = _test_hooks.read_bytes(output / "results.runrecord.json")
    if decode_run_record(narrow_json_to_dict(load_json_str(record.decode()))) != decode_run_record(
        require_dict(completion, "record")
    ):
        raise ValueError("RESEARCH_WRDS_RECEIPT: collected RunRecord differs from completion")
    return hashlib.sha256(record).hexdigest()


def poll(
    document: JSONObject,
    log: pathlib.Path,
    task_id: str,
    environment: tuple[tuple[str, str], ...] = (),
) -> JSONObject:
    """Submit once or recover the native receipt, then collect verified success.

    Args:
        document: Tenant-authorized private specification with pinned native spec.
        log: Private transcript under this request's stable directory.
        task_id: Queue-owned board task for program-generated native evidence.
        environment: Private native report credentials.

    Returns:
        Metadata only: state, native job id and verified record digest.

    Raises:
        ValueError: Invalid specification, changed artifact or controller failure.
        OSError: Missing/partial durable receipt; intent never authorizes a second submit.
    """
    if set(document) != {"tenant", "backend", "attempt", "python", "spec", "output", "adopt"}:
        raise ValueError("RESEARCH_WRDS_FIELDS: unexpected private controller fields")
    python = str(research_control.require_path(document, "python"))
    spec = research_control.verified_input(document, "spec")
    output = research_control.require_path(document, "output")
    attempt = require_str(document, "attempt")
    prefix = (python, "-m", "hpc3.cli.wrds")
    arguments = ("--spec", str(spec), "--attempt", attempt)
    if not require_bool(document, "adopt") and _reserve(log.parent):
        research_control.invoke((*prefix, "submit", *arguments), log, 300, environment)
    status = research_control.invoke((*prefix, "status", *arguments), log, 60, environment)
    job, state = require_str(status, "job_id"), require_str(status, "state")
    if state == "failed":
        return {"state": "failed", "native_job_id": job, "record_sha256": None}
    if state == "completed":
        if not _test_hooks.directory_exists(output):
            research_control.invoke(
                (*prefix, "collect", *arguments, "--output", str(output)), log, 300, environment
            )
        report = research_control.invoke(
            (*prefix, "report", *arguments, "--report-task", task_id), log, 300, environment
        )
        digest = _verify_collected(
            output,
            job,
            attempt,
            require_str(report, "identity"),
            require_str(report, "completion_sha256"),
        )
        return {"state": "complete", "native_job_id": job, "record_sha256": digest}
    if "h" in state and state != "accounting-pending":
        research_control.invoke((*prefix, "resume", *arguments), log, 60, environment)
    return {"state": "running", "native_job_id": job, "record_sha256": None}


__all__ = ["poll"]
