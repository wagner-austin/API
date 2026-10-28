"""Verify result provenance and copy only explicitly declared outputs."""

from __future__ import annotations

import hashlib
import pathlib

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_str,
    require_dict,
    require_str,
)
from platform_core.run_record import decode_run_record, encode_run_record

from hpc3.contracts.wrds_run import WrdsRun, decode_wrds_runtime, digest, run_identity
from hpc3.core import _test_hooks, remote, wrds_jobs, wrds_scheduler
from hpc3.wrds import _test_hooks as worker_hooks


def not_complete(identifier: str, state: str) -> str:
    """Word the refusal collect gives a job that did not succeed.

    Args:
        identifier: Scheduler job id.
        state: Its state from :func:`wrds_scheduler.status`.

    Returns:
        The ``WRDS_NOT_COMPLETE`` message.
    """
    return f"WRDS_NOT_COMPLETE: job {identifier} is {state}"


def run_record_bytes(result: JSONObject) -> bytes:
    """Serialize an attested run record exactly as collect writes it.

    Args:
        result: Verified completion metadata.

    Returns:
        The bytes of ``results.runrecord.json``.

    Raises:
        ValueError: When the completion carries no valid run record.
    """
    record = decode_run_record(require_dict(result, "record"))
    return (dump_json_str(encode_run_record(record)) + "\n").encode()


def verified_completion(run: WrdsRun, attempt: str) -> JSONObject:
    """Require terminal scheduler success and a matching worker attestation.

    Args:
        run: Experiment definition.
        attempt: Attempt label.

    Returns:
        Validated completion metadata.

    Raises:
        ValueError: For nonterminal/failed jobs, and from
            :func:`attested_completion`.
    """
    identifier = wrds_jobs.job_id(run, attempt)
    state = wrds_scheduler.status(run["host"], identifier)
    if state != "completed":
        raise ValueError(not_complete(identifier, state))
    return attested_completion(run, attempt, identifier)


def attested_completion(run: WrdsRun, attempt: str, identifier: str) -> JSONObject:
    """Require a worker attestation matching the experiment and its submission.

    Only meaningful once the scheduler reports the job completed; callers
    establish that first, as :func:`verified_completion` does.

    Args:
        run: Experiment definition.
        attempt: Attempt label.
        identifier: The attempt's scheduler job id.

    Returns:
        Validated completion metadata.

    Raises:
        ValueError: For mismatched identity, runtime, outputs, worker bytes or
            aggregate payload digest.
    """
    directory = wrds_jobs.attempt_directory(run, attempt)
    result = narrow_json_to_dict(
        load_json_str(wrds_jobs.read_verified(run["host"], directory + "/completion.json"))
    )
    submission = narrow_json_to_dict(
        load_json_str(wrds_jobs.read_verified(run["host"], directory + "/submission.json"))
    )
    if (
        require_str(result, "identity") != run_identity(run)
        or require_str(result, "attempt") != attempt
        or require_str(result, "job_id") != identifier
    ):
        raise ValueError(
            "WRDS_RESULT_IDENTITY: completion belongs to another experiment or attempt"
        )
    if (
        require_str(submission, "identity") != run_identity(run)
        or require_str(submission, "attempt") != attempt
        or require_str(submission, "worker_sha256") != require_str(result, "worker_sha256")
    ):
        raise ValueError("WRDS_WORKER_DRIFT: completion differs from the submitted worker")
    if decode_wrds_runtime(require_dict(result, "runtime")) != run["runtime"]:
        raise ValueError("WRDS_RUNTIME_DRIFT: completion runtime differs from the experiment")
    outputs = require_dict(result, "outputs")
    if set(outputs) != set(run["outputs"]):
        raise ValueError("WRDS_OUTPUTS: completion does not attest every declared output")
    digests = {name: digest(narrow_json_to_str(outputs[name])) for name in run["outputs"]}
    record = decode_run_record(require_dict(result, "record"))
    if record["experiment"] != run["project"] or record["label"] != run["name"]:
        raise ValueError("WRDS_RECORD_IDENTITY: run record belongs to another experiment")
    if hashlib.sha256(dump_json_str(digests).encode()).hexdigest() != record["payload_digest"]:
        raise ValueError("WRDS_RESULT_DIGEST: run record does not match the output manifest")
    return result


def collect(
    run: WrdsRun, attempt: str, output: pathlib.Path, ledger: pathlib.Path, timestamp: str
) -> JSONObject:
    """Collect exact declared files; write a local completion marker only at the end.

    Args:
        run: Experiment definition.
        attempt: Completed attempt.
        output: New local destination, with existing parent.
        ledger: Shared lifecycle ledger.
        timestamp: UTC observation time.

    Returns:
        Verified completion metadata, not dataset contents.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: When a transported artifact differs from its attested hash.
    """
    result = verified_completion(run, attempt)
    worker_hooks.new_directory(output)
    directory = wrds_jobs.attempt_directory(run, attempt)
    outputs = require_dict(result, "outputs")
    for name in run["outputs"]:
        path = directory + "/outputs/" + name
        target = output / "outputs" / name
        _test_hooks.make_dir(target.parent)
        transferred = _test_hooks.run(
            ["scp", *remote.SSH_OPTIONS, "--", f"{run['host']}:{path}", str(target)]
        )
        if transferred["returncode"] != 0:
            raise ValueError(f"WRDS_RESULT_TRANSFER: {name}: {transferred['stderr']}")
        if worker_hooks.digest_file(target) != require_str(outputs, name):
            raise ValueError(f"WRDS_RESULT_DIGEST: transported output changed: {name}")
    worker_hooks.write(output / "results.runrecord.json", run_record_bytes(result))
    worker_hooks.write(output / "completion.json", (dump_json_str(result) + "\n").encode())
    wrds_jobs.record_event(
        ledger, run, attempt, "collected", require_str(result, "job_id"), timestamp
    )
    return result


__all__ = [
    "attested_completion",
    "collect",
    "not_complete",
    "run_record_bytes",
    "verified_completion",
]
