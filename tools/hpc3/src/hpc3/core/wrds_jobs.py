"""Durable WRDS attempts through the existing SSH and shared-ledger machinery.

Jobs are submitted held. Their scheduler id is persisted remotely and locally
before release, so a lost SSH response is recovered with resume, never a
second submission. Atomic remote attempt-directory creation also coordinates
submitters on different machines.
"""

from __future__ import annotations

import hashlib
import pathlib
import shlex

from platform_core.json_utils import (
    JSONValue,
    dump_json_str,
    load_json_str,
)

from hpc3.contracts.wrds_run import (
    WrdsRun,
    WrdsRuntime,
    absolute_path,
    decode_wrds_runtime,
    execution_document,
    run_identity,
    safe_label,
)
from hpc3.core import _test_hooks, remote, wrds_scheduler
from hpc3.core.digest import check_remote_digest, parse_remote_digest
from hpc3.core.wrds_bundle import worker_bundle


def attempt_directory(run: WrdsRun, attempt: str) -> str:
    """Resolve an attempt without a mutable 'latest' pointer.

    Args:
        run: Experiment definition.
        attempt: Explicit attempt label.

    Returns:
        Stable remote directory.
    """
    return str(
        pathlib.PurePosixPath(run["root"])
        / run["project"]
        / run["name"]
        / run_identity(run)
        / safe_label(attempt)
    )


def upload_verified(host: str, path: str, payload: bytes) -> str:
    """Transfer exact bytes and compare the remote digest.

    Args:
        host: SSH alias.
        path: Destination.
        payload: Bytes to send.

    Returns:
        Verified content digest.
    """
    expected = hashlib.sha256(payload).hexdigest()
    remote.put_bytes(host, path, payload)
    check_remote_digest(path, expected, parse_remote_digest(remote.remote_digest(host, path), path))
    return expected


def read_verified(host: str, path: str) -> str:
    """Read text metadata and compare its transported bytes with WRDS.

    Args:
        host: SSH alias.
        path: Metadata path.

    Returns:
        Verified text.
    """
    text = remote.run_remote(host, shlex.join(["cat", "--", path]))
    digest = hashlib.sha256(text.encode()).hexdigest()
    check_remote_digest(path, digest, parse_remote_digest(remote.remote_digest(host, path), path))
    return text


def inspect_runtime(host: str, root: str, python: str) -> WrdsRuntime:
    """Measure a runtime through the exact worker bundle used by experiments.

    Args:
        host: SSH alias.
        root: Existing/private WRDS home directory for tool probes.
        python: Absolute interpreter path.

    Returns:
        A pin suitable for a run's runtime field.

    Raises:
        ValueError: For unsafe roots or an unexpected temporary-directory reply.
    """
    safe_label(host)
    absolute_path(python)
    absolute_path(root)
    if not root.startswith("/home/"):
        raise ValueError("WRDS_ROOT: tool probes belong beneath /home")
    remote.make_directory(host, root)
    directory = remote.run_remote(
        host, shlex.join(["mktemp", "-d", root + "/probe.XXXXXXXX"])
    ).strip()
    if not directory.startswith(root + "/probe.") or "\n" in directory:
        raise ValueError("WRDS_PROBE: unexpected probe directory")
    archive = directory + "/worker.pyz"
    upload_verified(host, archive, worker_bundle())
    text = remote.run_remote(host, shlex.join([python, "-s", archive, "inspect"]))
    return decode_wrds_runtime(load_json_str(text))


def check_local_code(run: WrdsRun, source_dir: pathlib.Path) -> dict[str, bytes]:
    """Read and validate the explicitly listed code snapshot before touching WRDS.

    Args:
        run: File declarations.
        source_dir: Directory relative local paths resolve against.

    Returns:
        Exact source bytes keyed by destination.

    Raises:
        ValueError: On changed source bytes or a snapshot larger than 100 MiB.
    """
    payloads: dict[str, bytes] = {}
    for item in run["code"]:
        payload = _test_hooks.read_bytes(source_dir / item["path"])
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"WRDS_CODE_DRIFT: {item['path']}")
        payloads[item["name"]] = payload
    if sum(len(value) for value in payloads.values()) > 100 * 1024 * 1024:
        raise ValueError(
            "WRDS_CODE_SIZE: code snapshots are limited to 100 MiB; use declared inputs"
        )
    return payloads


def preflight(run: WrdsRun, source_dir: pathlib.Path) -> WrdsRuntime:
    """Validate code, runtime and readable inputs without submitting an experiment.

    Input contents are hashed on the compute node before and after execution,
    not on the login node. Admission of the exact staged request is repeated
    by submit immediately before the held job is created.

    Args:
        run: Experiment.
        source_dir: Local code base.

    Returns:
        Verified runtime.

    Raises:
        ValueError: If the interpreter or packages differ from the document.
    """
    check_local_code(run, source_dir)
    actual = inspect_runtime(run["host"], run["root"] + "/tools", run["runtime"]["python"])
    if actual != run["runtime"]:
        raise ValueError("WRDS_RUNTIME_DRIFT: preflight runtime differs from the declared pin")
    for item in run["inputs"]:
        remote.run_remote(run["host"], shlex.join(["test", "-r", item["path"]]))
    return actual


def record_event(
    ledger: pathlib.Path, run: WrdsRun, attempt: str, event: str, job_id: str, timestamp: str
) -> None:
    """Append a lifecycle event to the repository's durable WRDS ledger.

    Args:
        ledger: Shared ledger file.
        run: Experiment.
        attempt: Attempt label.
        event: Observed lifecycle event.
        job_id: Scheduler id, empty only before submission.
        timestamp: UTC observation time.
    """
    entry: dict[str, JSONValue] = {
        "scheduler": "grid-engine",
        "host": run["host"],
        "project": run["project"],
        "name": run["name"],
        "identity": run_identity(run),
        "attempt": attempt,
        "directory": attempt_directory(run, attempt),
        "event": event,
        "job_id": job_id,
        "timestamp": timestamp,
    }
    _test_hooks.append_text(ledger, dump_json_str(entry) + "\n")


def job_id(run: WrdsRun, attempt: str) -> str:
    """Recover the scheduler id from the durable attempt, checking its identity.

    Args:
        run: Expected experiment.
        attempt: Attempt label.

    Returns:
        Verified job number; a missing/empty receipt never triggers resubmission.

    Raises:
        ValueError: If the remote run document has changed.
    """
    directory = attempt_directory(run, attempt)
    text = read_verified(run["host"], directory + "/run.json")
    if load_json_str(text) != execution_document(run):
        raise ValueError("WRDS_SPEC_DRIFT: remote experiment differs from the local definition")
    return wrds_scheduler.parse_job_id(
        remote.run_remote(run["host"], shlex.join(["cat", directory + "/job.id"]))
    )


def resume(run: WrdsRun, attempt: str, ledger: pathlib.Path, timestamp: str) -> str:
    """Record and release an already-submitted held job after an interrupted submit.

    Args:
        run: Experiment definition.
        attempt: Existing attempt; never created here.
        ledger: Shared local ledger.
        timestamp: UTC observation time.

    Returns:
        Scheduler job id.

    Raises:
        ValueError: When the job is not currently held.
    """
    identifier = job_id(run, attempt)
    state = wrds_scheduler.status(run["host"], identifier)
    if "h" not in state or state in {"completed", "failed", "accounting-pending"}:
        raise ValueError(f"WRDS_RESUME_STATE: job {identifier} is {state}, not held")
    record_event(ledger, run, attempt, "submitted-held", identifier, timestamp)
    remote.run_remote(run["host"], shlex.join(["qrls", identifier]))
    record_event(ledger, run, attempt, "released", identifier, timestamp)
    return identifier


def submit(
    run: WrdsRun, source_dir: pathlib.Path, attempt: str, ledger: pathlib.Path, timestamp: str
) -> str:
    """Preflight, stage an immutable attempt, submit held, record, then release.

    Args:
        run: Experiment definition.
        source_dir: Local code base.
        attempt: Explicit new attempt label.
        ledger: Shared local ledger.
        timestamp: UTC observation time.

    Returns:
        Durable scheduler job id.

    Raises:
        AppError: On admission, transfer or scheduler errors. An existing
            directory is refused; use resume after a lost submission response.
    """
    preflight(run, source_dir)
    payloads = check_local_code(run, source_dir)
    directory = attempt_directory(run, attempt)
    remote.make_directory(run["host"], str(pathlib.PurePosixPath(directory).parent))
    remote.run_remote(run["host"], shlex.join(["mkdir", "-m", "700", "--", directory]))
    record_event(ledger, run, attempt, "preparing", "", timestamp)
    for folder in ("code", "outputs"):
        remote.make_directory(run["host"], directory + "/" + folder)
    for name, payload in payloads.items():
        path = str(pathlib.PurePosixPath(directory) / "code" / name)
        remote.make_directory(run["host"], str(pathlib.PurePosixPath(path).parent))
        upload_verified(run["host"], path, payload)
    upload_verified(
        run["host"],
        directory + "/run.json",
        (dump_json_str(execution_document(run)) + "\n").encode(),
    )
    runner_sha = upload_verified(run["host"], directory + "/worker.pyz", worker_bundle())
    submission: dict[str, JSONValue] = {
        "identity": run_identity(run),
        "attempt": attempt,
        "worker_sha256": runner_sha,
        "created_at": timestamp,
    }
    upload_verified(
        run["host"], directory + "/submission.json", (dump_json_str(submission) + "\n").encode()
    )
    remote.run_remote(
        run["host"],
        shlex.join(
            [
                "chmod",
                "-R",
                "u-w",
                "--",
                directory + "/code",
                directory + "/run.json",
                directory + "/worker.pyz",
                directory + "/submission.json",
            ]
        ),
    )
    argv = wrds_scheduler.job_argv(run, directory, attempt, runner_sha)
    remote.run_remote(run["host"], shlex.join([argv[0], "-w", "v", *argv[1:]]))
    command = shlex.join([argv[0], "-h", *argv[1:]])
    remote.run_remote(run["host"], command + " > " + shlex.quote(directory + "/job.id"))
    return resume(run, attempt, ledger, timestamp)


def cancel(run: WrdsRun, attempt: str, ledger: pathlib.Path, timestamp: str) -> str:
    """Request cancellation and report the observed queue state afterwards.

    Args:
        run: Experiment definition.
        attempt: Attempt to cancel.
        ledger: Durable ledger.
        timestamp: UTC observation time.

    Returns:
        'cancelled' if absent from the queue, otherwise 'cancellation-pending'.
    """
    identifier = job_id(run, attempt)
    remote.run_remote(run["host"], shlex.join(["qdel", identifier]))
    state = wrds_scheduler.queue_state(remote.run_remote(run["host"], "qstat -xml"), identifier)
    outcome = "cancelled" if state == "absent" else "cancellation-pending"
    record_event(ledger, run, attempt, outcome, identifier, timestamp)
    return outcome


__all__ = [
    "attempt_directory",
    "cancel",
    "check_local_code",
    "inspect_runtime",
    "job_id",
    "preflight",
    "read_verified",
    "record_event",
    "resume",
    "submit",
    "upload_verified",
]
