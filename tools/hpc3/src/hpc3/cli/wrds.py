"""WRDS experiments: inspect, preflight, submit, resume, status, cancel, collect, trace, report.

Run as python -m hpc3.cli.wrds ACTION. Every execution action requires a
specification; every attempt action also names an explicit attempt. Structured
stdout contains metadata only, never payload logs or downloaded dataset rows.
``report`` measures the attempt on WRDS and posts the reading on the board task
named by ``--report-task``, as that task's ``load-report-<id>`` identity.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import JSONObject, dump_json_str, load_json_str, narrow_json_to_dict

from hpc3.cli import _test_hooks
from hpc3.cli._load_report import post_load_report, report_identity
from hpc3.contracts.wrds_run import WrdsRun, decode_wrds_run, encode_wrds_runtime, run_identity
from hpc3.core import _test_hooks as core_hooks
from hpc3.core import wrds_attempt_report, wrds_cleanup, wrds_jobs, wrds_results, wrds_scheduler
from hpc3.core.checkout import shared_directory
from hpc3.core.wrds_bundle import worker_bundle
from hpc3.wrds import _test_hooks as worker_hooks

_FLAGS = ("--spec", "--attempt", "--output", "--host", "--root", "--python", "--report-task")
_ACTIONS = {
    "bundle",
    "inspect",
    "preflight",
    "submit",
    "resume",
    "status",
    "cancel",
    "collect",
    "trace",
    "report",
    "scratch",
    "scratch-report",
}


def _trace(ledger: pathlib.Path, identity: str, attempt: str) -> None:
    """Emit only ledger events belonging to the requested attempt.

    Args:
        ledger: Repository ledger.
        identity: Full experiment identity.
        attempt: Attempt label.
    """
    for line in core_hooks.read_bytes(ledger).decode().splitlines():
        event = narrow_json_to_dict(load_json_str(line))
        if event.get("identity") == identity and event.get("attempt") == attempt:
            _test_hooks.emit(dump_json_str(event))


def _prepare(action: str, flags: dict[str, str]) -> None:
    """Prepare payload bytes or runtime pins before an experiment exists.

    Args:
        action: Validated bundle or inspect action.
        flags: Parsed preparation arguments.
    """
    if action == "bundle":
        output = pathlib.Path(cli_args.require_flag(flags, "--output"))
        worker_hooks.write(output, worker_bundle())
        _test_hooks.emit(
            dump_json_str({"output": str(output), "sha256": worker_hooks.digest_file(output)})
        )
    else:
        runtime = wrds_jobs.inspect_runtime(
            cli_args.require_flag(flags, "--host"),
            cli_args.require_flag(flags, "--root"),
            cli_args.require_flag(flags, "--python"),
        )
        _test_hooks.emit(dump_json_str(encode_wrds_runtime(runtime)))


def _evidence(run: WrdsRun, attempt: str, flags: dict[str, str], action: str) -> None:
    """Measure the requested native evidence and publish through one service identity.

    Args:
        run: Frozen native experiment.
        attempt: Exact attempt to measure.
        flags: Explicit report destination and task.
        action: Submit an observer, reconcile its report, or report the subject.

    Raises:
        ValueError: Invalid task or inconsistent native evidence.
        AppError: Transport or report publication fails.
    """
    if action == "scratch":
        output = pathlib.Path(cli_args.require_flag(flags, "--output"))
        observer = wrds_cleanup.submit(run, attempt, output)
        _test_hooks.emit(dump_json_str({"observer_job_id": observer, "output": str(output)}))
        return
    task_id = cli_args.require_flag(flags, "--report-task")
    report_identity(task_id, "wrds")
    result: JSONObject
    if action == "scratch-report":
        output = pathlib.Path(cli_args.require_flag(flags, "--output"))
        measured = wrds_cleanup.measure(run, attempt, output)
        if measured is None:
            _test_hooks.emit(dump_json_str({"state": "accounting-pending", "output": str(output)}))
            return
        body = measured
        result = {"report_task": task_id, "output": str(output)}
    else:
        body, completion_digest = wrds_attempt_report.measure(run, attempt)
        result = {
            "report_task": task_id,
            "identity": run_identity(run),
            "attempt": attempt,
            "completion_sha256": completion_digest,
        }
    post_load_report(task_id, body, "wrds")
    _test_hooks.emit(dump_json_str(result))


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one WRDS lifecycle action.

    Args:
        argv: Command arguments, or process arguments.

    Returns:
        Zero for a successful control-plane operation. A status report may
        describe a failed experiment; collect refuses every non-success state.

    Raises:
        ValueError: For unknown actions or missing required parameters.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] not in _ACTIONS:
        raise ValueError("WRDS_ACTION: expected " + ", ".join(sorted(_ACTIONS)))
    action = tokens[0]
    flags = cli_args.parse_single_flags(tokens[1:], _FLAGS)
    if action in {"bundle", "inspect"}:
        _prepare(action, flags)
        return 0
    spec_path = pathlib.Path(cli_args.require_flag(flags, "--spec")).resolve()
    run = decode_wrds_run(load_json_str(core_hooks.read_bytes(spec_path).decode()))
    ledger = shared_directory(spec_path.parent) / "wrds-ledger.jsonl"
    if action == "preflight":
        wrds_jobs.preflight(run, spec_path.parent)
        _test_hooks.emit(dump_json_str({"identity": run_identity(run), "preflight": "passed"}))
        return 0
    attempt = cli_args.require_flag(flags, "--attempt")
    now = _test_hooks.now_iso()
    if action == "submit":
        identifier = wrds_jobs.submit(run, spec_path.parent, attempt, ledger, now)
        _test_hooks.emit(
            dump_json_str(
                {"job_id": identifier, "directory": wrds_jobs.attempt_directory(run, attempt)}
            )
        )
    elif action == "resume":
        _test_hooks.emit(dump_json_str({"job_id": wrds_jobs.resume(run, attempt, ledger, now)}))
    elif action == "status":
        identifier = wrds_jobs.job_id(run, attempt)
        _test_hooks.emit(
            dump_json_str(
                {"job_id": identifier, "state": wrds_scheduler.status(run["host"], identifier)}
            )
        )
    elif action == "cancel":
        _test_hooks.emit(dump_json_str({"state": wrds_jobs.cancel(run, attempt, ledger, now)}))
    elif action == "collect":
        output = pathlib.Path(cli_args.require_flag(flags, "--output"))
        result = wrds_results.collect(run, attempt, output, ledger, now)
        _test_hooks.emit(dump_json_str({"output": str(output), "identity": result["identity"]}))
    elif action in {"report", "scratch", "scratch-report"}:
        _evidence(run, attempt, flags, action)
    else:
        _trace(ledger, run_identity(run), attempt)
    return 0


def entrypoint() -> None:
    """Use the same error boundary and report streams as the HPC3 commands."""
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main"]

if __name__ == "__main__":
    entrypoint()
