"""CLI: find every job of ours that is wrong in a way that looks fine.

Usage:
    hpc3-triage --config hpc3.json

The ledger it reconciles and the staleness threshold it applies both come
from the workspace, so this command necessarily reads the same record
``hpc3-submit`` wrote. A ``--ledger`` flag here would be the single easiest
way to get a clean board while jobs run unwatched.

Reconciles the local ledger against the cluster and reports three conditions
that a normal status check cannot distinguish from health:

* **blocked** -- pending on a reason that will never resolve. On HPC3, 261 of
  621 pending GPU jobs were sitting on ``DependencyNeverSatisfied``.
* **unaccounted** -- we recorded submitting it and accounting has never heard
  of it. No cluster-side query can find these, because the evidence is the
  absence of a cluster-side record.
* **silent** -- ``RUNNING``, holding GPUs, and its log has stopped growing.

Exits non-zero when anything is found. A triage command that reports problems
and exits 0 is a triage command whose output gets skimmed.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from hpc3.cli import _argv, _config, _fatal, _test_hooks
from hpc3.contracts.pending import PendingJob
from hpc3.contracts.workspace import workspace_cluster
from hpc3.core import ledger, logs
from hpc3.core.remote import run_remote
from hpc3.core.squeue import parse_squeue_output, squeue_command
from hpc3.core.status import parse_sacct_output, sacct_command
from hpc3.core.triage import (
    Finding,
    blocked_jobs,
    closures_for,
    live_entries,
    open_entries,
    silent_jobs,
    unaccounted_jobs,
)

_FLAGS = (_config.CONFIG_FLAG,)


def main(argv: Sequence[str] | None = None) -> int:
    """Reconcile the ledger against the cluster and report anything wrong.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when every recorded job is either finished or healthily
        in progress, and 1 when anything was found.

    Raises:
        ValueError: If ``--config`` is missing or an argument is unknown.
        AppError: If a remote command fails or its output cannot be read.
        JSONTypeError: If the workspace is malformed, or the ledger holds a
            malformed record. The ledger record is not
            skipped: a record that cannot be read is a job that cannot be
            found, which is the condition this command exists to catch.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = _argv.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)
    cluster = workspace_cluster(workspace)
    host = workspace["host"]
    quiet_seconds = workspace["quiet_seconds"]

    ledger_path = pathlib.Path(workspace["ledger"])
    closures_path = ledger.closure_path(ledger_path)

    recorded = ledger.read(ledger_path, cluster)
    if recorded == []:
        _test_hooks.emit("ledger is empty; nothing has been submitted from this machine")
        return 0

    # Jobs already observed to have ended are not asked about again. Without
    # this, every job older than the cluster's sacct retention window becomes
    # a permanent 'unaccounted' finding, and a board that is always red is the
    # same as no board.
    closed = ledger.read_closures(closures_path)
    entries = open_entries(recorded, closed)
    if entries == []:
        _test_hooks.emit(f"{len(recorded)} recorded, all closed; nothing left to reconcile")
        return 0

    job_ids = [entry["job_id"] for entry in entries]
    statuses = parse_sacct_output(run_remote(host, sacct_command(job_ids)), cluster)

    # squeue is asked ONLY about jobs accounting reports as queued. It holds a
    # job for minutes after it ends and then forgets it, and `squeue -j` on an
    # id it no longer holds exits non-zero with "Invalid job id specified" --
    # measured on the real cluster against a job that had completed perfectly.
    # That is not a triage failure, but asking would report it as one, and the
    # blocked-job check only concerns pending jobs anyway.
    pending_ids = [status["job_id"] for status in statuses if status["state"] == "PENDING"]
    pending: list[PendingJob] = []
    if pending_ids != []:
        pending = parse_squeue_output(run_remote(host, squeue_command(pending_ids)))

    still_live = live_entries(entries, statuses)
    ages = logs.log_ages(host, still_live)

    findings: list[Finding] = [
        *unaccounted_jobs(entries, statuses),
        *blocked_jobs(pending),
        *silent_jobs(statuses, ages, quiet_seconds=quiet_seconds),
    ]

    for finding in findings:
        _test_hooks.emit(
            f"{finding.kind.upper()} {finding.job_id} {finding.name}: {finding.detail}"
        )

    # Written AFTER the findings are built, so this run still reports on a job
    # it is closing, and only then stops asking. Recorded whatever the verdict
    # was: a job that ended is a job accounting will eventually forget.
    newly_closed = closures_for(statuses, closed_at=_test_hooks.now_iso())
    for closure in newly_closed:
        ledger.append_closure(closures_path, closure)

    _test_hooks.emit(
        f"{len(recorded)} recorded, {len(entries)} open, {len(still_live)} not finished, "
        f"{len(findings)} finding(s), {len(newly_closed)} newly closed"
    )
    return 1 if findings != [] else 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = ["entrypoint", "main"]
