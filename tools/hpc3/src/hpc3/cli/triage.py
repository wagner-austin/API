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
from hpc3.contracts.workspace import workspace_cluster
from hpc3.core import ledger, logs
from hpc3.core.remote import run_remote
from hpc3.core.squeue import parse_squeue_output, squeue_command
from hpc3.core.status import parse_sacct_output, sacct_command
from hpc3.core.triage import Finding, blocked_jobs, live_entries, silent_jobs, unaccounted_jobs

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

    entries = ledger.read(pathlib.Path(workspace["ledger"]), cluster)
    if entries == []:
        _test_hooks.emit("ledger is empty; nothing has been submitted from this machine")
        return 0

    job_ids = [entry["job_id"] for entry in entries]
    statuses = parse_sacct_output(run_remote(host, sacct_command(job_ids)), cluster)
    pending = parse_squeue_output(run_remote(host, squeue_command(job_ids)))

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

    _test_hooks.emit(
        f"{len(entries)} recorded, {len(still_live)} not finished, {len(findings)} finding(s)"
    )
    return 1 if findings != [] else 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = ["entrypoint", "main"]
