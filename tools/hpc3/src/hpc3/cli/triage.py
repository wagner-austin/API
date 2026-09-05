"""CLI: find every job of ours that is wrong in a way that looks fine.

Usage:
    hpc3-triage --config hpc3.json

The ledger it reconciles and the staleness threshold it applies both come
from the workspace, so this command necessarily reads the same record
``hpc3-submit`` wrote. A ``--ledger`` flag here would be the single easiest
way to get a clean board while jobs run unwatched.

Reconciles the local ledger against the cluster -- in both directions -- and
reports five conditions that a normal status check cannot distinguish from
health:

* **blocked** -- pending on a reason that will never resolve. On HPC3, 261 of
  621 pending GPU jobs were sitting on ``DependencyNeverSatisfied``.
* **unaccounted** -- we recorded submitting it and accounting has never heard
  of it. No cluster-side query can find these, because the evidence is the
  absence of a cluster-side record.
* **unclaimed** -- the cluster is holding it and our ledger has never heard of
  it. No ledger-side query can find these either, for the mirror reason: the
  evidence is the absence of a LOCAL record, so the enumeration has to start
  from the account.
* **silent** -- ``RUNNING``, holding GPUs, and its log has stopped growing.
* **oversized** -- the project asks Slurm for far more wall clock than its
  work has ever taken, so its jobs wait for a hole they never needed. The one
  finding drawn from history rather than from the queue, and the only one
  needing no cluster query.

Exits non-zero when anything is found. A triage command that reports problems
and exits 0 is a triage command whose output gets skimmed.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.contracts.pending import PendingJob
from hpc3.contracts.workspace import workspace_cluster
from hpc3.core import ledger, logs
from hpc3.core.remote import run_remote, run_remote_batched
from hpc3.core.rightsize import describe, oversized_projects
from hpc3.core.squeue import (
    account_command,
    parse_account_output,
    parse_squeue_output,
    squeue_commands,
)
from hpc3.core.status import parse_sacct_output, sacct_commands
from hpc3.core.triage import (
    Finding,
    blocked_jobs,
    closures_for,
    live_entries,
    open_entries,
    silent_jobs,
    unaccounted_jobs,
    unclaimed_jobs,
)

_FLAGS = (_config.CONFIG_FLAG,)


def _report(findings: Sequence[Finding]) -> int:
    """Emit every finding and decide the exit status.

    Exists because there are now three places the command can finish -- an
    empty ledger, a fully closed one, and the full reconciliation -- and the
    first two used to return 0 without reporting anything. A separate exit
    path is exactly where a finding gets dropped silently.

    Args:
        findings: Everything found, in the order it was found.

    Returns:
        1 when anything was found, 0 otherwise.
    """
    for finding in findings:
        _test_hooks.emit(
            f"{finding.kind.upper()} {finding.job_id} {finding.name}: {finding.detail}"
        )
    return 1 if findings != [] else 0


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
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)
    cluster = workspace_cluster(workspace)
    host = workspace["host"]
    quiet_seconds = workspace["quiet_seconds"]

    ledger_path = pathlib.Path(workspace["ledger"])
    closures_path = ledger.closure_path(ledger_path)

    recorded = ledger.read(ledger_path, cluster)

    # Asked FIRST, and asked whatever the ledger holds. Both of the early
    # returns below used to exit 0 before any cluster query ran, so a machine
    # with an empty ledger and seven jobs on the cluster reported "nothing has
    # been submitted from this machine" and exited clean -- which is the
    # strongest form of the condition this query exists to find, reported as
    # health. The enumeration does not depend on the ledger holding anything,
    # so it does not wait for it to.
    account = parse_account_output(run_remote(host, account_command()))
    findings: list[Finding] = unclaimed_jobs(recorded, account)

    # Read here, and the right-sizing done here, because BOTH early returns
    # below are reachable with a project that is over-requesting -- a fully
    # closed ledger is precisely a project whose work has all finished, which
    # is when its runtimes are most worth comparing against its request. This
    # check sat after those returns for one revision and was silent in the
    # steady state, which is the state it exists to describe.
    #
    # It therefore reports from closures written by EARLIER runs, so a project
    # becomes right-sizeable on the run after the one that finishes its second
    # member. That delay is real and preferable to a check that is skipped.
    closed = ledger.read_closures(closures_path)
    findings.extend(
        Finding(item["evidence"], item["project"], "oversized", describe(item))
        for item in oversized_projects(workspace["projects"], recorded, closed)
    )

    if recorded == []:
        _test_hooks.emit("ledger is empty; nothing has been submitted from this machine")
        return _report(findings)

    # Jobs already observed to have ended are not asked about again. Without
    # this, every job older than the cluster's sacct retention window becomes
    # a permanent 'unaccounted' finding, and a board that is always red is the
    # same as no board.
    entries = open_entries(recorded, closed)
    if entries == []:
        _test_hooks.emit(f"{len(recorded)} recorded, all closed; nothing left to reconcile")
        return _report(findings)

    # Batched, because this is the one query here whose size is the LEDGER's
    # rather than a command line's. At 6645 recorded rows the single-command
    # form built a ~70 KB argv and died in CreateProcess before ssh was
    # spawned, so triage was unrunnable on this machine and said so in a
    # FileNotFoundError that named no command (2026-09-05).
    job_ids = [entry["job_id"] for entry in entries]
    statuses = parse_sacct_output(run_remote_batched(host, sacct_commands(job_ids)), cluster)

    # squeue is asked ONLY about jobs accounting reports as queued. It holds a
    # job for minutes after it ends and then forgets it, and `squeue -j` on an
    # id it no longer holds exits non-zero with "Invalid job id specified" --
    # measured on the real cluster against a job that had completed perfectly.
    # That is not a triage failure, but asking would report it as one, and the
    # blocked-job check only concerns pending jobs anyway.
    pending_ids = [status["job_id"] for status in statuses if status["state"] == "PENDING"]
    pending: list[PendingJob] = []
    if pending_ids != []:
        pending = parse_squeue_output(run_remote_batched(host, squeue_commands(pending_ids)))

    still_live = live_entries(entries, statuses)
    ages = logs.log_ages(host, still_live)

    findings.extend(
        [
            *unaccounted_jobs(entries, statuses),
            *blocked_jobs(pending),
            *silent_jobs(statuses, ages, quiet_seconds=quiet_seconds),
        ]
    )

    # Written AFTER the job findings are built, so this run still reports on a
    # job it is closing, and only then stops asking. Recorded whatever the
    # verdict was: a job that ended is a job accounting will eventually forget.
    newly_closed = closures_for(statuses, closed_at=_test_hooks.now_iso())
    for closure in newly_closed:
        ledger.append_closure(closures_path, closure)

    exit_code = _report(findings)

    _test_hooks.emit(
        f"{len(recorded)} recorded, {len(account)} on the cluster, {len(entries)} open, "
        f"{len(still_live)} not finished, {len(findings)} finding(s), "
        f"{len(newly_closed)} newly closed"
    )
    return exit_code


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
