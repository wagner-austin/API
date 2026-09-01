"""CLI: report what jobs are doing and what they have actually charged.

Usage:
    hpc3-watch --config hpc3.json --job 55519937
    hpc3-watch --config hpc3.json --job 55519937,55520509,55520564
    hpc3-watch --config hpc3.json --job 55519937,55520509 --until-done 1

The reported cost applies the partition's usage factor, so a job on
``free-gpu`` reports 0.0000 SU however long it ran, and a short job on a
billing partition reports the fraction that ``sbank`` rounds away.

A comma-separated list becomes one ``sacct`` call, so every row in a sweep
is read from the same moment. Ids that accounting does not know are reported
by name at the end rather than silently omitted.

``--until-done 1`` re-reads accounting until every requested job reaches a
terminal state, emitting a row only when a job's state CHANGES, then the
ordinary summary. It exists because waiting on a batch used to mean a
hand-rolled ssh polling loop per session -- six were written in one day, and
one of them declared a running panel drained on a shell quoting bug the
loop's author could not see. ``--poll-seconds`` sets the cadence.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core import cli_args

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.layout import project_of
from hpc3.contracts.status import JobState, JobStatus, gpu_hours, is_terminal, service_units
from hpc3.contracts.workspace import Workspace, workspace_cluster
from hpc3.core.budget import check_consumption
from hpc3.core.remote import run_remote
from hpc3.core.status import parse_sacct_output, sacct_command

_FLAGS = (_config.CONFIG_FLAG, "--job", "--until-done", "--poll-seconds")

#: Seconds between accounting reads in follow mode, unless the caller sets
#: a cadence. A minute keeps a multi-hour batch legible without hammering
#: the scheduler the way an eager loop would.
POLL_SECONDS_DEFAULT = 60

#: Consecutive reads that may come back knowing NONE of the requested jobs
#: before follow mode rules the ids wrong and raises. Accounting can lag a
#: fresh submission by a read or two, so one empty answer is patience -- but
#: a follow loop with no such bound would spin forever on a mistyped id,
#: reporting nothing, which is the failure the one-shot mode's raise exists
#: to prevent.
UNKNOWN_POLL_LIMIT = 5


def format_status(status: JobStatus, cluster: ClusterFacts) -> str:
    """Render one status row for the report.

    Args:
        status: The job's accounting row.
        cluster: The cluster whose usage factors turn the billing rate into
            a real cost.

    Returns:
        A single line naming the job, its state, elapsed time, cost and
        placement. Cost carries four decimals because a real charge can be a
        small fraction of a unit and rounding it to zero is how a billed job
        gets read as free.
    """
    marker = "final" if is_terminal(status["state"]) else "live "
    node = status["node_list"] if status["node_list"] != "" else "-"
    return (
        f"{marker} {status['job_id']} {status['name']} "
        f"{status['state']} {status['elapsed_seconds']}s "
        f"{service_units(status, cluster):.4f} SU on {status['partition']} @ {node}"
    )


def _budget_groups(
    rows: Sequence[JobStatus], workspace: Workspace
) -> tuple[list[tuple[str, list[JobStatus]]], list[JobStatus]]:
    """Split accounting rows by the project whose cap governs each one.

    Watch is handed job IDs rather than run documents, so before caps became
    per-project this command applied the workspace's single cap to whatever
    it was shown. That was the defect: a workspace declaring 0.5 GPU-hours
    would fail an ``mi`` job submitted under a declared 12.0, and the same
    workspace pointed at a small job would pass one submitted under a cap it
    had already exceeded. Neither reading was about the job.

    The project comes from the job's own name, which ``qualified_name``
    guarantees is ``<project>.<name>``; accounting already carries it because
    the prefix exists to make a shared ``squeue`` self-describing.

    Args:
        rows: Accounting rows, in the order ``sacct`` returned them.
        workspace: The decoded workspace, for the declared project table.

    Returns:
        The rows grouped under each declared project in name order, and the
        rows belonging to no declared project.
    """
    declared = workspace["projects"]
    grouped: dict[str, list[JobStatus]] = {}
    unclaimed: list[JobStatus] = []
    for status in rows:
        project = project_of(status["name"])
        if project is None or project not in declared:
            unclaimed.append(status)
            continue
        grouped.setdefault(project, []).append(status)
    return sorted(grouped.items()), unclaimed


def _read_rows(host: str, requested: Sequence[str], cluster: ClusterFacts) -> list[JobStatus]:
    """Read one accounting snapshot for the whole requested set.

    One sacct call for the whole set: six separate calls would observe six
    different moments, and a sweep's rows are only comparable if they came
    from the same one.

    Args:
        host: The cluster's SSH host alias.
        requested: The job ids asked about.
        cluster: The cluster whose facts decode the accounting output.

    Returns:
        The accounting rows, in the order ``sacct`` returned them.

    Raises:
        AppError: If the remote command fails or the output is malformed.
    """
    output = run_remote(host, sacct_command(requested))
    return parse_sacct_output(output, cluster)


def _follow(
    host: str, requested: Sequence[str], cluster: ClusterFacts, poll_seconds: int
) -> list[JobStatus]:
    """Re-read accounting until every requested job is terminal.

    Emits a row only when a job's state CHANGES, so a six-hour panel writes
    a legible transition log rather than a screenful of identical RUNNING
    lines per minute.

    Args:
        host: The cluster's SSH host alias.
        requested: The job ids being waited on.
        cluster: The cluster whose facts decode the accounting output.
        poll_seconds: Seconds between reads.

    Returns:
        The final accounting snapshot, every row terminal.

    Raises:
        ValueError: When :data:`UNKNOWN_POLL_LIMIT` consecutive reads know
            none of the requested jobs -- the same wrong-id failure the
            one-shot mode raises on its single read.
        AppError: If the remote command fails or the output is malformed.
    """
    seen: dict[str, JobState] = {}
    unknown_reads = 0
    while True:
        rows = _read_rows(host, requested, cluster)
        if rows == []:
            unknown_reads += 1
            if unknown_reads >= UNKNOWN_POLL_LIMIT:
                raise ValueError(f"sacct knows no job in {list(requested)} on {host}")
            _test_hooks.emit(f"accounting knows none of {len(requested)} job(s) yet; waiting")
            _test_hooks.sleep(poll_seconds)
            continue
        unknown_reads = 0
        for status in rows:
            if seen.get(status["job_id"]) != status["state"]:
                seen[status["job_id"]] = status["state"]
                _test_hooks.emit(format_status(status, cluster))
        ids = {status["job_id"] for status in rows}
        if all(is_terminal(status["state"]) for status in rows) and set(requested) <= ids:
            return rows
        _test_hooks.sleep(poll_seconds)


def _summarize(
    rows: Sequence[JobStatus],
    requested: Sequence[str],
    workspace: Workspace,
    cluster: ClusterFacts,
) -> None:
    """Emit the totals, the missing ids, and the per-project budget verdicts.

    Args:
        rows: The accounting rows being summarized.
        requested: The job ids originally asked about.
        workspace: The decoded workspace, for the budget caps.
        cluster: The cluster whose usage factors price the rows.

    Raises:
        AppError: When a project's consumption exceeds its declared budget.
    """
    total = 0.0
    total_gpu_hours = 0.0
    counts: dict[JobState, int] = {}
    for status in rows:
        total += service_units(status, cluster)
        total_gpu_hours += gpu_hours(status)
        counts[status["state"]] = counts.get(status["state"], 0) + 1

    tally = " ".join(f"{state}={counts[state]}" for state in sorted(counts))
    _test_hooks.emit(f"total {total:.4f} SU across {len(rows)} row(s)")
    _test_hooks.emit(f"gpu-hours {total_gpu_hours:.2f}")
    _test_hooks.emit(f"states {tally}")

    missing = sorted(set(requested) - {status["job_id"] for status in rows})
    if missing != []:
        # Reported rather than raised: the rows that came back are real and
        # worth showing. Silence about the rest is what would mislead.
        _test_hooks.emit(f"NOT FOUND {','.join(missing)}")

    # Checked last: an overrun raises, and the rows above are worth seeing
    # whether or not it does. Each group is checked against ITS OWN project's
    # cap, so the ceiling this command enforces is the same one the submitting
    # command projected against -- which was the claim this comment made while
    # a single workspace cap was applied to every job it was shown.
    grouped, unclaimed = _budget_groups(rows, workspace)
    if unclaimed != []:
        # Reported, never silently skipped and never checked against someone
        # else's cap. A job this workspace did not submit has no declared
        # budget, and saying so is the honest reading.
        ids = ",".join(status["job_id"] for status in unclaimed)
        _test_hooks.emit(f"NO DECLARED BUDGET {ids}")
    for project, group in grouped:
        check_consumption(workspace["projects"][project]["budget"], group, cluster)
        _test_hooks.emit(f"budget OK {project}")


def main(argv: Sequence[str] | None = None) -> int:
    """Report the state and cost of one job, or follow a set to completion.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when accounting returned at least one row.

    Raises:
        ValueError: If a required flag is missing, an argument is unknown,
            or ``--poll-seconds`` is not a positive whole number.
        AppError: If the remote command fails, accounting output is
            malformed, or the job id is unknown to ``sacct``. An unknown job
            is a failure, not an empty report: the caller asked about a
            specific job and silence would read as "not running yet".
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)
    cluster = workspace_cluster(workspace)
    host = workspace["host"]
    requested = [part for part in cli_args.require_flag(parsed, "--job").split(",") if part != ""]
    if requested == []:
        raise ValueError("--job must name at least one job id")
    poll_seconds = int(parsed.get("--poll-seconds", str(POLL_SECONDS_DEFAULT)))
    if poll_seconds <= 0:
        raise ValueError(f"--poll-seconds must be positive, got {poll_seconds}")

    if parsed.get("--until-done", "0") != "0":
        rows = _follow(host, requested, cluster, poll_seconds)
    else:
        rows = _read_rows(host, requested, cluster)
        if rows == []:
            raise ValueError(f"sacct knows no job in {requested} on {host}")
        for status in rows:
            _test_hooks.emit(format_status(status, cluster))
    _summarize(rows, requested, workspace, cluster)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = ["entrypoint", "format_status", "main"]


if __name__ == "__main__":
    entrypoint()
