"""CLI: report what jobs are doing and what they have actually charged.

Usage:
    hpc3-watch --config hpc3.json --job 55519937
    hpc3-watch --config hpc3.json --job 55519937,55520509,55520564

The reported cost applies the partition's usage factor, so a job on
``free-gpu`` reports 0.0000 SU however long it ran, and a short job on a
billing partition reports the fraction that ``sbank`` rounds away.

A comma-separated list becomes one ``sacct`` call, so every row in a sweep
is read from the same moment. Ids that accounting does not know are reported
by name at the end rather than silently omitted.
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

_FLAGS = (_config.CONFIG_FLAG, "--job")


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


def main(argv: Sequence[str] | None = None) -> int:
    """Report the state and cost of one job.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when accounting returned at least one row.

    Raises:
        ValueError: If a required flag is missing or an argument is unknown.
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

    # One sacct call for the whole set: six separate calls would observe six
    # different moments, and a sweep's rows are only comparable if they came
    # from the same one.
    output = run_remote(host, sacct_command(requested))
    rows = parse_sacct_output(output, cluster)
    if rows == []:
        raise ValueError(f"sacct knows no job in {requested} on {host}")

    total = 0.0
    total_gpu_hours = 0.0
    counts: dict[JobState, int] = {}
    for status in rows:
        _test_hooks.emit(format_status(status, cluster))
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
