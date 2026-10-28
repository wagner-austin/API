"""Inventory WRDS SQL privileges: python -m hpc3.cli.wrds_sql_catalog.

Requires --host, --remote-dir, --output and --report-task. Reads pg_catalog
metadata and opens each accessible relation with a zero-row query; no data
rows are downloaded. The measured counts are posted on the report task by the
program.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli

from hpc3.cli import _test_hooks
from hpc3.cli._load_report import post_load_report, report_identity
from hpc3.contracts.wrds_sql import SqlReceipt
from hpc3.core.wrds_sql import sql_inventory


def report(receipt: SqlReceipt) -> str:
    """Render the load report a SQL inventory posts.

    Args:
        receipt: The validated run's receipt.

    Returns:
        One paragraph naming both jobs, the catalog digest and every count.
    """
    counts = receipt["status_counts"]
    statuses = ", ".join(f"{status} {count}" for status, count in counts.items())
    return (
        f"WRDS SQL LOAD-REPORT jobs {receipt['metadata_job_id']} (catalog) and "
        f"{receipt['validation_job_id']} (validation) on {receipt['host']} "
        f"({receipt['scheduler']}): catalog sha256 {receipt['catalog_sha256']}; "
        f"{receipt['visible_relations']} visible relations, "
        f"{receipt['accessible_relations']} accessible relations "
        f"({counts['direct_table_granted']} direct tables, "
        f"{counts['alias_and_underlying_granted']} aliases), every one opened by "
        "SELECT * LIMIT 0 in one read-only transaction; status counts: "
        f"{statuses}; query sha256 {receipt['query_sha256']}, validation sha256 "
        f"{receipt['validation_sha256']}. A visible relation is not an accessible one. "
        "Metadata only; no dataset rows."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SQL inventory, report its counts and post them on the report task.

    Args:
        argv: Command arguments or process arguments.

    Returns:
        Zero after a validated inventory whose report the board accepted.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    flags = cli_args.parse_single_flags(
        tokens, ("--host", "--remote-dir", "--output", "--report-task")
    )
    host = cli_args.require_flag(flags, "--host")
    remote = cli_args.require_flag(flags, "--remote-dir")
    output = pathlib.Path(cli_args.require_flag(flags, "--output"))
    task_id = cli_args.require_flag(flags, "--report-task")
    report_identity(task_id, "wrds")  # refuses a malformed task before a job is spent
    run = sql_inventory(host, remote, output)
    body = report(run["receipt"])
    _test_hooks.emit(body)
    _test_hooks.emit(
        f"SQL catalog: {output}; accessible shortlist: {output}.accessible.csv; "
        f"scheduler receipt: {output}.receipt.json"
    )
    post_load_report(task_id, body, "wrds")
    _test_hooks.emit(f"Posted on board task {task_id}")
    return 0


def entrypoint() -> None:
    """Run the command through the common CLI error boundary."""
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main", "report"]


if __name__ == "__main__":
    entrypoint()
