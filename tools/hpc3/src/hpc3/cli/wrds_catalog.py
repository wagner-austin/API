"""Inventory WRDS filesystem metadata: python -m hpc3.cli.wrds_catalog.

Requires --host, --remote-dir, --output and --report-task. No data rows are
downloaded; the measured counts are posted on the report task by the program.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli

from hpc3.cli import _test_hooks
from hpc3.cli._load_report import post_load_report, report_identity
from hpc3.contracts.wrds_catalog import CatalogRun
from hpc3.core.wrds_catalog import inventory


def report(run: CatalogRun) -> str:
    """Render the load report a filesystem inventory posts.

    Args:
        run: The verified catalog and its receipt.

    Returns:
        One paragraph naming the job, the catalog digest and every count.
    """
    entries = run["entries"]
    receipt = run["receipt"]
    files = [entry for entry in entries if entry["kind"] == "file"]
    readable = [entry for entry in files if entry["readable"]]
    denied = sum(entry["kind"] == "denied_directory" for entry in entries)
    unresolved = sum(entry["kind"] == "unresolved_link" for entry in entries)
    return (
        f"WRDS FILESYSTEM LOAD-REPORT job {receipt['job_id']} on {receipt['host']} "
        f"({receipt['scheduler']}): catalog sha256 {receipt['catalog_sha256']}; "
        f"{len(files)} visible files, {len(readable)} readable files, "
        f"{sum(entry['bytes'] for entry in readable)} readable logical bytes, "
        f"{denied} inaccessible directories, {unresolved} unresolved links; "
        f"probe sha256 {receipt['source_sha256']}. A visible file is not a readable one, "
        "and neither is a SQL privilege. Metadata only; no dataset rows."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run an inventory, report its counts and post them on the report task.

    Args:
        argv: Command arguments or process arguments.

    Returns:
        Zero after a complete inventory whose report the board accepted.
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
    run = inventory(host, remote, output)
    body = report(run)
    _test_hooks.emit(body)
    _test_hooks.emit(f"Metadata catalog: {output}; scheduler receipt: {output}.receipt.json")
    post_load_report(task_id, body, "wrds")
    _test_hooks.emit(f"Posted on board task {task_id}")
    return 0


def entrypoint() -> None:
    """Run the command through the common CLI error boundary."""
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main", "report"]


if __name__ == "__main__":
    entrypoint()
