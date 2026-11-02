"""Measure OptionMetrics surface coverage on WRDS: python -m hpc3.cli.wrds_coverage.

Requires --host, --remote-dir, --output and --report-task. Runs one
aggregates-only query job; no dataset rows are downloaded. The measured
counts are posted on the report task by the program.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.load_report import post_load_report, report_identity

from hpc3.cli import _config, _test_hooks
from hpc3.contracts.wrds_coverage import CoverageReceipt
from hpc3.core.wrds_coverage import option_coverage


def report(receipt: CoverageReceipt) -> str:
    """Render the load report a coverage measurement posts.

    Args:
        receipt: The validated measurement's receipt.

    Returns:
        One paragraph naming the job, the digests and every relation's counts.
    """
    relations = "; ".join(
        f"{row['library']}.{row['relation']}: {row['rows']} rows, {row['securities']} "
        f"securities, {row['first_date'] or 'no dates'} to {row['last_date'] or 'no dates'}, "
        f"slice {row['slice_rows']} rows over {row['slice_securities']} securities"
        for row in receipt["rows"]
    )
    return (
        f"WRDS OPTION COVERAGE LOAD-REPORT job {receipt['job_id']} on {receipt['host']} "
        f"({receipt['scheduler']}): coverage sha256 {receipt['coverage_sha256']}, "
        f"query sha256 {receipt['query_sha256']}; slice {receipt['slice']}; {relations}. "
        "Aggregate counts and bounding dates only; no dataset rows."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Measure coverage, report its counts and post them on the report task.

    Args:
        argv: Command arguments or process arguments.

    Returns:
        Zero after a validated measurement whose report the board accepted.
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
    body = report(option_coverage(host, remote, output))
    _test_hooks.emit(body)
    _test_hooks.emit(f"Coverage CSV: {output}; scheduler receipt: {output}.receipt.json")
    post_load_report(
        post=_test_hooks.http_post,
        read_text=_config.stack_file_text,
        task_id=task_id,
        body=body,
        family="wrds",
    )
    _test_hooks.emit(f"Posted on board task {task_id}")
    return 0


def entrypoint() -> None:
    """Run the command through the common CLI error boundary."""
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main", "report"]


if __name__ == "__main__":
    entrypoint()
