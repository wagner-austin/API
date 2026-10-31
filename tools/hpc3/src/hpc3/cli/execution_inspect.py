"""Recover one research submission and measure its native execution evidence."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import JSONObject, dump_json_str, require_dict, require_str
from platform_core.load_report import post_load_report, report_identity
from platform_core.run_record import encode_run_record

from hpc3.cli import _config, _test_hooks
from hpc3.contracts.ledger import encode_ledger_entry
from hpc3.contracts.workspace import workspace_cluster
from hpc3.core import _test_hooks as core_hooks
from hpc3.core import ledger
from hpc3.core.execution_evidence import allocation_document, measure, parse_checksums


def main(argv: Sequence[str] | None = None) -> int:
    """Locate exactly one durable attempt and publish verified native evidence.

    Args:
        argv: Explicit arguments or process arguments.

    Returns:
        Zero with a structured ledger/accounting report, including pending states.

    Raises:
        ValueError: Ambiguous/missing attempt, invalid accounting or altered results.
        AppError: Remote access or report publication failure.
    """
    flags = cli_args.parse_single_flags(
        list(argv) if argv is not None else list(sys.argv[1:]),
        ("--config", "--attempt", "--checksums", "--report-task"),
    )
    task = cli_args.require_flag(flags, "--report-task")
    report_identity(task, "hpc")
    attempt = cli_args.require_flag(flags, "--attempt")
    workspace = _config.load_workspace(flags)
    cluster = workspace_cluster(workspace)
    entries = ledger.read(pathlib.Path(workspace["ledger"]), cluster)
    matches = [row for row in entries if row["experiment"].get("research_attempt") == attempt]
    if len(matches) != 1:
        raise ValueError("HPC_RESEARCH_RECOVERY: require exactly one recorded attempt")
    entry = ledger.require_entry(entries, matches[0]["job_id"], workspace["host"])
    accounting = allocation_document(entry, cluster)
    report: JSONObject = {
        "kind": "HPC3_ATTEMPT_EVIDENCE",
        "ledger": encode_ledger_entry(entry),
        "accounting": accounting,
    }
    if require_str(require_dict(accounting, "job"), "state") == "COMPLETED":
        path = pathlib.Path(cli_args.require_flag(flags, "--checksums"))
        measured = measure(
            entry, cluster, parse_checksums(core_hooks.read_bytes(path).decode("utf-8"))
        )
        report["measurement"] = measured.report
        report["record"] = encode_run_record(measured.record)
    body = "HPC3 ATTEMPT LOAD-REPORT\n" + dump_json_str(report)
    post_load_report(
        post=_test_hooks.http_post,
        read_text=_config.stack_file_text,
        task_id=task,
        body=body,
        family="hpc",
    )
    _test_hooks.emit(dump_json_str(report))
    return 0


def entrypoint() -> None:
    """Apply the common CLI error boundary and preserve the process exit status."""
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
