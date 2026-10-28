"""Measure a recorded benchmark replay and publish its native cluster evidence."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import dump_json_str

from hpc3.cli import _config, _test_hooks
from hpc3.cli._load_report import post_load_report, report_identity
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.workspace import workspace_cluster
from hpc3.core import _test_hooks as core_hooks
from hpc3.core import ledger
from hpc3.core.execution_evidence import parse_checksums, replay_report


def _select(entries: Sequence[LedgerEntry], job: str, host: str) -> LedgerEntry:
    """Resolve exactly one submission on the configured host.

    Args:
        entries: Persisted submissions.
        job: Requested allocation id.
        host: Workspace SSH alias.

    Returns:
        Unique matching ledger entry.

    Raises:
        ValueError: HPC_EVIDENCE_LEDGER for absent, duplicate or foreign entries.
    """
    matches = [entry for entry in entries if entry["job_id"] == job]
    if len(matches) != 1 or matches[0]["host"] != host:
        raise ValueError("HPC_EVIDENCE_LEDGER: require one submission on the workspace host")
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Read real scheduler/artifact evidence and post it as the measuring program.

    Args:
        argv: Explicit arguments or process arguments.

    Returns:
        Zero for identical replay, two for measured disagreement.

    Raises:
        ValueError: Invalid flags, lineage, accounting or artifact evidence.
        AppError: Failed remote reads or rejected board publication.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    flags = cli_args.parse_single_flags(
        tokens, ("--config", "--job", "--repeat", "--checksums", "--report-task")
    )
    task = cli_args.require_flag(flags, "--report-task")
    report_identity(task, "hpc")
    job, repeat = cli_args.require_flag(flags, "--job"), cli_args.require_flag(flags, "--repeat")
    path = pathlib.Path(cli_args.require_flag(flags, "--checksums"))
    checksums = parse_checksums(core_hooks.read_bytes(path).decode("utf-8"))
    workspace = _config.load_workspace(flags)
    cluster = workspace_cluster(workspace)
    entries = ledger.read(pathlib.Path(workspace["ledger"]), cluster)
    report = replay_report(
        _select(entries, job, workspace["host"]),
        _select(entries, repeat, workspace["host"]),
        cluster,
        checksums,
    )
    body = "HPC3 REPLAY LOAD-REPORT\n" + dump_json_str(report)
    post_load_report(task, body, "hpc")
    _test_hooks.emit(body)
    return 0 if report["comparison"] == "identical" else 2


def entrypoint() -> None:
    """Run through the common error boundary and preserve its exit status."""
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
