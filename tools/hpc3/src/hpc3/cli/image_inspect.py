"""Publish verified native image-build evidence through the existing board reporter."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import dump_json_str

from hpc3.cli import _config, _test_hooks
from hpc3.cli._load_report import post_load_report, report_identity
from hpc3.clusters import require_cluster
from hpc3.core import _test_hooks as core_hooks
from hpc3.core import ledger
from hpc3.core.execution_evidence import parse_checksums
from hpc3.core.image_evidence import measure_build


def main(argv: Sequence[str] | None = None) -> int:
    """Verify one ledger-recorded build and post only measured metadata.

    Args:
        argv: Explicit arguments or process arguments.

    Returns:
        Zero after every verification and report publication succeeds.

    Raises:
        ValueError: Missing flags, invalid accounting, build or checksum evidence.
        AppError: Remote read or board publication failure.
    """
    flags = cli_args.parse_single_flags(
        list(sys.argv[1:] if argv is None else argv),
        ("--config", "--job", "--checksums", "--report-task"),
    )
    task = cli_args.require_flag(flags, "--report-task")
    report_identity(task, "hpc")
    connection = _config.load_workspace_connection(flags)
    cluster = require_cluster(connection["cluster"])
    entry = ledger.require_entry(
        ledger.read(pathlib.Path(connection["ledger"]), cluster),
        cli_args.require_flag(flags, "--job"),
        connection["host"],
    )
    path = pathlib.Path(cli_args.require_flag(flags, "--checksums"))
    report = measure_build(
        entry, cluster, parse_checksums(core_hooks.read_bytes(path).decode("utf-8"))
    )
    payload = dump_json_str(report)
    post_load_report(task, "HPC3 IMAGE BUILD LOAD-REPORT\n" + payload, "hpc")
    _test_hooks.emit(payload)
    return 0


def entrypoint() -> None:
    """Run the shared CLI boundary, preserving refusal and publication failures."""
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
