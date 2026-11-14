"""Publish that a recorded job array's artifacts are, byte for byte, the files kept locally.

Run as ``python -m hpc3.cli.artifact_report --config <workspace> --array <id>
--local-dir <dir> --report-task <task>``. Every member of the array must have
completed with exit 0 in Slurm's own accounting, written its artifact inside
its allocation, and left an artifact whose cluster-side sha256 equals the
same-named file in ``--local-dir``; only then is the measurement posted to
the task as a load-report, the record a board audit can read for a claim
that committed results came from those jobs.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import dump_json_str
from platform_core.load_report import post_load_report, report_identity

from hpc3.cli import _config, _test_hooks
from hpc3.clusters import require_cluster
from hpc3.core import ledger
from hpc3.core.artifact_evidence import artifact_report


def main(argv: Sequence[str] | None = None) -> int:
    """Measure one array against local copies of its artifacts and post the result.

    Args:
        argv: Explicit arguments or process arguments.

    Returns:
        Zero after every member is verified and the report is published.

    Raises:
        ValueError: Missing flags, an unknown array, or a member whose
            accounting or artifact timing refuses.
        AppError: A remote read failed, a local copy differs from its
            artifact, or the board refused the report.
        OSError: A local copy cannot be read.
    """
    flags = cli_args.parse_single_flags(
        list(sys.argv[1:] if argv is None else argv),
        ("--config", "--array", "--local-dir", "--report-task"),
    )
    task = cli_args.require_flag(flags, "--report-task")
    report_identity(task, "hpc")
    connection = _config.load_workspace_connection(flags)
    cluster = require_cluster(connection["cluster"])
    report = artifact_report(
        ledger.read(pathlib.Path(connection["ledger"]), cluster),
        cli_args.require_flag(flags, "--array"),
        cluster,
        pathlib.Path(cli_args.require_flag(flags, "--local-dir")),
    )
    payload = dump_json_str(report)
    post_load_report(
        post=_test_hooks.http_post,
        read_text=_config.stack_file_text,
        task_id=task,
        body="HPC3 ARTIFACT LOAD-REPORT\n" + payload,
        family="hpc",
    )
    _test_hooks.emit(payload)
    return 0


def entrypoint() -> None:
    """Run the shared CLI boundary, preserving refusal and publication failures."""
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
