"""Publish measured installed-controller integrity through the existing board reporter."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.load_report import post_load_report, report_identity

from hpc3.cli import _config, _test_hooks
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.controller_evidence import measure_controller
from hpc3.core.digest import sha256_hex


def main(argv: Sequence[str] | None = None) -> int:
    """Read frozen provenance, verify the installed runtime and publish its measurements.

    Args:
        argv: Explicit arguments or process arguments.

    Returns:
        Zero only after every check and board publication succeeds.

    Raises:
        ValueError: Missing arguments, invalid provenance or failed integrity checks.
        AppError: Board publication failure.
    """
    flags = cli_args.parse_single_flags(
        list(sys.argv[1:] if argv is None else argv),
        ("--provenance", "--wheels-dir", "--selfcheck", "--selfcheck-sha256", "--report-task"),
    )
    task = cli_args.require_flag(flags, "--report-task")
    report_identity(task, "hpc")
    provenance = core_hooks.read_bytes(Path(cli_args.require_flag(flags, "--provenance")))
    report = measure_controller(
        narrow_json_to_dict(load_json_str(provenance.decode("utf-8"))),
        Path(cli_args.require_flag(flags, "--wheels-dir")),
        Path(cli_args.require_flag(flags, "--selfcheck")),
        cli_args.require_flag(flags, "--selfcheck-sha256"),
    )
    report["provenance_sha256"] = sha256_hex(provenance)
    payload = dump_json_str(report)
    post_load_report(
        post=_test_hooks.http_post,
        read_text=_config.stack_file_text,
        task_id=task,
        body="HPC3 CONTROLLER LOAD-REPORT\n" + payload,
        family="hpc",
    )
    _test_hooks.emit(payload)
    return 0


def entrypoint() -> None:
    """Use the shared CLI failure boundary."""
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
