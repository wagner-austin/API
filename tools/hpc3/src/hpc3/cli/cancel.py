"""CLI: stop jobs, and say which ones were actually still running.

Usage:
    hpc3-cancel --config hpc3.json --job 55519937
    hpc3-cancel --config hpc3.json --job 55519937,55520509

``scancel`` is silent about a job that had already finished, so this reports
each outcome explicitly. "Cancelled 3 jobs" when two of them ended an hour
ago is the kind of report that gets believed.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core import cli_args

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.contracts.workspace import workspace_cluster
from hpc3.core.cancel import cancel, summarise

_FLAGS = (_config.CONFIG_FLAG, "--job")


def main(argv: Sequence[str] | None = None) -> int:
    """Cancel one or more jobs.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when accounting could be read for at least one job.

    Raises:
        ValueError: If a required flag is missing, an argument is unknown, or
            no job id was named -- a bare ``scancel`` would take every job the
            user has.
        AppError: If a remote command fails or accounting output is
            malformed.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)
    host = workspace["host"]
    requested = [part for part in cli_args.require_flag(parsed, "--job").split(",") if part != ""]
    if requested == []:
        raise ValueError("--job must name at least one job id")

    outcomes = cancel(host, requested, workspace_cluster(workspace))
    if outcomes == []:
        raise ValueError(f"sacct knows no job in {requested} on {host}")

    for outcome in outcomes:
        verb = "stopped" if outcome.was_running else "already finished as"
        _test_hooks.emit(f"{outcome.job_id} {verb} {outcome.state}")

    stopped, already_over = summarise(outcomes)
    _test_hooks.emit(f"stopped {stopped}, already finished {already_over}")
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = ["entrypoint", "main"]
