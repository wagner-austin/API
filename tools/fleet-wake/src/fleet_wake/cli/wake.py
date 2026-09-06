"""CLI: run one bridge cycle and exit.

Usage:
    fleet-wake --config /path/to/fleet.json

Reads the same workspace document every ``fleet-*`` command reads, so the
bridge cannot disagree with ``fleet-run`` about where the ledger is. One
cycle, then exit; the interval belongs to the scheduler that calls this, where
it is visible, for the same reason ``fleet-watch`` has no ``--follow``.

Environment (all required, exported once where the scheduler runs):
    TASKBOARD_MCP_API_KEY   taskboard-mcp's own x-api-key
    CORVIS_TENANT_ID        the tenants row whose board is posted to
    FLEET_WAKE_TASK_ID      the standing task announcements land in
    BOARD_WATCH_URL         optional; defaults to loopback :8033
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args

from fleet_wake.cycle import load_workspace, run_cycle

CONFIG_FLAG = "--config"

ALLOWED_FLAGS = (CONFIG_FLAG,)


def main(argv: Sequence[str]) -> int:
    """Run one cycle against the workspace named on the command line.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        0 always. Every failure raises instead, so the scheduler records a
        non-zero exit rather than a status line nobody reads.

    Raises:
        AppError: Any configuration or board failure, from
            :func:`fleet_wake.cycle.run_cycle`.
        JSONTypeError: A workspace, ledger row, or position line that does
            not decode.
        ValueError: A missing ``--config`` flag.
        OSError: A file that cannot be read or written.
    """
    parsed = cli_args.parse_single_flags(argv, ALLOWED_FLAGS)
    config_path = pathlib.Path(cli_args.require_flag(parsed, CONFIG_FLAG))
    run_cycle(load_workspace(config_path))
    return 0


def entrypoint() -> None:
    """Console-script wrapper.

    Raises:
        SystemExit: Always, carrying :func:`main`'s status.
    """
    raise SystemExit(main(sys.argv[1:]))


# Without this, ``python -m fleet_wake.cli.wake`` imports the module, runs
# nothing and exits 0 -- a form that looks like a cycle with nothing to say.
if __name__ == "__main__":
    entrypoint()
