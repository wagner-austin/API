"""CLI: run one bridge cycle and exit.

Usage:
    hpc-wake --config /path/to/hpc3.json

Reads the same workspace document every hpc3 command reads, so the bridge
cannot disagree with ``hpc3-submit`` about where the ledger is. One cycle,
then exit; the interval belongs to the scheduler that calls this, where it
is visible, for the same reason ``board-watch`` has no ``--follow``.

Environment (all required, exported once where the scheduler runs):
    TASKBOARD_MCP_API_KEY   taskboard-mcp's own x-api-key
    CORVIS_TENANT_ID        the tenants row whose board is posted to
    HPC_WAKE_TASK_ID        the standing task announcements land in
    BOARD_WATCH_URL         optional; defaults to loopback :8033
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from hpc3.clusters import require_cluster
from hpc3.contracts.workspace import decode_workspace_connection
from platform_core import cli_args
from platform_core.json_utils import load_json_str

from hpc_wake.cycle import run_cycle

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
        AppError: Any configuration, board, or cluster failure, from
            :func:`hpc_wake.cycle.run_cycle`.
        JSONTypeError: A workspace or ledger row that does not decode.
        ValueError: A missing ``--config`` flag.
    """
    parsed = cli_args.parse_single_flags(argv, ALLOWED_FLAGS)
    config_path = pathlib.Path(cli_args.require_flag(parsed, CONFIG_FLAG))
    value = load_json_str(config_path.read_text(encoding="utf-8"))
    connection = decode_workspace_connection(value, config_dir=config_path.parent)
    run_cycle(connection, require_cluster(connection["cluster"]))
    return 0


def entrypoint() -> None:
    """Console-script wrapper.

    Raises:
        SystemExit: Always, carrying :func:`main`'s status.
    """
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    entrypoint()
