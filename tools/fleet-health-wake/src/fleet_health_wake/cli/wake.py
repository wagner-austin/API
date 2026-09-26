"""CLI: run one bridge cycle and exit.

Usage:
    fleet-health-wake --journal /path/to/fleet-mcp/state/health-events.jsonl

One cycle, then exit. The interval belongs to the pump that calls this
(``tools/hpc-wake/scripts/run_cycle.py``'s PUBLISHERS table), where it is
visible, for the same reason no sibling bridge carries a loop.

THE JOURNAL IS NAMED, NOT DISCOVERED: it lives in another repository, and a
bridge that guessed at its layout would break the day the layout moved
while reading as a quiet fleet. The position file derives from the
journal's path (:func:`platform_core.journal_cursor.cursor_path`), so the
two cannot disagree.

Environment (all required, exported once where the pump runs):
    TASKBOARD_MCP_API_KEY       taskboard-mcp's own x-api-key
    CORVIS_TENANT_ID            the tenants row whose board is posted to
    FLEET_HEALTH_WAKE_TASK_ID   the standing task posts land in
    BOARD_WATCH_URL             optional; defaults to the declared taskboard-mcp
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args

from fleet_health_wake.cycle import run_cycle

JOURNAL_FLAG = "--journal"
ALLOWED_FLAGS = (JOURNAL_FLAG,)


def main(argv: Sequence[str]) -> int:
    """Run one cycle against the journal named on the command line.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        0 always. Every failure raises instead, so the pump records a
        non-zero exit rather than a status line nobody reads.

    Raises:
        AppError: Any configuration or board failure, from
            :func:`fleet_health_wake.cycle.run_cycle`.
        JSONTypeError: A journal line or position file that does not
            decode.
        ValueError: A missing ``--journal`` flag, or a position past the
            journal's end.
        OSError: A file that cannot be read or written.
    """
    parsed = cli_args.parse_single_flags(argv, ALLOWED_FLAGS)
    run_cycle(pathlib.Path(cli_args.require_flag(parsed, JOURNAL_FLAG)))
    return 0


def entrypoint() -> None:
    """Console-script wrapper.

    Raises:
        SystemExit: Always, carrying :func:`main`'s status.
    """
    raise SystemExit(main(sys.argv[1:]))


# Without this, ``python -m fleet_health_wake.cli.wake`` imports the module,
# runs nothing and exits 0 -- a form that looks like a cycle with nothing to say.
if __name__ == "__main__":
    entrypoint()
