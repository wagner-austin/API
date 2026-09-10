"""CLI: run one bridge cycle and exit.

Usage:
    ci-wake --enrolment /path/to/pushes.jsonl

One cycle, then exit. The interval belongs to the scheduler that calls this,
where it is visible, for the same reason ``fleet-watch`` has no ``--follow``
and ``fleet-wake`` has no loop.

THE ENROLMENT RECORD IS NAMED, NOT DISCOVERED, and it is the same path the
``pre-push`` hooks write to. Both halves take it as an argument so the two
cannot disagree about where the record is -- the failure that would produce
is a bridge announcing nothing while every push is enrolled correctly, which
reads exactly like a quiet week.

Environment (all required, exported once where the scheduler runs):
    TASKBOARD_MCP_API_KEY   taskboard-mcp's own x-api-key
    CORVIS_TENANT_ID        the tenants row whose board is posted to
    CI_WAKE_TASK_ID         the standing task announcements land in
    BOARD_WATCH_URL         optional; defaults to loopback :8033

And on PATH: ``gh``, logged in. See :mod:`ci_wake._test_hooks` on why this
bridge borrows the machine's GitHub identity rather than minting one.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args

from ci_wake.cycle import run_cycle

ENROLMENT_FLAG = "--enrolment"

ALLOWED_FLAGS = (ENROLMENT_FLAG,)


def main(argv: Sequence[str]) -> int:
    """Run one cycle against the enrolment record named on the command line.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        0 always. Every failure raises instead, so the scheduler records a
        non-zero exit rather than a status line nobody reads.

    Raises:
        AppError: Any configuration, ``gh`` or board failure, from
            :func:`ci_wake.cycle.run_cycle`.
        JSONTypeError: An enrolment row, position line or GitHub payload
            that does not decode.
        ValueError: A missing ``--enrolment`` flag.
        OSError: A record that cannot be read or written.
    """
    parsed = cli_args.parse_single_flags(argv, ALLOWED_FLAGS)
    run_cycle(pathlib.Path(cli_args.require_flag(parsed, ENROLMENT_FLAG)))
    return 0


def entrypoint() -> None:
    """Console-script wrapper.

    Raises:
        SystemExit: Always, carrying :func:`main`'s status.
    """
    raise SystemExit(main(sys.argv[1:]))


# Without this, ``python -m ci_wake.cli.wake`` imports the module, runs
# nothing and exits 0 -- a form that looks like a cycle with nothing to say.
if __name__ == "__main__":
    entrypoint()
