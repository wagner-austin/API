"""CLI: run one bridge cycle and exit.

Usage:
    lock-wake --journal /path/to/.fleet-events.jsonl --check-journal /path/to/.check-events.jsonl
              --action-journal /path/to/.action-events.jsonl
    lock-wake --remote-journal host:/abs/path/.fleet-events.jsonl --cursor-dir /local/dir
    lock-wake --remote-action-journal host:/abs/path/.action-events.jsonl --cursor-dir /local/dir

One cycle, then exit. The interval belongs to the pump that calls this
(``tools/hpc-wake/scripts/run_cycle.py``'s PUBLISHERS table), where it is
visible, for the same reason no sibling bridge carries a loop.

THE JOURNAL IS NAMED, NOT DISCOVERED: it lives in another repository's
root, and a bridge that guessed at repo layout would break the day the
layout moved while reading as a quiet fleet. The position file derives
from the journal's path (:func:`platform_core.journal_cursor.cursor_path`), so
the two cannot disagree. All three hub journals are required: the check
journal is the MCPs check lock's record of finished ``make test`` runs
(MCPs board task ea2ea29c), and the action journal is every MCPs command
that changes state and takes no lock (MCPs board task b00ab472), each
beside the fleet journal in the same clone root. A remote row names one
journal, and which kind it is, by its flag.

Environment (all required, exported once where the pump runs):
    TASKBOARD_MCP_API_KEY   taskboard-mcp's own x-api-key
    CORVIS_TENANT_ID        the tenants row whose board is posted to
    LOCK_WAKE_TASK_ID       the standing task announcements land in
    BOARD_WATCH_URL         optional; defaults to the declared taskboard-mcp
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args

from lock_wake.cycle import run_cycle, run_remote_cycle
from lock_wake.remote import parse_remote_journal

JOURNAL_FLAG = "--journal"

CHECK_JOURNAL_FLAG = "--check-journal"

ACTION_JOURNAL_FLAG = "--action-journal"

REMOTE_JOURNAL_FLAG = "--remote-journal"

REMOTE_ACTION_JOURNAL_FLAG = "--remote-action-journal"

CURSOR_DIR_FLAG = "--cursor-dir"

LOCAL_FLAGS = (JOURNAL_FLAG, CHECK_JOURNAL_FLAG, ACTION_JOURNAL_FLAG)

REMOTE_FLAGS = (REMOTE_JOURNAL_FLAG, REMOTE_ACTION_JOURNAL_FLAG)

ALLOWED_FLAGS = (*LOCAL_FLAGS, *REMOTE_FLAGS, CURSOR_DIR_FLAG)


def main(argv: Sequence[str]) -> int:
    """Run one cycle against the journals named on the command line.

    Two forms, one per pump row. ``--journal``, ``--check-journal`` and
    ``--action-journal`` read the hub's three local journals.
    ``--remote-journal <host>:<path>`` or ``--remote-action-journal
    <host>:<path>`` with ``--cursor-dir <dir>`` reads one journal on another
    host over ssh and keeps its position in ``dir`` (MCPs board task
    03590bf9). A flag of one form beside the other, or both remote flags,
    is refused, never ignored.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        0 always. Every failure raises instead, so the pump records a
        non-zero exit rather than a status line nobody reads.

    Raises:
        AppError: Any configuration, board or ssh failure, from
            :func:`lock_wake.cycle.run_cycle` or
            :func:`lock_wake.cycle.run_remote_cycle`.
        JSONTypeError: A journal line or position file that does not
            decode.
        ValueError: A missing flag, the two forms mixed, both remote flags,
            a remote journal that is not ``host:/path``, or a position past a
            journal's end.
        OSError: A file that cannot be read or written.
    """
    parsed = cli_args.parse_single_flags(argv, ALLOWED_FLAGS)
    remote = [flag for flag in REMOTE_FLAGS if flag in parsed]
    if len(remote) > 1:
        raise ValueError(
            f"{REMOTE_JOURNAL_FLAG} and {REMOTE_ACTION_JOURNAL_FLAG} are one row each; "
            "a remote row reads one journal"
        )
    if remote:
        mixed = [flag for flag in LOCAL_FLAGS if flag in parsed]
        if mixed:
            raise ValueError(
                f"{remote[0]} reads one remote journal and takes no "
                f"{' or '.join(mixed)}; the hub's local journals are a row of their own"
            )
        run_remote_cycle(
            parse_remote_journal(parsed[remote[0]]),
            pathlib.Path(cli_args.require_flag(parsed, CURSOR_DIR_FLAG)),
            actions=remote[0] == REMOTE_ACTION_JOURNAL_FLAG,
        )
        return 0
    if CURSOR_DIR_FLAG in parsed:
        raise ValueError(
            f"{CURSOR_DIR_FLAG} belongs to {REMOTE_JOURNAL_FLAG} or "
            f"{REMOTE_ACTION_JOURNAL_FLAG}; a local journal's position lives beside it"
        )
    run_cycle(
        pathlib.Path(cli_args.require_flag(parsed, JOURNAL_FLAG)),
        pathlib.Path(cli_args.require_flag(parsed, CHECK_JOURNAL_FLAG)),
        pathlib.Path(cli_args.require_flag(parsed, ACTION_JOURNAL_FLAG)),
    )
    return 0


def entrypoint() -> None:
    """Console-script wrapper.

    Raises:
        SystemExit: Always, carrying :func:`main`'s status.
    """
    raise SystemExit(main(sys.argv[1:]))


# Without this, ``python -m lock_wake.cli.wake`` imports the module, runs
# nothing and exits 0 -- a form that looks like a cycle with nothing to say.
if __name__ == "__main__":
    entrypoint()
