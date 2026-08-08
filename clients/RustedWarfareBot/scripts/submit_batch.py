"""Queue a batch of matches on the match service.

The submission door of [[harness-match-service]], phase zero: the same job
file and the same knobs ``scripts.sweep`` takes, but the matches land in the
queue for whatever workers are polling instead of holding this shell for the
night. Resubmitting a batch queues only what earlier submissions missed,
which is the resume semantics sweeps already have on disk.

Usage::

    poetry run python -m scripts.submit_batch <dsn> <job-file> <name> \
        [lockstep] [map difficulty [pin-delta-ms [fast-forward]]]
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rw_bot.harness import _test_hooks
from rw_bot.harness.match import decode_match_config, describe
from rw_bot.harness.runner import TREE_DIR, SweepConfig, decode_sweep_config
from rw_bot.harness.sweep import parse_jobs
from rw_bot.service import _test_hooks as service_hooks
from rw_bot.service.queue import bootstrap, submit
from scripts.sweep import (
    CLONE_PREFIX,
    DEFAULT_LOCKSTEP,
    DUEL_OPPONENTS,
    SOURCE_GAME_DIR,
    SWEEP_ROOT,
)

EXIT_OK = 0
EXIT_BAD_USAGE = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Read a job file and queue every match in it.

    Args:
        argv: ``<dsn> <job-file> <name> [lockstep] [map difficulty
            [pin-delta-ms [fast-forward]]]``. ``None`` reads the process
            arguments.

    Returns:
        ``EXIT_OK`` after queueing, ``EXIT_BAD_USAGE`` on a bad argument
        count.

    Raises:
        SweepError: When the job file is malformed.
        DecodeError: When an argument is out of range.
        OSError: When the job file cannot be read.
        Exception: Whatever the database driver raises when the queue is
            unreachable.
    """
    args = list(argv) if argv is not None else _test_hooks.read_argv()
    if len(args) not in (3, 4, 6, 7, 8):
        _test_hooks.write_line(
            "usage: submit_batch <dsn> <job-file> <name> [lockstep] "
            "[map difficulty [pin-delta-ms [fast-forward]]]"
        )
        return EXIT_BAD_USAGE
    match = (
        decode_match_config(
            {"map_path": args[4], "opponents": DUEL_OPPONENTS, "difficulty": int(args[5])}
        )
        if len(args) >= 6
        else None
    )
    jobs = parse_jobs(_test_hooks.read_text_lines(Path(args[1])))
    out_dir = Path(SWEEP_ROOT) / args[2]
    # Workers is 1 in the stored configuration: how many workers play a
    # queued batch is decided by who polls, not by the submission.
    config: SweepConfig = decode_sweep_config(
        {
            "out_dir": str(out_dir),
            "workers": 1,
            "lockstep": int(args[3]) if len(args) >= 4 else DEFAULT_LOCKSTEP,
            "clone_prefix": CLONE_PREFIX,
            "source_game_dir": SOURCE_GAME_DIR,
            "tree": str(out_dir / TREE_DIR),
            "pin_delta": int(args[6]) if len(args) >= 7 else 0,
            "fast_forward": int(args[7]) if len(args) == 8 else 0,
        },
        match,
    )
    if match is not None:
        _test_hooks.write_line(f"[submit] {describe(match)}")
    conn = service_hooks.connect(args[0])
    bootstrap(conn)
    queued = submit(conn, args[2], config, jobs)
    conn.close()
    _test_hooks.write_line(f"[submit] {queued} of {len(jobs)} matches newly queued for {args[2]}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
