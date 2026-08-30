"""Play a batch of matches in parallel and file one result per match.

The command-line adapter for experiments: it parses arguments, builds the batch
configuration, and reports. Deciding what a match is belongs to
:mod:`rw_bot.harness.sweep`, and playing one belongs to
:mod:`rw_bot.harness.runner`.

Run as ``python -m scripts.sweep <job-file> <name> [workers] [lockstep]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from rw_bot.harness import _test_hooks
from rw_bot.harness.clone import CLONE_PREFIX
from rw_bot.harness.match import decode_match_config, describe
from rw_bot.harness.records import batch_fingerprint, read_batch_rows, write_arm_records
from rw_bot.harness.results_layout import PINNED_GAME_DIR, SWEEP_ROOT, TRACE_ROOT
from rw_bot.harness.runner import (
    TREE_DIR,
    SweepConfig,
    decode_sweep_config,
    outstanding,
    prepare_tree,
    run_worker,
)
from rw_bot.harness.sweep import parse_jobs

#: Opponents asked for when a match is given. One, because the goal is to beat
#: one -- and because the engine caps the count by the map anyway, so a
#: two-player map would give one whatever was asked.
DUEL_OPPONENTS = 1

#: The pinned game directory worker copies are taken from. The workstation's
#: own; the cluster's staged copy is named per member instead, because a
#: compute node has no repository to be relative to.
SOURCE_GAME_DIR = PINNED_GAME_DIR

#: Engine frames between samples, defaulting to the observed free-running rate
#: so a locked match covers the same ground as the unlocked ones already
#: recorded ([[policy-determinism]]).
DEFAULT_LOCKSTEP = 75

DEFAULT_WORKERS = 4

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_BAD_USAGE = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Read a job file, play what is outstanding, and report.

    Args:
        argv: ``<job-file> <name> [workers] [lockstep] [map difficulty
            [pin-delta-ms [fast-forward]]]``. ``None`` reads the process
            arguments.

    Returns:
        ``EXIT_OK`` when every match in the file has a result, ``EXIT_INCOMPLETE``
        when any is still outstanding, ``EXIT_BAD_USAGE`` on a bad argument
        count.

    Raises:
        SweepError: When the job file is malformed.
        DecodeError: When an argument is out of range.
        CloneError: When a worker's copy of the game is unusable.
        OSError: When the job file cannot be read or a result cannot be written.
    """
    args = list(argv) if argv is not None else _test_hooks.read_argv()
    if len(args) not in (2, 3, 4, 6, 7, 8):
        _test_hooks.write_line(
            "usage: sweep <job-file> <name> [workers] [lockstep] "
            "[map difficulty [pin-delta-ms [fast-forward]]]"
        )
        return EXIT_BAD_USAGE

    # The map decides the opponent count, because the engine caps teams by the
    # map's own -- so a two-player map is a duel and the count needs no saying
    # ([[policy-determinism]]).
    match = (
        decode_match_config(
            {"map_path": args[4], "opponents": DUEL_OPPONENTS, "difficulty": int(args[5])}
        )
        if len(args) >= 6
        else None
    )

    jobs = parse_jobs(_test_hooks.read_text_lines(Path(args[0])))
    # Forward-slashed rather than through `Path`, whose str() is
    # backslashed on Windows. It goes into the batch's own config and out
    # again into every path composed from it, including one the launcher
    # hands to a child process.
    out_dir = f"{SWEEP_ROOT}/{args[1]}"
    _test_hooks.make_dirs(Path(out_dir))
    todo = outstanding(jobs, Path(out_dir))

    # Never more workers than there are matches, so a batch of two does not copy
    # the game four times to leave two of the copies idle.
    asked = int(args[2]) if len(args) >= 3 else DEFAULT_WORKERS
    config: SweepConfig = decode_sweep_config(
        {
            "out_dir": out_dir,
            # The repository-relative root, which is what a workstation batch
            # has always used. A cluster member is told an absolute one
            # instead, because its process does not start here.
            "traces": TRACE_ROOT,
            "workers": min(asked, len(todo)) if todo else asked,
            "lockstep": int(args[3]) if len(args) >= 4 else DEFAULT_LOCKSTEP,
            "clone_prefix": CLONE_PREFIX,
            "source_game_dir": SOURCE_GAME_DIR,
            # Under the results directory, so a batch and the code it ran are
            # one artifact -- resuming a batch resumes its code, whatever has
            # happened to the working tree since ([[policy-loop]]).
            "tree": f"{out_dir}/{TREE_DIR}",
            # Zero leaves the engine on the wall clock, which is what a tree
            # frozen before the option existed requires; a pinned batch says
            # so explicitly ([[policy-determinism]]).
            "pin_delta": int(args[6]) if len(args) >= 7 else 0,
            # The gym knob, certified bit-exact against realtime at 10
            # (log 2026-08-06): a fast batch is the realtime batch, sooner.
            "fast_forward": int(args[7]) if len(args) == 8 else 0,
        },
        match,
    )
    if match is not None:
        _test_hooks.write_line(f"[sweep] {describe(match)}")

    already = len(jobs) - len(todo)
    _test_hooks.write_line(
        f"[sweep] {len(jobs)} matches, {already} already played, "
        f"{len(todo)} to go over {config['workers']} workers"
    )

    played = 0
    if todo:
        # Frozen before the first worker starts, so no thread races the copy
        # and the whole batch imports one tree. Skipped when nothing is left
        # to play: a completed batch's re-run should not freeze anything.
        prepare_tree(config)
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            counts = pool.map(partial(run_worker, todo, config=config), range(config["workers"]))
            played = sum(counts)

    _test_hooks.write_line(f"[sweep] {already + played}/{len(jobs)} matches have results")

    # Recomputed over EVERY filed result, not just the ones this pass played:
    # a run that finishes the last four matches of a twelve-match batch must
    # leave a record covering all twelve. The scorecards are the store, so
    # deriving from them is what keeps the record from disagreeing with them.
    rows = read_batch_rows(Path(out_dir), Path(TRACE_ROOT), args[1])
    if rows:
        arms = write_arm_records(Path(out_dir), args[1], rows, batch_fingerprint(SOURCE_GAME_DIR))
        _test_hooks.write_line(f"[sweep] recorded {len(arms)} arm(s): {', '.join(arms)}")

    return EXIT_OK if already + played == len(jobs) else EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main(None))
