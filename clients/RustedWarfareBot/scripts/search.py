"""The doctrine search driver: propose, panel, halve, repeat, report.

The executable half of :mod:`rw_bot.harness.search`. Each round writes
the surviving candidates' doctrine files, submits one interleaved batch
through the queue (every candidate arm paired against one shared control
on the same fresh seeds), waits for the fleet to play it, scores every
arm by paired margin delta, and keeps the top half for a bigger round.

The search proposes; the bar disposes: the final report ranks survivors
for graduation to an ordinary full panel judged on wins against the +4
bar and then fresh-tree replication (laws six and nine). Nothing here
adopts anything.

Run as ``python -m scripts.search <dsn> <name> [rng-seed]``. The knob
space and round schedule are code, not flags, so a search's definition
is versioned beside its laws.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from rw_bot.harness.margin import batch_margins
from rw_bot.harness.search import (
    Candidate,
    apply_moves,
    candidate_label,
    keep_top,
    paired_delta,
    sampled_pairs,
    single_moves,
)
from rw_bot.harness.sweep import parse_jobs
from rw_bot.policy.doctrine_file import format_doctrine, parse_doctrine_lines
from rw_bot.service import _test_hooks
from rw_bot.service.queue import batch_status, bootstrap, submit
from rw_bot.service.submit import batch_config

SWEEP_ROOT = Path("runs/sweeps")

#: The champion the search perturbs.
BASE_DOCTRINE = Path("doctrines/flame-nocover.doctrine")

#: Where variant doctrine files land, frozen into each batch's tree.
VARIANT_DIR = Path("doctrines/search")

#: Alternative knob values around the champion (flame 2, close 3, raid 3,
#: tech 1 per the ledger's champion stack) -- the champion's own values
#: stay out so every candidate is a real move. Values must satisfy the
#: doctrine codec's ranges; a bad one stops the search at round zero.
SPACE: Mapping[str, tuple[int, ...]] = {
    "flame": (0, 4),
    "close": (0, 6),
    "raid": (0, 6),
    "tech": (0, 2),
    "medics": (1,),
    "decoys": (2,),
}

#: Two-knob candidates drawn per search, law two's sample of the cross
#: product the arm ladder could never afford.
PAIR_CANDIDATES = 6

#: Pairs per candidate per round; each round keeps the top half.
SCHEDULE: tuple[int, ...] = (8, 16)

#: The panels' own match settings, unchanged so search rounds are
#: comparable with panel history.
MAP_PATH = "maps/skirmish/[p2]duel_lake.tmx"
DIFFICULTY = 2
LOCKSTEP = 75
PIN_DELTA = 3
FAST_FORWARD = 10
SAMPLES = 10000

#: Seconds between polls of a running round.
POLL_SECONDS = 120.0

EXIT_OK = 0
EXIT_BAD_USAGE = 2


def round_seeds(rng_seed: int, round_index: int, pairs: int) -> tuple[int, ...]:
    """Fresh, deterministic seeds for one round.

    Args:
        rng_seed: The search's reproducibility anchor.
        round_index: Which round the seeds are for.
        pairs: How many seeds the round plays.

    Returns:
        Distinct odd seeds no other round of this search uses.
    """
    base = 200_000 + rng_seed * 10_000 + round_index * 1_000
    return tuple(base + 2 * k + 1 for k in range(pairs))


def round_job_lines(
    survivors: Sequence[Candidate], seeds: Sequence[int], variant_dir: Path
) -> tuple[str, ...]:
    """The round's job lines, pairs interleaved so verdicts pair early.

    Args:
        survivors: The candidates still alive.
        seeds: The round's seeds.
        variant_dir: Where the candidates' doctrine files live.

    Returns:
        One control line plus one line per candidate, per seed.
    """
    lines: list[str] = []
    for seed in seeds:
        lines.append(f"control|{seed}|{BASE_DOCTRINE.as_posix()}|{SAMPLES}")
        for moves in survivors:
            label = candidate_label(moves)
            doctrine = (variant_dir / f"{label}.doctrine").as_posix()
            lines.append(f"{label}|{seed}|{doctrine}|{SAMPLES}")
    return tuple(lines)


def write_variants(survivors: Sequence[Candidate], variant_dir: Path) -> None:
    """Write every surviving candidate's doctrine file.

    Args:
        survivors: The candidates still alive.
        variant_dir: The directory the files land in, created if absent.

    Raises:
        OSError: When a file cannot be written.
        SearchError: Through ``apply_moves``, on a knob outside the
            doctrine.
    """
    base = parse_doctrine_lines(BASE_DOCTRINE.read_text(encoding="utf-8").splitlines())
    variant_dir.mkdir(parents=True, exist_ok=True)
    for moves in survivors:
        variant = apply_moves(base, moves)
        path = variant_dir / f"{candidate_label(moves)}.doctrine"
        path.write_text("".join(f"{line}\n" for line in format_doctrine(variant)), encoding="utf-8")


def wait_for_batch(dsn: str, batch: str) -> None:
    """Block until a batch has no queued or running matches.

    One connection per poll, never one held across the wait -- the
    worker's own lifecycle law.

    Args:
        dsn: The queue database.
        batch: The batch to wait on.
    """
    while True:
        conn = _test_hooks.connect(dsn)
        status = batch_status(conn, batch)
        conn.close()
        if status["queued"] == 0 and status["running"] == 0:
            return
        _test_hooks.sleep(POLL_SECONDS)


def run_search(
    dsn: str,
    name: str,
    rng_seed: int,
    sweeps_root: Path = SWEEP_ROOT,
    variant_dir: Path = VARIANT_DIR,
) -> tuple[str, ...]:
    """Run every round and return the ranked report.

    Args:
        dsn: The queue database.
        name: The search's name; round batches file as ``<name>-r<i>``.
        rng_seed: Reproducibility anchor for pair sampling and seeds.
        sweeps_root: Where batch artifacts land, injectable for tests.
        variant_dir: Where variant doctrines land, injectable for tests.

    Returns:
        Report lines: each round's scores, then the survivors ranked for
        graduation.

    Raises:
        MatchServiceError: Through the queue, on unreadable rows.
        SweepError: When a job line cannot be parsed back.
    """
    survivors: tuple[Candidate, ...] = (
        *single_moves(SPACE),
        *sampled_pairs(SPACE, PAIR_CANDIDATES, rng_seed),
    )
    lines: list[str] = [f"# search {name} (rng {rng_seed}): {len(survivors)} candidates"]
    for round_index, pairs in enumerate(SCHEDULE):
        batch = f"{name}-r{round_index}"
        write_variants(survivors, variant_dir)
        seeds = round_seeds(rng_seed, round_index, pairs)
        jobs = parse_jobs(round_job_lines(survivors, seeds, variant_dir))
        config = batch_config(batch, LOCKSTEP, MAP_PATH, DIFFICULTY, PIN_DELTA, FAST_FORWARD)
        conn = _test_hooks.connect(dsn)
        bootstrap(conn)
        queued = submit(conn, batch, config, jobs)
        conn.close()
        lines.append(
            f"# round {round_index}: {len(survivors)} arms, {pairs} pairs, {queued} queued"
        )
        wait_for_batch(dsn, batch)
        margins = batch_margins(sweeps_root / batch)
        scores: dict[Candidate, float] = {}
        for moves in survivors:
            n, mean, sd = paired_delta(margins, candidate_label(moves), "control")
            scores[moves] = mean
            lines.append(
                f"{batch} {candidate_label(moves):24} n={n:3}"
                f"  margin delta {mean:+.3f} (sd {sd:.3f})"
            )
        survivors = keep_top(scores, max(1, len(survivors) // 2))
    lines.append("# graduation order (full win-bar panel next, laws six and nine):")
    for moves in survivors:
        lines.append(f"#   {candidate_label(moves)}")
    return tuple(lines)


def main(
    argv: Sequence[str] | None = None,
    sweeps_root: Path = SWEEP_ROOT,
    variant_dir: Path = VARIANT_DIR,
) -> int:
    """Run one search from the command line.

    Args:
        argv: ``<dsn> <name> [rng-seed]``. ``None`` reads
            ``sys.argv[1:]``.
        sweeps_root: Where batch artifacts land, injectable for tests.
        variant_dir: Where variant doctrines land, injectable for tests.

    Returns:
        ``EXIT_OK``, or ``EXIT_BAD_USAGE`` on a bad argument count.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (2, 3):
        sys.stdout.write("usage: search <dsn> <name> [rng-seed]\n")
        return EXIT_BAD_USAGE
    rng_seed = int(args[2]) if len(args) == 3 else 0
    for line in run_search(args[0], args[1], rng_seed, sweeps_root, variant_dir):
        sys.stdout.write(line + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
