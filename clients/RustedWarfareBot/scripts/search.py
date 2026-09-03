"""The doctrine search driver: propose, panel, halve, repeat, report.

The executable half of :mod:`rw_bot.harness.search`. Each round writes
the surviving candidates' doctrine files, plays one interleaved batch
(every candidate arm paired against one shared control on the same fresh
seeds), scores every arm by paired margin delta, and keeps the top half
for a bigger round.

WHO PLAYS THE ROUND IS A SEAM. The driver hands each round's job lines to
a :class:`RoundRunner` and reads scorecards back from
``runs/sweeps/<batch>/`` -- everything between those two points is the
runner's. :class:`QueueRunner` plays through the workstation fleet's
match-service queue; :class:`~rw_bot.harness.cluster_round.ClusterRound`
plays through HPC3, where a round of a hundred matches costs nothing and
finishes in the wall-clock of its slowest member.

The search proposes; the bar disposes: the final report ranks survivors
for graduation to an ordinary full panel judged on wins against the +4
bar and then fresh-tree replication (laws six and nine). Nothing here
adopts anything.

Run as ``python -m scripts.search <where> <name> [rng-seed]`` with
``<where>`` either a queue DSN or ``hpc3:<workspace.json>``. The knob
space and round schedule are code, not flags, so a search's definition
is versioned beside its laws.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from rw_bot.harness import _test_hooks as host_hooks
from rw_bot.harness.cluster_round import ClusterRound
from rw_bot.harness.margin import batch_margins
from rw_bot.harness.search import (
    Candidate,
    apply_moves,
    candidate_label,
    effective_space,
    keep_top,
    paired_delta,
    sampled_pairs,
    single_moves,
)
from rw_bot.harness.sweep import parse_jobs
from rw_bot.policy.doctrine_file import format_doctrine, parse_doctrine_lines
from rw_bot.service import _test_hooks
from rw_bot.service._test_hooks import Connection
from rw_bot.service.queue import batch_status, bootstrap, submit
from rw_bot.service.submit import batch_config
from scripts.search_specs import SearchSpec, require_search_spec

SWEEP_ROOT = Path("runs/sweeps")

#: Where variant doctrine files land, frozen into each batch's tree.
VARIANT_DIR = Path("doctrines/search")


#: The panels' own match settings, invariant across regimes so search
#: rounds stay comparable with panel history on the same rung.
MAP_PATH = "maps/skirmish/[p2]duel_lake.tmx"
LOCKSTEP = 75
PIN_DELTA = 3
FAST_FORWARD = 10

#: Seconds between polls of a running round.
POLL_SECONDS = 120.0

#: The explicit route marker for cluster-played rounds. A prefix rather
#: than sniffing the argument's shape: a DSN and a workspace path can both
#: look like anything, and a guessed backend is a submission to the wrong
#: machine.
CLUSTER_PREFIX = "hpc3:"

#: The cluster identity a cluster-played search runs against, matching the
#: workspace the ``hpc3:`` argument names. Code, not flags, like the knob
#: space: a search's definition is versioned beside its laws.
CLUSTER_HOST = "hpc3"
CLUSTER_ROOT = "/pub/wagnera3/rusted"

#: Where cluster rounds keep their frozen trees and campaign documents --
#: run artifacts beside the sweeps, never repository content.
CLUSTER_SCRATCH = Path("runs/search-staging")

#: Where a cluster round's job file is written: inside the repository's
#: ``sweeps`` tree, because the payload freeze copies that directory and
#: the members must read the same file the driver wrote.
CLUSTER_JOBS_DIR = Path("sweeps/search")

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
    survivors: Sequence[Candidate],
    seeds: Sequence[int],
    variant_dir: Path,
    spec: SearchSpec,
) -> tuple[str, ...]:
    """The round's job lines, pairs interleaved so verdicts pair early.

    Args:
        survivors: The candidates still alive.
        seeds: The round's seeds.
        variant_dir: Where the candidates' doctrine files live.
        spec: The regime, for the control doctrine and sample budget.

    Returns:
        One control line plus one line per candidate, per seed.
    """
    control = Path(spec["base"]).as_posix()
    samples = spec["samples"]
    lines: list[str] = []
    for seed in seeds:
        lines.append(f"control|{seed}|{control}|{samples}")
        for moves in survivors:
            label = candidate_label(moves)
            doctrine = (variant_dir / f"{label}.doctrine").as_posix()
            lines.append(f"{label}|{seed}|{doctrine}|{samples}")
    return tuple(lines)


def write_variants(survivors: Sequence[Candidate], variant_dir: Path, base_path: Path) -> None:
    """Write every surviving candidate's doctrine file.

    Args:
        survivors: The candidates still alive.
        variant_dir: The directory the files land in, created if absent.
        base_path: The doctrine the candidates perturb.

    Raises:
        OSError: When a file cannot be written.
        SearchError: Through ``apply_moves``, on a knob outside the
            doctrine.
    """
    base = parse_doctrine_lines(base_path.read_text(encoding="utf-8").splitlines())
    variant_dir.mkdir(parents=True, exist_ok=True)
    for moves in survivors:
        variant = apply_moves(base, moves)
        path = variant_dir / f"{candidate_label(moves)}.doctrine"
        path.write_text("".join(f"{line}\n" for line in format_doctrine(variant)), encoding="utf-8")


class RoundRunner(Protocol):
    """Plays one round's jobs and files scorecards under the sweeps root.

    The driver's whole contract with whoever plays: hand over the batch
    name and the job lines, and when ``run`` returns, one scorecard per
    job sits in ``runs/sweeps/<batch>/`` for the margin scorer to read.
    """

    def run(self, batch: str, job_lines: Sequence[str]) -> None:
        """Play one round to completion.

        Args:
            batch: The round's batch name.
            job_lines: The round's job file content, comments included.
        """
        ...


def patient_connect(dsn: str) -> Connection:
    """Connect to the queue, outlasting a database outage.

    Docker crashed four times in three days and the fourth killed the
    first search mid-poll on a connection timeout (log 2026-08-11). A
    driver that runs for hours must survive the outages its queue rows
    already do: name the outage, wait, try again, forever.

    Args:
        dsn: The queue database.

    Returns:
        An open connection.
    """
    while True:
        try:
            return _test_hooks.connect(dsn)
        except Exception as error:
            # The database sits behind an untyped seam on purpose (the
            # service's own Protocol discipline), so the driver names the
            # library by module instead of importing its classes: only a
            # psycopg failure is an outage; anything else is a bug and
            # propagates.
            if type(error).__module__.split(".")[0] != "psycopg":
                raise
            host_hooks.write_line(
                f"# database unreachable ({error}); retrying in {POLL_SECONDS:.0f}s"
            )
            _test_hooks.sleep(POLL_SECONDS)


def wait_for_batch(dsn: str, batch: str) -> None:
    """Block until a batch has no queued or running matches.

    One connection per poll, never one held across the wait -- the
    worker's own lifecycle law. A round that sits queued with nothing
    claiming it is named loudly once: vhsearch1's first round waited on
    an empty fleet in silence because the workers drain-and-exit when
    the queue empties before a submission (log 2026-08-11).

    Args:
        dsn: The queue database.
        batch: The batch to wait on.
    """
    stalled = 0
    warned = False
    while True:
        conn = patient_connect(dsn)
        status = batch_status(conn, batch)
        conn.close()
        if status["queued"] == 0 and status["running"] == 0:
            return
        if status["running"] == 0:
            stalled += 1
            if stalled >= 3 and not warned:
                sys.stdout.write(
                    f"# WARNING {batch}: queued matches but nothing claims them;"
                    " is the fleet up? (make fleet-up)\n"
                )
                warned = True
        else:
            stalled = 0
        _test_hooks.sleep(POLL_SECONDS)


class QueueRunner:
    """Plays a round through the workstation fleet's match-service queue.

    The original round transport, lifted whole when the cluster runner
    arrived: submit to the queue, then wait for the fleet to drain it,
    outlasting database outages and naming an unclaimed round loudly.

    Attributes:
        dsn: The queue database.
    """

    def __init__(self, dsn: str, difficulty: int) -> None:
        """Bind the runner to its queue and regime.

        Args:
            dsn: The queue database.
            difficulty: The AI difficulty every round plays at.
        """
        self.dsn = dsn
        self.difficulty = difficulty

    def run(self, batch: str, job_lines: Sequence[str]) -> None:
        """Submit the round and wait for the fleet to play it.

        Args:
            batch: The round's batch name.
            job_lines: The round's job file content.

        Raises:
            MatchServiceError: Through the queue, on unreadable rows.
            SweepError: When a job line cannot be parsed back.
        """
        jobs = parse_jobs(job_lines)
        config = batch_config(batch, LOCKSTEP, MAP_PATH, self.difficulty, PIN_DELTA, FAST_FORWARD)
        conn = patient_connect(self.dsn)
        bootstrap(conn)
        submit(conn, batch, config, jobs)
        conn.close()
        wait_for_batch(self.dsn, batch)


def run_search(
    runner: RoundRunner,
    spec: SearchSpec,
    name: str,
    rng_seed: int,
    sweeps_root: Path = SWEEP_ROOT,
    variant_dir: Path = VARIANT_DIR,
) -> tuple[str, ...]:
    """Run every round and return the ranked report.

    Args:
        runner: Who plays each round -- the queue or the cluster.
        spec: The registered regime: base doctrine, knob space, schedule.
        name: The search's name; round batches file as ``<name>-r<i>``.
        rng_seed: Reproducibility anchor for pair sampling and seeds.
        sweeps_root: Where batch artifacts land, injectable for tests.
        variant_dir: Where variant doctrines land, injectable for tests.

    Returns:
        Report lines: each round's scores, then the survivors ranked for
        graduation. Every line is also written the moment it happens --
        a driver that runs for fifteen hours and prints only at the end
        is unreadable while it matters (log 2026-08-11).

    Raises:
        MatchServiceError: Through the queue, on unreadable rows.
        SweepError: When a job line cannot be parsed back.
    """
    lines: list[str] = []

    def note(text: str) -> None:
        host_hooks.write_line(text)
        lines.append(text)

    base_path = Path(spec["base"])
    base = parse_doctrine_lines(base_path.read_text(encoding="utf-8").splitlines())
    space = effective_space(spec["space"], base)
    survivors: tuple[Candidate, ...] = (
        *single_moves(space),
        *sampled_pairs(space, spec["pair_candidates"], rng_seed),
    )
    note(f"# search {name} (rng {rng_seed}): {len(survivors)} candidates")
    for round_index, pairs in enumerate(spec["schedule"]):
        batch = f"{name}-r{round_index}"
        write_variants(survivors, variant_dir, base_path)
        seeds = round_seeds(rng_seed, round_index, pairs)
        lines_out = round_job_lines(survivors, seeds, variant_dir, spec)
        note(f"# round {round_index}: {len(survivors)} arms, {pairs} pairs, {len(lines_out)} jobs")
        runner.run(batch, lines_out)
        margins = batch_margins(sweeps_root / batch)
        scores: dict[Candidate, float] = {}
        for moves in survivors:
            n, mean, sd = paired_delta(margins, candidate_label(moves), "control")
            scores[moves] = mean
            note(
                f"{batch} {candidate_label(moves):24} n={n:3}"
                f"  margin delta {mean:+.3f} (sd {sd:.3f})"
            )
        survivors = keep_top(scores, max(1, len(survivors) // 2))
    note("# graduation order (full win-bar panel next, laws six and nine):")
    for moves in survivors:
        note(f"#   {candidate_label(moves)}")
    return tuple(lines)


def main(
    argv: Sequence[str] | None = None,
    sweeps_root: Path = SWEEP_ROOT,
    variant_dir: Path = VARIANT_DIR,
) -> int:
    """Run one search from the command line.

    Args:
        argv: ``<where> <spec> <name> [rng-seed]``, with ``<where>`` a
            queue DSN or ``hpc3:<workspace.json>`` to play the rounds on
            the cluster, and ``<spec>`` a registered regime from
            :data:`SPECS`. The prefix routes explicitly; nothing is
            sniffed. ``None`` reads ``sys.argv[1:]``.
        sweeps_root: Where batch artifacts land, injectable for tests.
        variant_dir: Where variant doctrines land, injectable for tests.

    Returns:
        ``EXIT_OK``, or ``EXIT_BAD_USAGE`` on a bad argument count.

    Raises:
        SearchError: ``RW-SEARCH-002`` on an unregistered spec name.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4):
        sys.stdout.write("usage: search <dsn|hpc3:workspace.json> <spec> <name> [rng-seed]\n")
        return EXIT_BAD_USAGE
    spec = require_search_spec(args[1])
    rng_seed = int(args[3]) if len(args) == 4 else 0
    where = args[0]
    runner: RoundRunner
    if where.startswith(CLUSTER_PREFIX):
        runner = ClusterRound(
            config=where[len(CLUSTER_PREFIX) :],
            host=CLUSTER_HOST,
            cluster_root=CLUSTER_ROOT,
            map_path=MAP_PATH,
            difficulty=spec["difficulty"],
            fast_forward=FAST_FORWARD,
            scratch=CLUSTER_SCRATCH,
            sweeps_root=sweeps_root,
            jobs_dir=CLUSTER_JOBS_DIR,
            poll_seconds=POLL_SECONDS,
        )
    else:
        runner = QueueRunner(where, spec["difficulty"])
    # The report streams as it happens; the return value is for callers.
    run_search(runner, spec, args[2], rng_seed, sweeps_root, variant_dir)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
