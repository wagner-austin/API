"""One paired panel, played by the cluster and judged, with no hand between.

Usage:
    python -m scripts.panel <hpc3:workspace.json> <batch> \\
        <control-doctrine> <arm-label> <arm-doctrine> <pairs> <difficulty>

The win-bar panel is the only measurement adoption listens to (laws six
and nine), and until 2026-09-03 every one was driven by hand: pick seeds
inline, write the sweep file, freeze, submit, watch a monitor, fire
converge passes at casualties, pull, judge. Each step already had a
canonical tool; this module is nothing but their order. The seed picker
is :func:`rw_bot.harness.sweep.fresh_seeds` (lifted after four inline
copies), the transport is :class:`~rw_bot.harness.cluster_round.ClusterRound`
(freeze -> document -> converge-until-full -> pull, transport-hardened),
and the judgement is the same ``pairs`` + ``margin`` report the manual
flow printed. No second submission path, no second judging path.

Seed namespaces are disjoint by construction: panels allocate
thousand-aligned blocks BELOW :data:`SEARCH_SEED_FLOOR`, and the search's
``round_seeds`` builds all of its seeds at or above it. A panel that
would cross the floor is refused rather than risking a collision with a
search round's block.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot import RwBotError
from rw_bot.harness import _test_hooks as host_hooks
from rw_bot.harness.cluster_round import ClusterRound
from rw_bot.harness.margin import batch_margins, report
from rw_bot.harness.sweep import fresh_seeds, parse_jobs
from scripts.pairs import format_pairs, pair_up, read_grades
from scripts.search import (
    CLUSTER_HOST,
    CLUSTER_PREFIX,
    CLUSTER_ROOT,
    CLUSTER_SCRATCH,
    FAST_FORWARD,
    MAP_PATH,
    POLL_SECONDS,
    SWEEP_ROOT,
    RoundRunner,
)

EXIT_OK = 0
EXIT_BAD_USAGE = 2

#: Sample budget per match, the regime invariant every panel has used.
SAMPLES = 10000

#: Where committed sweep files live; every file here contributes its
#: seeds to the used set, and the panel's own job file lands here too so
#: the freeze carries it to the members.
JOBS_DIR = Path("sweeps")

#: Width of one panel's seed block. A 48-pair panel uses 48 of the
#: ~2,500 odd values in a block, so consecutive panels never crowd.
SEED_BLOCK = 5000

#: Where panel seed allocation starts when no sweep file exists yet.
FIRST_SEED = 10001

#: The search's seed namespace begins here (``round_seeds`` constructs
#: ``200_000 + rng * 10_000 + round * 1_000 + odd``); panel blocks stay
#: strictly below it.
SEARCH_SEED_FLOOR = 200_000

#: The second panel region, opened 2026-09-04 when the sub-200k region
#: filled (the block after impden48's would have crossed the floor).
#: It sits strictly above every other namespace: searches top out below
#: 500_000 and evolution constructs ``500_000 + rng * 10_000 +
#: generation * 1_000`` with rng capped at :data:`scripts.evolve.RNG_CAP`
#: (= 49), so no evolution seed reaches 996_000. Unbounded above --
#: engine seeds are 31-bit safe and a panel consumes 5_000 per run.
PANEL_HIGH_FLOOR = 1_000_000

#: Where allocation starts inside the high region.
FIRST_HIGH_SEED = 1_000_001


class PanelError(RwBotError):
    """A panel could not be laid out.

    Args:
        code: ``RW-PANEL-002`` when fewer than one pair is asked for,
            ``RW-PANEL-003`` when a relaunch's request does not match the
            batch's existing job file, ``RW-PANEL-004`` when a sweep
            file's job line carries no readable seed. ``RW-PANEL-001``
            (namespace exhausted) fired once, 2026-09-04, and was retired
            the same day by the high region -- allocation can no longer
            exhaust.
        message: Human-readable description of the refusal.
    """


def used_seeds(jobs_dir: Path) -> set[int]:
    """Every seed any sweep file has ever named, committed or generated.

    Disjointness needs the SEED COLUMN, not the whole job, and the seed
    has been field two of every pipe format this project has ever
    written: the current ``label|seed|doctrine|samples`` and the
    pre-doctrine seven-field era still standing in the committed
    historical sweeps (``aggression.txt`` and kin, which the strict job
    parser rightly refuses to read as jobs). So this reads the one
    column both eras agree on and refuses anything that breaks even
    that.

    Args:
        jobs_dir: The sweep-file directory, searched recursively so the
            gitignored ``sweeps/search`` round files count too -- a
            search that ran on this machine has consumed its seeds no
            matter what git thinks of the file.

    Returns:
        The union of every job line's seed.

    Raises:
        PanelError: ``RW-PANEL-004`` naming the file and line when a
            non-comment line has fewer than four pipe fields or a
            non-integer second field -- a line whose seed cannot be read
            is a seed that cannot be proven disjoint.
    """
    used: set[int] = set()
    for path in sorted(jobs_dir.rglob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            parts = stripped.split("|")
            if len(parts) < 4 or not parts[1].strip().isdigit():
                raise PanelError(
                    "RW-PANEL-004",
                    f"{path.as_posix()}: cannot read a seed from job line "
                    f"{stripped!r}; every format this project has written "
                    "carries an integer seed as pipe field two",
                )
            used.add(int(parts[1]))
    return used


def seed_block(used: set[int]) -> tuple[int, int]:
    """The next thousand-aligned block above everything panels have used.

    Allocates below :data:`SEARCH_SEED_FLOOR` while room remains -- the
    original region -- and continues in the high region above
    :data:`PANEL_HIGH_FLOOR` once a low block would cross the floor,
    which the block after impden48's did on 2026-09-04. Colliding with a
    search or evolution block was never an acceptable fallback; the high
    region sits above both by construction.

    Args:
        used: Every consumed seed.

    Returns:
        ``(start, stop)`` for :func:`fresh_seeds`.
    """
    below_floor = {seed for seed in used if seed < SEARCH_SEED_FLOOR}
    start = FIRST_SEED if below_floor == set() else (max(below_floor) // 1000 + 1) * 1000 + 1
    if start + SEED_BLOCK <= SEARCH_SEED_FLOOR:
        return start, start + SEED_BLOCK
    high = {seed for seed in used if seed >= PANEL_HIGH_FLOOR}
    start = FIRST_HIGH_SEED if high == set() else (max(high) // 1000 + 1) * 1000 + 1
    return start, start + SEED_BLOCK


def panel_job_lines(
    batch: str,
    control_doctrine: str,
    arm_label: str,
    arm_doctrine: str,
    seeds: Sequence[int],
) -> tuple[str, ...]:
    """The panel's job file content, pairs interleaved seed by seed.

    Args:
        batch: The panel's batch name.
        control_doctrine: Repository path of the control arm's doctrine.
        arm_label: The candidate arm's label.
        arm_doctrine: Repository path of the candidate arm's doctrine.
        seeds: The panel's fresh seeds.

    Returns:
        Header comments naming the layout, then one control line and one
        arm line per seed.
    """
    lines = [
        f"# Paired panel {batch}: control ({control_doctrine}) vs",
        f"# {arm_label} ({arm_doctrine}), {len(seeds)} fresh seeds",
        f"# {seeds[0]}-{seeds[-1]}, {SAMPLES} samples. Laid out by",
        "# scripts/panel.py; judged by the pairs + margin report.",
        "#",
        "# label | seed | doctrine | samples",
    ]
    for seed in seeds:
        lines.append(f"control|{seed}|{control_doctrine}|{SAMPLES}")
        lines.append(f"{arm_label}|{seed}|{arm_doctrine}|{SAMPLES}")
    return tuple(lines)


def run_panel(
    runner: RoundRunner,
    batch: str,
    control_doctrine: str,
    arm_label: str,
    arm_doctrine: str,
    pairs: int,
    sweeps_root: Path = SWEEP_ROOT,
    jobs_dir: Path = JOBS_DIR,
) -> tuple[str, ...]:
    """Lay out, play, and judge one paired panel.

    Args:
        runner: Who plays the panel -- the cluster, through the same
            transport the search rides.
        batch: The panel's batch name; scorecards file under it.
        control_doctrine: Repository path of the control arm's doctrine.
        arm_label: The candidate arm's label in job lines and scorecards.
        arm_doctrine: Repository path of the candidate arm's doctrine.
        pairs: How many paired seeds to play.
        sweeps_root: Where scorecards land locally.
        jobs_dir: Where sweep files live and the panel's job file lands.

    Returns:
        The judgement lines -- the paired win report, then the margin
        report -- each also written the moment it is produced.

    Raises:
        PanelError: ``RW-PANEL-002`` when fewer than one pair is asked
            for, ``RW-PANEL-003`` when a relaunch's pair count does not
            match the batch's existing job file, or through
            :func:`seed_block` on namespace exhaustion.
        SweepError: Through :func:`used_seeds` or :func:`fresh_seeds`.
        ClusterRoundError: Through the runner, on a round that cannot
            deliver its scorecards.
    """
    if pairs < 1:
        raise PanelError("RW-PANEL-002", f"a panel needs at least one pair, got {pairs}")
    own_file = jobs_dir / f"{batch}.txt"
    if own_file.exists():
        # RELAUNCH, not a new panel: the batch's own job file already
        # names its seeds, and picking fresh ones would abandon every
        # match the first invocation submitted. Reusing the file makes a
        # relaunch idempotent the same way the search's rounds are -- the
        # converge inside the runner dedupes against the cluster.
        # Discovered before it bit: the session harness sweeps long-lived
        # local drivers (2026-09-03), so a panel MUST survive a mid-drain
        # kill plus relaunch.
        recorded = parse_jobs(own_file.read_text(encoding="utf-8").splitlines())
        seeds = tuple(job["seed"] for job in recorded if job["label"] == arm_label)
        if len(seeds) != pairs:
            raise PanelError(
                "RW-PANEL-003",
                f"{own_file.as_posix()} exists with {len(seeds)} {arm_label!r} "
                f"seed(s) but {pairs} pairs were asked for; a relaunch must "
                "repeat the original request exactly, and a different panel "
                "needs a different batch name",
            )
    else:
        used = used_seeds(jobs_dir)
        start, stop = seed_block(used)
        seeds = fresh_seeds(used, pairs, start, stop)
    lines_out = panel_job_lines(batch, control_doctrine, arm_label, arm_doctrine, seeds)
    host_hooks.write_line(
        f"# panel {batch}: {pairs} pairs, seeds {seeds[0]}-{seeds[-1]}, control vs {arm_label}"
    )
    runner.run(batch, lines_out)

    produced: list[str] = []

    def note(text: str) -> None:
        host_hooks.write_line(text)
        produced.append(text)

    batch_dir = sweeps_root / batch
    control_grades = read_grades(batch_dir, "control")
    arm_grades = read_grades(batch_dir, arm_label)
    paired = pair_up(control_grades, arm_grades)
    only_control = len(set(control_grades) - set(arm_grades))
    only_arm = len(set(arm_grades) - set(control_grades))
    for line in format_pairs(
        f"{batch}:control", f"{batch}:{arm_label}", paired, only_control, only_arm
    ):
        note(line)
    for line in report(batch, batch_margins(batch_dir)):
        note(line)
    return tuple(produced)


def main(
    argv: Sequence[str] | None = None,
    sweeps_root: Path = SWEEP_ROOT,
    jobs_dir: Path = JOBS_DIR,
) -> int:
    """Run one panel from the command line.

    Args:
        argv: ``<hpc3:workspace.json> <batch> <control-doctrine>
            <arm-label> <arm-doctrine> <pairs> <difficulty>``. ``None``
            reads ``sys.argv[1:]``.
        sweeps_root: Where scorecards land, injectable for tests.
        jobs_dir: The sweep-file directory, injectable for tests.

    Returns:
        ``EXIT_OK``, or ``EXIT_BAD_USAGE`` on a bad argument count or a
        destination that is not a cluster route -- the panel is a
        cluster tool; the queue path never grew a panel mode and a
        second half-supported transport would be a fallback.

    Raises:
        PanelError: Through :func:`run_panel`.
        ClusterRoundError: Through the runner.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) != 7 or not args[0].startswith(CLUSTER_PREFIX):
        sys.stdout.write(
            "usage: panel <hpc3:workspace.json> <batch> <control-doctrine> "
            "<arm-label> <arm-doctrine> <pairs> <difficulty>\n"
        )
        return EXIT_BAD_USAGE
    runner = ClusterRound(
        config=args[0][len(CLUSTER_PREFIX) :],
        host=CLUSTER_HOST,
        cluster_root=CLUSTER_ROOT,
        map_path=MAP_PATH,
        difficulty=int(args[6]),
        fast_forward=FAST_FORWARD,
        scratch=CLUSTER_SCRATCH,
        sweeps_root=sweeps_root,
        jobs_dir=jobs_dir,
        poll_seconds=POLL_SECONDS,
    )
    run_panel(runner, args[1], args[2], args[3], args[4], int(args[5]), sweeps_root, jobs_dir)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
