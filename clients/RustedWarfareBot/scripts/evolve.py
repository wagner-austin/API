"""The population search driver: sample, play, select, refit, repeat.

Usage:
    python -m scripts.evolve <hpc3:workspace.json> <name> [rng-seed]

The cross-entropy method over the army-composition simplex
(``[[harness-population-search]]``): each generation samples a
population of genomes from a per-dimension Gaussian in LOGIT space
(softmax keeps every sample on the simplex with no clipping or repair),
compiles each to an ordinary doctrine through
:func:`rw_bot.harness.genome.compile_genome`, plays the whole generation
as ONE interleaved batch against the sitting champion on fresh seeds,
ranks by paired margin delta, and refits the Gaussian to the elite.

v1 is the simplex ONLY -- four dimensions. The integer-knob space is
deliberately absent from the genome: the night this was written, the
knob vocabulary was measured a confirmed local optimum (vhsearch4 flat
in every direction, flame2 rejected at the bar), so re-searching it
inside the genome would spend the small population's power on a space
already known flat. Knobs rejoin as v1.1 if the simplex proves the loop.

Every draw comes from one seeded generator, so a relaunch replays the
identical genomes and the cluster's converge turns completed generations
into instant fast-forwards -- the same boundary-resumable property the
doctrine search and the panel already have, under the same harness that
sweeps long-lived drivers.

THE LOOP PROPOSES; THE BAR DISPOSES. The final report names the best
genome's doctrine, and that doctrine faces laws six and nine on
untouched seeds like every other arm. Generation fitness carries
selection bias by construction and is never evidence for adoption --
strike5000 and flame2 were both refuted the night this file was born,
and both had looked consistent inside their own searches.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Sequence
from math import exp, sqrt
from pathlib import Path

from rw_bot.harness import _test_hooks as host_hooks
from rw_bot.harness.cluster_round import ClusterRound
from rw_bot.harness.genome import ARMY_VOCABULARY, compile_genome
from rw_bot.harness.margin import batch_margins
from rw_bot.harness.search import paired_delta
from rw_bot.policy.doctrine_file import format_doctrine, parse_doctrine_lines
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
from scripts.search_specs import require_search_spec

EXIT_OK = 0
EXIT_BAD_USAGE = 2

#: Genomes sampled per generation. Sixteen arms plus the shared control
#: at eight pairs each is 136 matches -- the proven round size.
POPULATION = 16

#: Elite genomes the refit keeps. A quarter of the population is the
#: cross-entropy method's ordinary selection pressure.
ELITE = 4

#: Generations per run. Six spends ~800 matches -- one evening.
GENERATIONS = 6

#: Paired seeds per generation.
PAIRS = 8

#: Initial per-dimension standard deviation in logit space. One unit of
#: logit is roughly a factor-e weight swing -- wide enough to explore
#: compositions the champion never fielded.
SIGMA_START = 1.0

#: The refit never tightens below this, so a lucky elite cannot collapse
#: exploration to a point after one generation.
SIGMA_FLOOR = 0.15

#: Where per-generation candidate doctrines land, gitignored run
#: artifacts like the search's variants.
VARIANT_DIR = Path("doctrines/evolve")

#: Where generation job files land, committed alongside the search's.
JOBS_DIR = Path("sweeps/evolve")

#: Seed namespace: panels sit below 200k, search rounds at
#: ``200k + rng*10k + round*1k``, and generations here at
#: ``500k + rng*10k + generation*1k`` -- disjoint from both by
#: construction for every rng below 30.
SEED_FLOOR = 500_000

#: The largest rng seed an evolution may run under. Caps the namespace:
#: ``500_000 + 49 * 10_000 + 5 * 1_000 + 15`` stays below the panel high
#: region at 1_000_000 (``scripts.panel.PANEL_HIGH_FLOOR``), keeping the
#: two namespaces disjoint by construction rather than by luck.
RNG_CAP = 49

#: Sample budget per match, the regime invariant.
SAMPLES = 10000


def generation_seeds(rng_seed: int, generation: int) -> tuple[int, ...]:
    """Fresh, deterministic seeds for one generation.

    Args:
        rng_seed: The run's reproducibility anchor, at most
            :data:`RNG_CAP`.
        generation: Which generation the seeds are for.

    Returns:
        Distinct odd seeds no other generation of this run uses.

    Raises:
        ValueError: When ``rng_seed`` exceeds :data:`RNG_CAP` or is
            negative -- a seed past the cap would push this run's matches
            into the panel high region, and a collision there would
            silently pair a panel against replayed evolution matches.
    """
    if not 0 <= rng_seed <= RNG_CAP:
        raise ValueError(
            f"rng seed {rng_seed} is outside [0, {RNG_CAP}]; past the cap the "
            f"run's seeds would cross into the panel high region at 1,000,000"
        )
    base = SEED_FLOOR + rng_seed * 10_000 + generation * 1_000
    return tuple(base + 2 * k + 1 for k in range(PAIRS))


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """Map logits onto the simplex.

    Args:
        logits: One value per vocabulary unit.

    Returns:
        Positive weights summing to one, in the same order.
    """
    peak = max(logits)
    exps = [exp(value - peak) for value in logits]
    total = sum(exps)
    return tuple(value / total for value in exps)


def sample_population(
    rng: random.Random, mean: Sequence[float], sigma: Sequence[float]
) -> tuple[tuple[float, ...], ...]:
    """Draw one generation's genomes in logit space.

    Args:
        rng: The run's single seeded generator; draws advance its state,
            which is what makes a relaunch replay identical genomes.
        mean: Per-dimension logit mean.
        sigma: Per-dimension logit standard deviation.

    Returns:
        :data:`POPULATION` logit vectors.
    """
    return tuple(
        tuple(rng.gauss(m, s) for m, s in zip(mean, sigma, strict=True)) for _ in range(POPULATION)
    )


def refit(
    elite: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Fit the next generation's Gaussian to the elite.

    Args:
        elite: The top logit vectors, at least one.

    Returns:
        Per-dimension mean and standard deviation, the deviation floored
        at :data:`SIGMA_FLOOR`.
    """
    dims = len(elite[0])
    mean: list[float] = []
    sigma: list[float] = []
    for d in range(dims):
        values = [member[d] for member in elite]
        m = sum(values) / len(values)
        # (value - m) * (value - m) rather than ** 2: int-exponent power
        # types as Any in the stubs, and this module forbids Any.
        variance = sum((value - m) * (value - m) for value in values) / len(values)
        mean.append(m)
        sigma.append(max(sqrt(variance), SIGMA_FLOOR))
    return tuple(mean), tuple(sigma)


def member_label(generation: int, index: int) -> str:
    """Name one genome's arm.

    Args:
        generation: The generation number.
        index: The member's index within it.

    Returns:
        ``g<generation>m<index>``, the label its scorecards file under.
    """
    return f"g{generation}m{index}"


def write_generation(
    base_path: Path,
    genomes: Sequence[Sequence[float]],
    generation: int,
    variant_dir: Path,
) -> None:
    """Compile and write every genome's doctrine file.

    Args:
        base_path: The champion doctrine the population perturbs.
        genomes: The generation's logit vectors.
        generation: The generation number, for labels.
        variant_dir: Where the doctrine files land, created if absent.

    Raises:
        OSError: When a file cannot be written.
        GenomeError: Through the compiler, on a defect this module would
            itself have caused.
    """
    base = parse_doctrine_lines(base_path.read_text(encoding="utf-8").splitlines())
    variant_dir.mkdir(parents=True, exist_ok=True)
    for index, logits in enumerate(genomes):
        weights = dict(zip(ARMY_VOCABULARY, softmax(logits), strict=True))
        label = member_label(generation, index)
        variant = compile_genome(base, weights, {}, label)
        path = variant_dir / f"{label}.doctrine"
        path.write_text("".join(f"{line}\n" for line in format_doctrine(variant)), encoding="utf-8")


def generation_job_lines(
    base_path: Path,
    genomes: Sequence[Sequence[float]],
    generation: int,
    seeds: Sequence[int],
    variant_dir: Path,
) -> tuple[str, ...]:
    """The generation's job lines, pairs interleaved seed by seed.

    Args:
        base_path: The control doctrine's repository path.
        genomes: The generation's logit vectors.
        generation: The generation number.
        seeds: The generation's fresh seeds.
        variant_dir: Where the candidates' doctrine files live.

    Returns:
        One control line plus one line per member, per seed.
    """
    control = base_path.as_posix()
    lines: list[str] = []
    for seed in seeds:
        lines.append(f"control|{seed}|{control}|{SAMPLES}")
        for index in range(len(genomes)):
            label = member_label(generation, index)
            doctrine = (variant_dir / f"{label}.doctrine").as_posix()
            lines.append(f"{label}|{seed}|{doctrine}|{SAMPLES}")
    return tuple(lines)


def run_evolution(
    runner: RoundRunner,
    name: str,
    rng_seed: int,
    sweeps_root: Path = SWEEP_ROOT,
    variant_dir: Path = VARIANT_DIR,
) -> tuple[str, ...]:
    """Run every generation and return the report.

    Args:
        runner: Who plays each generation -- the cluster.
        name: The run's name; generation batches file as ``<name>-g<i>``.
        rng_seed: Reproducibility anchor for every draw and seed.
        sweeps_root: Where batch scorecards land, injectable for tests.
        variant_dir: Where candidate doctrines land, injectable for tests.

    Returns:
        Report lines, each also written the moment it happens.

    Raises:
        ClusterRoundError: Through the runner.
        GenomeError: Through the compiler.
    """
    lines: list[str] = []

    def note(text: str) -> None:
        host_hooks.write_line(text)
        lines.append(text)

    spec = require_search_spec("vh")
    base_path = Path(spec["base"])
    rng = random.Random(rng_seed)
    mean: tuple[float, ...] = (0.0,) * len(ARMY_VOCABULARY)
    sigma: tuple[float, ...] = (SIGMA_START,) * len(ARMY_VOCABULARY)
    note(
        f"# evolve {name} (rng {rng_seed}): population {POPULATION}, elite {ELITE}, "
        f"{GENERATIONS} generations of {PAIRS} pairs vs {base_path.as_posix()}"
    )

    best_label = ""
    best_score = float("-inf")
    for generation in range(GENERATIONS):
        batch = f"{name}-g{generation}"
        genomes = sample_population(rng, mean, sigma)
        write_generation(base_path, genomes, generation, variant_dir)
        seeds = generation_seeds(rng_seed, generation)
        note(f"# generation {generation}: {POPULATION} members, {PAIRS} pairs")
        runner.run(batch, generation_job_lines(base_path, genomes, generation, seeds, variant_dir))
        margins = batch_margins(sweeps_root / batch)
        scored: list[tuple[float, int]] = []
        for index in range(POPULATION):
            n, delta, sd = paired_delta(margins, member_label(generation, index), "control")
            scored.append((delta, index))
            note(
                f"{batch} {member_label(generation, index):8} n={n:3}"
                f"  margin delta {delta:+.3f} (sd {sd:.3f})"
            )

        def rank(pair: tuple[float, int]) -> tuple[float, int]:
            return (-pair[0], pair[1])

        scored.sort(key=rank)
        elite = [genomes[index] for _, index in scored[:ELITE]]
        top_delta, top_index = scored[0]
        if top_delta > best_score:
            best_score = top_delta
            best_label = member_label(generation, top_index)
        weights = softmax(elite[0])
        note(
            f"# generation {generation} elite mean weights: "
            + ", ".join(
                f"{unit}={weight:.2f}"
                for unit, weight in zip(ARMY_VOCABULARY, weights, strict=True)
            )
        )
        mean, sigma = refit(elite)
    note(
        f"# best member: {best_label} (margin delta {best_score:+.3f}); its doctrine "
        f"file under {variant_dir.as_posix()} goes to the win-bar panel next -- "
        "generation fitness is selection-biased and adopts nothing (laws six, nine)"
    )
    return tuple(lines)


def main(
    argv: Sequence[str] | None = None,
    sweeps_root: Path = SWEEP_ROOT,
    variant_dir: Path = VARIANT_DIR,
    jobs_dir: Path = JOBS_DIR,
) -> int:
    """Run one evolution from the command line.

    Args:
        argv: ``<hpc3:workspace.json> <name> [rng-seed]``. ``None`` reads
            the process arguments.
        sweeps_root: Where batch scorecards land, injectable for tests.
        variant_dir: Where candidate doctrines land, injectable for tests.
        jobs_dir: Where generation job files land, injectable for tests.

    Returns:
        ``EXIT_OK``, or ``EXIT_BAD_USAGE`` on a bad argument count or a
        non-cluster destination -- a 136-match generation belongs on the
        cluster, and the queue path never grew a population mode.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (2, 3) or not args[0].startswith(CLUSTER_PREFIX):
        sys.stdout.write("usage: evolve <hpc3:workspace.json> <name> [rng-seed]\n")
        return EXIT_BAD_USAGE
    rng_seed = int(args[2]) if len(args) == 3 else 0
    runner = ClusterRound(
        config=args[0][len(CLUSTER_PREFIX) :],
        host=CLUSTER_HOST,
        cluster_root=CLUSTER_ROOT,
        map_path=MAP_PATH,
        difficulty=require_search_spec("vh")["difficulty"],
        fast_forward=FAST_FORWARD,
        scratch=CLUSTER_SCRATCH,
        sweeps_root=sweeps_root,
        jobs_dir=jobs_dir,
        poll_seconds=POLL_SECONDS,
    )
    run_evolution(runner, args[1], rng_seed, sweeps_root, variant_dir)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
