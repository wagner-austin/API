"""Orchestration of a full benchmark run.

The runner owns the measurement protocol and nothing else: it does not know
which libraries it is comparing, does not read the clock directly, and does
not load data. Every collaborator arrives through a Protocol, so the whole
protocol is exercisable from fakes and real learners alike.

Two properties of the protocol are deliberate and load-bearing:

* **Every arm is measured in the same run.** A fixed reference measured now is
  the only way to tell a code change from a machine-state change; a number
  carried forward from an older manifest cannot distinguish them.
* **Order rotates across seeds.** Whichever arm runs first at a seed gets the
  coolest CPU, so a fixed order hands one of them a systematic advantage.
  With ``k`` arms the order rotates by one slot per seed, so over any ``k``
  consecutive seeds each arm occupies each slot exactly once.

Arms are a list rather than a fixed pair so a variant of one learner can be
compared against its own baseline and against the reference implementation in
a single manifest — which is the point of a variant axis. Two arms remains the
common case and is not special-cased.
"""

from __future__ import annotations

from collections.abc import Sequence

from . import _test_hooks
from .protocols import DataSplit, SplitFactoryProto, TrainerProto
from .quality import compute_quality
from .timing import summarize_timings
from .types import (
    ERR_DUPLICATE_TRAINER,
    ERR_INVALID_REPEATS,
    ERR_NO_SEEDS,
    ERR_TOO_FEW_TRAINERS,
    MANIFEST_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkManifest,
    DatasetInfo,
    SeedResult,
)


def measure_trainer(
    trainer: TrainerProto,
    split: DataSplit,
    seed: int,
    config: BenchmarkConfig,
    position: int,
) -> SeedResult:
    """Measure one learner at one seed.

    Runs ``config["warmups"]`` discarded fits, then ``config["repeats"]``
    timed fits, and scores the held-out fold with the final fitted model.

    Args:
        trainer: The learner to measure.
        split: The partition to train and score on.
        seed: Seed for the split and the model's internal randomness.
        config: Shared hyperparameters, including repeat and warm-up counts.
        position: Zero-based slot this arm occupied at this seed.

    Returns:
        The learner's record for this seed.

    Raises:
        ValueError: If ``config["repeats"]`` is less than one, which would
            leave no fitted model to score and nothing to summarise.
    """
    repeats = config["repeats"]
    if repeats < 1:
        raise ValueError(
            f"[{ERR_INVALID_REPEATS}] config['repeats'] must be at least 1, got {repeats}"
        )

    for _ in range(config["warmups"]):
        trainer.fit(split, seed)

    samples_s: list[float] = []
    # All but the final timed fit; the final one is run separately so the
    # fitted model it produces is bound without an optional.
    for _ in range(repeats - 1):
        started = _test_hooks.monotonic_clock()
        trainer.fit(split, seed)
        samples_s.append(_test_hooks.monotonic_clock() - started)

    started = _test_hooks.monotonic_clock()
    fitted = trainer.fit(split, seed)
    samples_s.append(_test_hooks.monotonic_clock() - started)

    positive_proba = fitted.predict_positive_proba(split.x_test)
    return {
        "model": trainer.model_name,
        "seed": seed,
        "position": position,
        "timing": summarize_timings(samples_s),
        "quality": compute_quality(split.y_test, positive_proba),
        "mean_leaves": fitted.mean_leaves(),
    }


def run_benchmark(
    trainers: Sequence[TrainerProto],
    build_split: SplitFactoryProto,
    seeds: Sequence[int],
    config: BenchmarkConfig,
    dataset: DatasetInfo,
) -> BenchmarkManifest:
    """Measure every arm across every seed and assemble the manifest.

    Args:
        trainers: The arms to compare, at least two. Their order sets the
            rotation at the first seed.
        build_split: Produces the partition for a seed.
        seeds: Seeds to measure, in execution order.
        config: Shared hyperparameters.
        dataset: Identity of the input data.

    Returns:
        The complete manifest for this invocation.

    Raises:
        ValueError: If fewer than two arms are given, if two arms share a
            name, if ``seeds`` is empty, or if ``config["repeats"]`` is less
            than one.
        RuntimeError: If the process cannot opt out of power throttling, which
            would leave every fit time attributable to an unknown mix of two
            power regimes.
    """
    if len(trainers) < 2:
        raise ValueError(
            f"[{ERR_TOO_FEW_TRAINERS}] At least two arms are required to compare, "
            f"got {len(trainers)}"
        )

    # A manifest is grouped by arm name, so two arms sharing one is not a
    # comparison with a duplicate — it is two different configurations merged
    # into one series, silently.
    names = [trainer.model_name for trainer in trainers]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        joined = ", ".join(f"'{name}'" for name in duplicates)
        raise ValueError(
            f"[{ERR_DUPLICATE_TRAINER}] Each arm must have a distinct name; repeated: {joined}"
        )

    if len(seeds) == 0:
        raise ValueError(f"[{ERR_NO_SEEDS}] At least one seed is required, got none")

    # Once, before the first fit. Windows otherwise demotes this process to a
    # throttled power regime a few seconds in -- measured at up to 13x on this
    # workload, arriving mid-run and never lifting. See `power` for the data;
    # the short version is that rotation cannot cancel a one-way step change,
    # so the opt-out has to happen before anything is timed.
    _test_hooks.power_throttling_opt_out()

    results: list[SeedResult] = []
    for index, seed in enumerate(seeds):
        split = build_split(seed)
        # Rotate by one slot per seed. Over any len(trainers) consecutive
        # seeds every arm occupies every slot exactly once, which is what
        # makes the cold-CPU slot cancel instead of favouring one arm.
        offset = index % len(trainers)
        rotated = list(trainers[offset:]) + list(trainers[:offset])
        for position, trainer in enumerate(rotated):
            results.append(measure_trainer(trainer, split, seed, config, position=position))

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "estimator": "median",
        "config": config,
        "dataset": dataset,
        "seeds": list(seeds),
        "results": results,
    }


__all__ = ["measure_trainer", "run_benchmark"]
