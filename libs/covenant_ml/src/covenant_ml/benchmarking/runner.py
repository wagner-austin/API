"""Orchestration of a full benchmark run.

The runner owns the measurement protocol and nothing else: it does not know
which libraries it is comparing, does not read the clock directly, and does
not load data. Every collaborator arrives through a Protocol, so the whole
protocol is exercisable from fakes and real learners alike.

Two properties of the protocol are deliberate and load-bearing:

* **Both learners are measured in the same run.** A fixed reference measured
  now is the only way to tell a code change from a machine-state change; a
  number carried forward from an older manifest cannot distinguish them.
* **Order alternates across seeds.** Whichever learner runs first at a seed
  gets the coolest CPU, so a fixed order hands one of them a systematic
  advantage.
"""

from __future__ import annotations

from collections.abc import Sequence

from . import _test_hooks
from .protocols import DataSplit, SplitFactoryProto, TrainerProto
from .quality import compute_quality
from .timing import summarize_timings
from .types import (
    ERR_INVALID_REPEATS,
    ERR_NO_SEEDS,
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
    ran_first: bool,
) -> SeedResult:
    """Measure one learner at one seed.

    Runs ``config["warmups"]`` discarded fits, then ``config["repeats"]``
    timed fits, and scores the held-out fold with the final fitted model.

    Args:
        trainer: The learner to measure.
        split: The partition to train and score on.
        seed: Seed for the split and the model's internal randomness.
        config: Shared hyperparameters, including repeat and warm-up counts.
        ran_first: Whether this learner was measured before the other at this
            seed.

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
        "ran_first": ran_first,
        "timing": summarize_timings(samples_s),
        "quality": compute_quality(split.y_test, positive_proba),
        "mean_leaves": fitted.mean_leaves(),
    }


def run_benchmark(
    first_trainer: TrainerProto,
    second_trainer: TrainerProto,
    build_split: SplitFactoryProto,
    seeds: Sequence[int],
    config: BenchmarkConfig,
    dataset: DatasetInfo,
) -> BenchmarkManifest:
    """Measure both learners across every seed and assemble the manifest.

    Args:
        first_trainer: The learner measured first at even-indexed seeds.
        second_trainer: The learner measured first at odd-indexed seeds.
        build_split: Produces the partition for a seed.
        seeds: Seeds to measure, in execution order.
        config: Shared hyperparameters.
        dataset: Identity of the input data.

    Returns:
        The complete manifest for this invocation.

    Raises:
        ValueError: If ``seeds`` is empty, or if ``config["repeats"]`` is less
            than one.
    """
    if len(seeds) == 0:
        raise ValueError(f"[{ERR_NO_SEEDS}] At least one seed is required, got none")

    results: list[SeedResult] = []
    for index, seed in enumerate(seeds):
        split = build_split(seed)
        first_leads = index % 2 == 0
        leader = first_trainer if first_leads else second_trainer
        follower = second_trainer if first_leads else first_trainer

        results.append(measure_trainer(leader, split, seed, config, ran_first=True))
        results.append(measure_trainer(follower, split, seed, config, ran_first=False))

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "estimator": "median",
        "config": config,
        "dataset": dataset,
        "seeds": list(seeds),
        "results": results,
    }


__all__ = ["measure_trainer", "run_benchmark"]
