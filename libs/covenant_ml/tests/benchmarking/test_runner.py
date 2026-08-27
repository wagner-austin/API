"""Tests for the benchmark measurement protocol.

Collaborators are real fake implementations of the package's Protocols,
injected through ``_test_hooks`` by save-and-restore. Nothing is mocked.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.comparability import NO_VALUE
from platform_core.determinism_env import SINGLE_THREAD
from platform_core.determinism_record import determinism_record
from platform_core.testing import sample_run_fingerprint

from covenant_ml.benchmarking import _test_hooks
from covenant_ml.benchmarking.protocols import DataSplit, TrainedModelProto
from covenant_ml.benchmarking.runner import measure_trainer, run_benchmark
from covenant_ml.benchmarking.types import (
    ERR_DUPLICATE_TRAINER,
    ERR_INVALID_REPEATS,
    ERR_NO_SEEDS,
    ERR_POWER_THROTTLING,
    ERR_TOO_FEW_TRAINERS,
    MANIFEST_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkModelName,
    DatasetInfo,
)

#: A stated configuration, so every manifest these tests build carries the
#: axes a published one must. Built through the canonical builder rather than
#: written out, so it cannot fall behind the type.
_FINGERPRINT = sample_run_fingerprint(
    image_digest="sha256:" + "ef" * 32,
    gpu_model=NO_VALUE,
    driver_version=NO_VALUE,
    determinism=determinism_record("cpu", {"OMP_NUM_THREADS": SINGLE_THREAD}),
)


class SteppingClock:
    """A clock that advances by a fixed step on every other reading.

    The runner reads the clock twice per timed fit, so a constant step makes
    each measured duration exactly ``step``.
    """

    def __init__(self, step: float) -> None:
        """Bind the per-reading increment.

        Args:
            step: Seconds added between consecutive readings.
        """
        self._step = step
        self._now = 0.0
        self.reads = 0

    def __call__(self) -> float:
        """Read and advance the clock.

        Returns:
            The current value, before advancing.
        """
        self.reads += 1
        current = self._now
        self._now += self._step
        return current


class ScriptedClock:
    """A clock that returns a fixed sequence of readings."""

    def __init__(self, readings: list[float]) -> None:
        """Bind the reading sequence.

        Args:
            readings: Values returned in order.
        """
        self._readings = readings
        self._index = 0

    def __call__(self) -> float:
        """Return the next scripted reading.

        Returns:
            The next value in the sequence.
        """
        value = self._readings[self._index]
        self._index += 1
        return value


class FakeTrainedModel:
    """A fitted model that returns fixed predictions and leaf count."""

    def __init__(self, positive_proba: float, mean_leaves: float) -> None:
        """Bind the values this model reports.

        Args:
            positive_proba: Probability returned for every row.
            mean_leaves: Mean leaves per tree to report.
        """
        self._positive_proba = positive_proba
        self._mean_leaves = mean_leaves

    def predict_positive_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the fixed probability for every row.

        Args:
            x: Feature matrix.

        Returns:
            Constant probabilities, one per row.
        """
        return np.full(len(x), self._positive_proba, dtype=np.float64)

    def mean_leaves(self) -> float:
        """Return the configured leaf count.

        Returns:
            Mean leaves per tree.
        """
        return self._mean_leaves


class RecordingTrainer:
    """A trainer that records every fit it is asked to perform."""

    def __init__(self, name: BenchmarkModelName, mean_leaves: float = 10.0) -> None:
        """Bind the trainer's identity.

        Args:
            name: Model name to report.
            mean_leaves: Leaf count its fitted models report.
        """
        self._name = name
        self._mean_leaves = mean_leaves
        self.fit_seeds: list[int] = []

    @property
    def model_name(self) -> BenchmarkModelName:
        """Return the configured model name.

        Returns:
            The model name.
        """
        return self._name

    def fit(self, split: DataSplit, seed: int) -> TrainedModelProto:
        """Record the fit and return a fake fitted model.

        Args:
            split: Ignored.
            seed: Recorded.

        Returns:
            A fake fitted model.
        """
        self.fit_seeds.append(seed)
        return FakeTrainedModel(positive_proba=0.5, mean_leaves=self._mean_leaves)


def make_split(n_rows: int = 8) -> DataSplit:
    """Build a small partition.

    Args:
        n_rows: Rows in each fold.

    Returns:
        The partition.
    """
    features: NDArray[np.float64] = np.zeros((n_rows, 2), dtype=np.float64)
    labels: NDArray[np.int64] = np.arange(n_rows, dtype=np.int64) % 2
    return DataSplit(
        x_train=features,
        y_train=labels,
        x_val=features,
        y_val=labels,
        x_test=features,
        y_test=labels,
    )


def make_config(repeats: int = 3, warmups: int = 2) -> BenchmarkConfig:
    """Build a configuration with the given repeat counts.

    Args:
        repeats: Timed fits per model per seed.
        warmups: Discarded fits before timing.

    Returns:
        The configuration.
    """
    return {
        "n_estimators": 2,
        "max_depth": 2,
        "learning_rate": 0.1,
        "max_bins": 8,
        "min_data_in_leaf": 1,
        "num_leaves": 3,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "repeats": repeats,
        "warmups": warmups,
    }


def make_dataset_info() -> DatasetInfo:
    """Build a dataset identity.

    Returns:
        The identity record.
    """
    return {"sha256": "b" * 64, "n_rows": 8, "n_features": 2}


def constant_split(seed: int) -> DataSplit:
    """Return the same partition for any seed.

    Args:
        seed: Ignored.

    Returns:
        A fixed partition.
    """
    return make_split()


class RecordingOptOut:
    """Records power-throttling opt-out requests instead of making them."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls = 0

    def __call__(self) -> None:
        """Record one opt-out request.

        Returns:
            None.
        """
        self.calls += 1


class FailingOptOut:
    """An opt-out that reports the platform refused the request."""

    def __call__(self) -> None:
        """Refuse.

        Raises:
            RuntimeError: Always, carrying the traceable code.
        """
        raise RuntimeError(f"[{ERR_POWER_THROTTLING}] refused")


@pytest.fixture()
def stepping_clock() -> Generator[SteppingClock, None, None]:
    """Install a stepping clock for the duration of a test.

    Yields:
        The installed clock.
    """
    previous = _test_hooks.monotonic_clock
    clock = SteppingClock(step=0.5)
    _test_hooks.monotonic_clock = clock
    yield clock
    _test_hooks.monotonic_clock = previous


@pytest.fixture()
def recorded_opt_out() -> Generator[RecordingOptOut, None, None]:
    """Install a recording power-throttling opt-out.

    Keeps the suite from altering the host's power state while still letting a
    test assert the request was made.

    Yields:
        The installed recorder.
    """
    previous = _test_hooks.power_throttling_opt_out
    recorder = RecordingOptOut()
    _test_hooks.power_throttling_opt_out = recorder
    yield recorder
    _test_hooks.power_throttling_opt_out = previous


def test_power_throttling_is_disabled_once_per_run(
    stepping_clock: SteppingClock,
    recorded_opt_out: RecordingOptOut,
) -> None:
    """Once for the whole run, not once per seed or per arm.

    Windows demotes this process to a throttled power regime a few seconds in,
    and the demotion never lifts. Opting out after the first fit would leave
    the earliest measurements in a different regime from the rest.
    """
    run_benchmark(
        [RecordingTrainer("cleargbm"), RecordingTrainer("lightgbm")],
        constant_split,
        [42, 43, 44],
        make_config(repeats=1, warmups=0),
        make_dataset_info(),
        _FINGERPRINT,
    )
    assert recorded_opt_out.calls == 1


def test_a_refused_opt_out_aborts_the_run(stepping_clock: SteppingClock) -> None:
    """No fallback: a run that cannot opt out must not report timings."""
    previous = _test_hooks.power_throttling_opt_out
    _test_hooks.power_throttling_opt_out = FailingOptOut()
    try:
        with pytest.raises(RuntimeError, match=ERR_POWER_THROTTLING):
            run_benchmark(
                [RecordingTrainer("cleargbm"), RecordingTrainer("lightgbm")],
                constant_split,
                [42],
                make_config(repeats=1, warmups=0),
                make_dataset_info(),
                _FINGERPRINT,
            )
    finally:
        _test_hooks.power_throttling_opt_out = previous


def test_no_fit_runs_before_the_opt_out(stepping_clock: SteppingClock) -> None:
    """The abort must happen before any learner is touched.

    A run that fitted first and opted out afterwards would already have
    produced measurements in the wrong regime.
    """
    trainer = RecordingTrainer("cleargbm")
    previous = _test_hooks.power_throttling_opt_out
    _test_hooks.power_throttling_opt_out = FailingOptOut()
    try:
        with pytest.raises(RuntimeError, match=ERR_POWER_THROTTLING):
            run_benchmark(
                [trainer, RecordingTrainer("lightgbm")],
                constant_split,
                [42],
                make_config(repeats=1, warmups=0),
                make_dataset_info(),
                _FINGERPRINT,
            )
    finally:
        _test_hooks.power_throttling_opt_out = previous
    assert trainer.fit_seeds == []


def test_warmup_fits_run_but_are_not_timed(stepping_clock: SteppingClock) -> None:
    trainer = RecordingTrainer("cleargbm")
    result = measure_trainer(trainer, make_split(), 42, make_config(repeats=3, warmups=2), 0)

    # 2 warmups + 3 timed fits.
    assert len(trainer.fit_seeds) == 5
    # Only the timed fits are summarised.
    assert len(result["timing"]["samples_s"]) == 3
    # The clock is read exactly twice per timed fit.
    assert stepping_clock.reads == 6


def test_each_timed_fit_is_bracketed_by_two_readings() -> None:
    previous = _test_hooks.monotonic_clock
    _test_hooks.monotonic_clock = ScriptedClock([0.0, 1.0, 10.0, 13.0])
    trainer = RecordingTrainer("cleargbm")
    result = measure_trainer(trainer, make_split(), 1, make_config(repeats=2, warmups=0), 0)
    _test_hooks.monotonic_clock = previous

    assert result["timing"]["samples_s"] == [1.0, 3.0]
    assert result["timing"]["canonical_s"] == 2.0


def test_result_carries_model_seed_and_order(stepping_clock: SteppingClock) -> None:
    trainer = RecordingTrainer("lightgbm", mean_leaves=31.0)
    result = measure_trainer(trainer, make_split(), 44, make_config(), position=1)

    assert result["model"] == "lightgbm"
    assert result["seed"] == 44
    assert result["position"] == 1
    assert result["mean_leaves"] == 31.0


def test_every_fit_uses_the_requested_seed(stepping_clock: SteppingClock) -> None:
    trainer = RecordingTrainer("cleargbm")
    measure_trainer(trainer, make_split(), 99, make_config(repeats=2, warmups=1), 0)
    assert trainer.fit_seeds == [99, 99, 99]


def test_zero_repeats_raises(stepping_clock: SteppingClock) -> None:
    trainer = RecordingTrainer("cleargbm")
    with pytest.raises(ValueError, match=ERR_INVALID_REPEATS):
        measure_trainer(trainer, make_split(), 1, make_config(repeats=0), 0)


def test_run_benchmark_alternates_which_model_goes_first(
    stepping_clock: SteppingClock,
) -> None:
    """Whichever model runs first gets the coolest CPU, so order must rotate."""
    manifest = run_benchmark(
        [RecordingTrainer("cleargbm"), RecordingTrainer("lightgbm")],
        constant_split,
        [42, 43, 44],
        make_config(repeats=1, warmups=0),
        make_dataset_info(),
        _FINGERPRINT,
    )

    first_by_seed = {
        result["seed"]: result["model"] for result in manifest["results"] if result["position"] == 0
    }
    assert first_by_seed == {42: "cleargbm", 43: "lightgbm", 44: "cleargbm"}


def test_run_benchmark_rotates_three_arms_through_every_slot(
    stepping_clock: SteppingClock,
) -> None:
    """With k arms, k consecutive seeds must give each arm each slot once.

    A rotation that merely swapped the leader would leave the arms behind it
    in a fixed relative order, so one of them would always trail the same
    other one and inherit its thermal state.
    """
    manifest = run_benchmark(
        [
            RecordingTrainer("cleargbm"),
            RecordingTrainer("cleargbm@leaf_wise"),
            RecordingTrainer("lightgbm"),
        ],
        constant_split,
        [42, 43, 44],
        make_config(repeats=1, warmups=0),
        make_dataset_info(),
        _FINGERPRINT,
    )

    slots: dict[str, list[int]] = {}
    for result in manifest["results"]:
        slots.setdefault(result["model"], []).append(result["position"])

    assert sorted(slots.keys()) == ["cleargbm", "cleargbm@leaf_wise", "lightgbm"]
    for model, positions in slots.items():
        assert sorted(positions) == [0, 1, 2], f"{model} did not occupy every slot"


def test_run_benchmark_requires_at_least_two_arms(stepping_clock: SteppingClock) -> None:
    with pytest.raises(ValueError, match=ERR_TOO_FEW_TRAINERS):
        run_benchmark(
            [RecordingTrainer("cleargbm")],
            constant_split,
            [42],
            make_config(repeats=1, warmups=0),
            make_dataset_info(),
            _FINGERPRINT,
        )


def test_run_benchmark_rejects_two_arms_sharing_a_name(
    stepping_clock: SteppingClock,
) -> None:
    """Two arms under one name merge into a single series, silently."""
    with pytest.raises(ValueError, match=ERR_DUPLICATE_TRAINER):
        run_benchmark(
            [
                RecordingTrainer("cleargbm"),
                RecordingTrainer("cleargbm"),
                RecordingTrainer("lightgbm"),
            ],
            constant_split,
            [42],
            make_config(repeats=1, warmups=0),
            make_dataset_info(),
            _FINGERPRINT,
        )


def test_run_benchmark_records_both_models_at_every_seed(
    stepping_clock: SteppingClock,
) -> None:
    manifest = run_benchmark(
        [RecordingTrainer("cleargbm"), RecordingTrainer("lightgbm")],
        constant_split,
        [42, 43],
        make_config(repeats=1, warmups=0),
        make_dataset_info(),
        _FINGERPRINT,
    )

    assert len(manifest["results"]) == 4
    assert manifest["seeds"] == [42, 43]
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["estimator"] == "median"


def test_run_benchmark_passes_each_seed_to_the_split_factory(
    stepping_clock: SteppingClock,
) -> None:
    seen: list[int] = []

    def build_split(seed: int) -> DataSplit:
        seen.append(seed)
        return make_split()

    run_benchmark(
        [RecordingTrainer("cleargbm"), RecordingTrainer("lightgbm")],
        build_split,
        [7, 8],
        make_config(repeats=1, warmups=0),
        make_dataset_info(),
        _FINGERPRINT,
    )
    assert seen == [7, 8]


def test_no_seeds_raises(stepping_clock: SteppingClock) -> None:
    with pytest.raises(ValueError, match=ERR_NO_SEEDS):
        run_benchmark(
            [RecordingTrainer("cleargbm"), RecordingTrainer("lightgbm")],
            constant_split,
            [],
            make_config(),
            make_dataset_info(),
            _FINGERPRINT,
        )
