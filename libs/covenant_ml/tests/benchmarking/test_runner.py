"""Tests for the benchmark measurement protocol.

Collaborators are real fake implementations of the package's Protocols,
injected through ``_test_hooks`` by save-and-restore. Nothing is mocked.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.benchmarking import _test_hooks
from covenant_ml.benchmarking.protocols import DataSplit, TrainedModelProto
from covenant_ml.benchmarking.runner import measure_trainer, run_benchmark
from covenant_ml.benchmarking.types import (
    ERR_INVALID_REPEATS,
    ERR_NO_SEEDS,
    MANIFEST_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkModelName,
    DatasetInfo,
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


def test_warmup_fits_run_but_are_not_timed(stepping_clock: SteppingClock) -> None:
    trainer = RecordingTrainer("cleargbm")
    result = measure_trainer(trainer, make_split(), 42, make_config(repeats=3, warmups=2), True)

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
    result = measure_trainer(trainer, make_split(), 1, make_config(repeats=2, warmups=0), True)
    _test_hooks.monotonic_clock = previous

    assert result["timing"]["samples_s"] == [1.0, 3.0]
    assert result["timing"]["canonical_s"] == 2.0


def test_result_carries_model_seed_and_order(stepping_clock: SteppingClock) -> None:
    trainer = RecordingTrainer("lightgbm", mean_leaves=31.0)
    result = measure_trainer(trainer, make_split(), 44, make_config(), ran_first=False)

    assert result["model"] == "lightgbm"
    assert result["seed"] == 44
    assert result["ran_first"] is False
    assert result["mean_leaves"] == 31.0


def test_every_fit_uses_the_requested_seed(stepping_clock: SteppingClock) -> None:
    trainer = RecordingTrainer("cleargbm")
    measure_trainer(trainer, make_split(), 99, make_config(repeats=2, warmups=1), True)
    assert trainer.fit_seeds == [99, 99, 99]


def test_zero_repeats_raises(stepping_clock: SteppingClock) -> None:
    trainer = RecordingTrainer("cleargbm")
    with pytest.raises(ValueError, match=ERR_INVALID_REPEATS):
        measure_trainer(trainer, make_split(), 1, make_config(repeats=0), True)


def test_run_benchmark_alternates_which_model_goes_first(
    stepping_clock: SteppingClock,
) -> None:
    """Whichever model runs first gets the coolest CPU, so order must rotate."""
    cleargbm = RecordingTrainer("cleargbm")
    lightgbm = RecordingTrainer("lightgbm")
    manifest = run_benchmark(
        cleargbm,
        lightgbm,
        constant_split,
        [42, 43, 44],
        make_config(repeats=1, warmups=0),
        make_dataset_info(),
    )

    first_by_seed = {
        result["seed"]: result["model"] for result in manifest["results"] if result["ran_first"]
    }
    assert first_by_seed == {42: "cleargbm", 43: "lightgbm", 44: "cleargbm"}


def test_run_benchmark_records_both_models_at_every_seed(
    stepping_clock: SteppingClock,
) -> None:
    manifest = run_benchmark(
        RecordingTrainer("cleargbm"),
        RecordingTrainer("lightgbm"),
        constant_split,
        [42, 43],
        make_config(repeats=1, warmups=0),
        make_dataset_info(),
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
        RecordingTrainer("cleargbm"),
        RecordingTrainer("lightgbm"),
        build_split,
        [7, 8],
        make_config(repeats=1, warmups=0),
        make_dataset_info(),
    )
    assert seen == [7, 8]


def test_no_seeds_raises(stepping_clock: SteppingClock) -> None:
    with pytest.raises(ValueError, match=ERR_NO_SEEDS):
        run_benchmark(
            RecordingTrainer("cleargbm"),
            RecordingTrainer("lightgbm"),
            constant_split,
            [],
            make_config(),
            make_dataset_info(),
        )
