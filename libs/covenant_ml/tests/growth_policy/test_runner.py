"""Tests for the growth-policy measurement protocol.

Collaborators are real implementations of the package's Protocols, and the
clock is injected through ``_test_hooks`` by save-and-restore. Nothing is
mocked: the trainers here genuinely fit and the models genuinely predict, they
are simply small and deterministic so the orchestration can be asserted
exactly. The real learners are exercised in ``test_trainers.py``.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.growth_policy import _test_hooks
from covenant_ml.growth_policy.protocols import (
    TrainedModelProto,
    TwoWaySplit,
)
from covenant_ml.growth_policy.runner import fit_repeatedly, measure_arm, run_experiment
from covenant_ml.growth_policy.types import (
    ERR_INVALID_REPEATS,
    ERR_NO_ARMS,
    ERR_NO_SEEDS,
    REPORT_SCHEMA_VERSION,
)

from .factories import make_config, make_dataset_info
from .numeric import label_mask, mean_of, positive_rate, select


class SteppingClock:
    """A clock advancing by a fixed step on every reading.

    The runner reads twice per timed fit, so a constant step makes each
    measured duration exactly ``step``.
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
            The value before advancing.
        """
        self.reads += 1
        current = self._now
        self._now += self._step
        return current


class ScriptedClock:
    """A clock returning a fixed sequence of readings."""

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


class ThresholdModel:
    """A model that genuinely scores rows against a threshold."""

    def __init__(self, threshold: float, leaves: float, generation: int) -> None:
        """Bind the decision threshold and reported shape.

        Args:
            threshold: Rows whose first feature exceeds this score high.
            leaves: Mean leaves per tree to report.
            generation: Which fit produced this model, so a test can tell the
                last timed fit from an earlier one.
        """
        self._threshold = threshold
        self._leaves = leaves
        self.generation = generation

    def predict_positive_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score every row against the threshold.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            0.9 for rows above the threshold, 0.1 otherwise.
        """
        column: NDArray[np.float64] = x[:, 0]
        above: NDArray[np.bool_] = column > self._threshold
        high: NDArray[np.float64] = np.full(len(above), 0.9, dtype=np.float64)
        low: NDArray[np.float64] = np.full(len(above), 0.1, dtype=np.float64)
        scored: NDArray[np.float64] = np.where(above, high, low)
        return scored

    def mean_leaves(self) -> float:
        """Report the configured leaf count.

        Returns:
            Mean leaves per tree.
        """
        return self._leaves


class CountingTrainer:
    """A trainer that fits a real threshold model and counts its fits."""

    def __init__(self, name: str, leaves: float = 10.0) -> None:
        """Bind the arm's identity.

        Args:
            name: Arm name to report.
            leaves: Leaf count its fitted models report.
        """
        self._name = name
        self._leaves = leaves
        self.fit_seeds: list[int] = []
        self.fitted: list[TrainedModelProto] = []

    @property
    def arm_name(self) -> str:
        """Return the configured arm name.

        Returns:
            The arm name.
        """
        return self._name

    def fit(self, split: TwoWaySplit, seed: int) -> TrainedModelProto:
        """Fit a threshold model on the training fold's mean.

        Args:
            split: The partition to fit on.
            seed: Recorded so a test can assert the seeds reached the arm.

        Returns:
            The fitted model.
        """
        self.fit_seeds.append(seed)
        column: NDArray[np.float64] = split.x_train[:, 0]
        threshold = mean_of(column)
        model = ThresholdModel(threshold, self._leaves, len(self.fit_seeds))
        self.fitted.append(model)
        return model


class CountingMetrics:
    """Metrics that genuinely score and record how often they were asked."""

    def __init__(self) -> None:
        """Start the call counters at zero."""
        self.calls = 0

    def auc_roc(self, y_true: NDArray[np.int64], positive_proba: NDArray[np.float64]) -> float:
        """Return the mean predicted probability among positives.

        Args:
            y_true: True labels.
            positive_proba: Predicted probabilities.

        Returns:
            A real statistic over the inputs.
        """
        self.calls += 1
        return mean_of(select(positive_proba, label_mask(y_true, 1)))

    def auc_pr(self, y_true: NDArray[np.int64], positive_proba: NDArray[np.float64]) -> float:
        """Return the positive rate.

        Args:
            y_true: True labels.
            positive_proba: Predicted probabilities.

        Returns:
            A real statistic over the inputs.
        """
        self.calls += 1
        return positive_rate(y_true)

    def log_loss(self, y_true: NDArray[np.int64], positive_proba: NDArray[np.float64]) -> float:
        """Return the mean absolute error.

        Args:
            y_true: True labels.
            positive_proba: Predicted probabilities.

        Returns:
            A real statistic over the inputs.
        """
        self.calls += 1
        difference: NDArray[np.float64] = np.abs(positive_proba - y_true.astype(np.float64))
        return mean_of(difference)


class SeedRecordingSplitFactory:
    """A split factory that builds a real partition and records its seeds."""

    def __init__(self, split: TwoWaySplit) -> None:
        """Bind the partition to return.

        Args:
            split: The partition every seed receives.
        """
        self._split = split
        self.seeds: list[int] = []

    def __call__(self, seed: int) -> TwoWaySplit:
        """Record the seed and return the partition.

        Args:
            seed: Seed requested.

        Returns:
            The bound partition.
        """
        self.seeds.append(seed)
        return self._split


def make_split(row_count: int = 8) -> TwoWaySplit:
    """Build a small real partition.

    Args:
        row_count: Rows in each fold.

    Returns:
        The partition.
    """
    features: NDArray[np.float64] = np.arange(row_count * 2, dtype=np.float64).reshape(row_count, 2)
    labels: NDArray[np.int64] = (np.arange(row_count, dtype=np.int64) % 2).astype(np.int64)
    return TwoWaySplit(x_train=features, y_train=labels, x_test=features, y_test=labels)


@pytest.fixture()
def stepping_clock() -> Generator[SteppingClock, None, None]:
    """Bind a stepping clock for the duration of one test.

    Yields:
        The installed clock.
    """
    previous = _test_hooks.monotonic_clock
    clock = SteppingClock(step=0.5)
    _test_hooks.monotonic_clock = clock
    try:
        yield clock
    finally:
        _test_hooks.monotonic_clock = previous


@pytest.fixture()
def scripted_clock_factory() -> Generator[list[ScriptedClock], None, None]:
    """Allow a test to install a scripted clock and have it restored.

    Yields:
        A list the test appends its installed clock to.
    """
    previous = _test_hooks.monotonic_clock
    installed: list[ScriptedClock] = []
    try:
        yield installed
    finally:
        _test_hooks.monotonic_clock = previous


class TestFitRepeatedly:
    """Warmups, repeats, timing and which model is returned."""

    def test_runs_warmups_then_repeats(self, stepping_clock: SteppingClock) -> None:
        """Total fits should be warmups plus repeats."""
        trainer = CountingTrainer("arm-a")

        fit_repeatedly(trainer, make_split(), seed=42, repeats=3, warmups=2)

        assert len(trainer.fit_seeds) == 5

    def test_times_only_the_repeats(self, stepping_clock: SteppingClock) -> None:
        """The clock should be read twice per timed fit and not for warmups."""
        trainer = CountingTrainer("arm-a")

        fit_repeatedly(trainer, make_split(), seed=42, repeats=3, warmups=2)

        assert stepping_clock.reads == 6

    def test_reports_the_median_duration(self, scripted_clock_factory: list[ScriptedClock]) -> None:
        """The canonical time should be the median, not the mean or minimum."""
        clock = ScriptedClock([0.0, 1.0, 10.0, 12.0, 20.0, 23.0])
        scripted_clock_factory.append(clock)
        _test_hooks.monotonic_clock = clock
        trainer = CountingTrainer("arm-a")

        _, seconds = fit_repeatedly(trainer, make_split(), seed=42, repeats=3, warmups=0)

        assert seconds == 2.0

    def test_returns_the_model_from_the_last_timed_fit(self, stepping_clock: SteppingClock) -> None:
        """The scored model must be one that was actually measured."""
        trainer = CountingTrainer("arm-a")

        model, _ = fit_repeatedly(trainer, make_split(), seed=42, repeats=3, warmups=1)

        assert model is trainer.fitted[-1]
        assert len(trainer.fitted) == 4

    def test_rejects_zero_repeats(self, stepping_clock: SteppingClock) -> None:
        """No timed fit leaves nothing to summarise and no model to score."""
        with pytest.raises(ValueError, match=ERR_INVALID_REPEATS):
            fit_repeatedly(CountingTrainer("arm-a"), make_split(), seed=42, repeats=0, warmups=1)


class TestMeasureArm:
    """One arm at one seed."""

    def test_records_every_field(self, stepping_clock: SteppingClock) -> None:
        """The result should carry the arm, the seed, the timing and the shape."""
        trainer = CountingTrainer("arm-a", leaves=7.5)

        result = measure_arm(trainer, make_split(), 42, CountingMetrics(), make_config())

        assert result["arm"] == "arm-a"
        assert result["seed"] == 42
        assert result["fit_seconds"] == 0.5
        assert result["mean_leaves"] == 7.5

    def test_scores_the_held_out_fold(self, stepping_clock: SteppingClock) -> None:
        """All three metrics should be asked exactly once."""
        metrics = CountingMetrics()

        measure_arm(CountingTrainer("arm-a"), make_split(), 42, metrics, make_config())

        assert metrics.calls == 3

    def test_honours_the_configured_repeat_counts(self, stepping_clock: SteppingClock) -> None:
        """Repeats and warmups should come from the configuration."""
        trainer = CountingTrainer("arm-a")

        measure_arm(
            trainer,
            make_split(),
            42,
            CountingMetrics(),
            make_config(repeats=2, warmups=3),
        )

        assert len(trainer.fit_seeds) == 5


class TestRunExperiment:
    """Every arm at every seed."""

    def test_measures_each_arm_at_each_seed(self, stepping_clock: SteppingClock) -> None:
        """Two arms across three seeds should produce six results."""
        arms = [CountingTrainer("arm-a"), CountingTrainer("arm-b")]

        report = run_experiment(
            arms,
            SeedRecordingSplitFactory(make_split()),
            [42, 43, 44],
            CountingMetrics(),
            make_config(),
            make_dataset_info(),
        )

        assert len(report["results"]) == 6
        assert arms[0].fit_seeds == [42, 43, 44]

    def test_builds_the_partition_once_per_seed(self, stepping_clock: SteppingClock) -> None:
        """Every arm at one seed must see the identical partition."""
        factory = SeedRecordingSplitFactory(make_split())

        run_experiment(
            [CountingTrainer("arm-a"), CountingTrainer("arm-b")],
            factory,
            [42, 43],
            CountingMetrics(),
            make_config(),
            make_dataset_info(),
        )

        assert factory.seeds == [42, 43]

    def test_summarises_each_arm(self, stepping_clock: SteppingClock) -> None:
        """The report should carry one summary per arm, in arm order."""
        report = run_experiment(
            [CountingTrainer("arm-a"), CountingTrainer("arm-b")],
            SeedRecordingSplitFactory(make_split()),
            [42, 43],
            CountingMetrics(),
            make_config(),
            make_dataset_info(),
        )

        assert [summary["arm"] for summary in report["summaries"]] == [
            "arm-a",
            "arm-b",
        ]
        assert [summary["seed_count"] for summary in report["summaries"]] == [2, 2]

    def test_stamps_the_schema_version_and_inputs(self, stepping_clock: SteppingClock) -> None:
        """The report should carry its own provenance."""
        config = make_config()
        info = make_dataset_info("bank")

        report = run_experiment(
            [CountingTrainer("arm-a")],
            SeedRecordingSplitFactory(make_split()),
            [42],
            CountingMetrics(),
            config,
            info,
        )

        assert report["schema_version"] == REPORT_SCHEMA_VERSION
        assert report["config"] == config
        assert report["dataset"] == info
        assert report["seeds"] == [42]

    def test_rejects_no_arms(self, stepping_clock: SteppingClock) -> None:
        """A report over zero arms would state nothing."""
        with pytest.raises(ValueError, match=ERR_NO_ARMS):
            run_experiment(
                [],
                SeedRecordingSplitFactory(make_split()),
                [42],
                CountingMetrics(),
                make_config(),
                make_dataset_info(),
            )

    def test_rejects_no_seeds(self, stepping_clock: SteppingClock) -> None:
        """A report over zero seeds would state nothing."""
        with pytest.raises(ValueError, match=ERR_NO_SEEDS):
            run_experiment(
                [CountingTrainer("arm-a")],
                SeedRecordingSplitFactory(make_split()),
                [],
                CountingMetrics(),
                make_config(),
                make_dataset_info(),
            )


class TestClockHookIsRestored:
    """The production clock must survive the fixtures."""

    def test_default_clock_is_a_real_monotonic_counter(self) -> None:
        """Outside a fixture the hook should be the production clock."""
        first = _test_hooks.monotonic_clock()
        second = _test_hooks.monotonic_clock()

        assert second >= first
