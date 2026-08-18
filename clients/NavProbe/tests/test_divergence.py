"""Tests for comparing two different configurations.

Driven against real simulators throughout. The whole question is whether two
configurations agree with each other, and a stand-in on either side would be
answering about the stand-in.
"""

from __future__ import annotations

import pytest

from navprobe.canonical import CanonicalEncodingError
from navprobe.divergence import (
    DivergenceError,
    compare_observations,
    measure_divergence,
)
from navprobe.records import DivergenceRecord
from tests.factories import (
    DriftingSimulatorFactory,
    EmptyWorldSimulatorFactory,
    LinearSimulatorFactory,
    NaNSimulatorFactory,
    WideningSimulatorFactory,
)


class TestCompareObservations:
    """Tests for :func:`compare_observations`."""

    def test_identical_observations_report_no_difference(self) -> None:
        """Agreement is reported as zeros throughout, not as an absence."""
        assert compare_observations([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == DivergenceRecord(
            observation_length=3,
            differing_elements=0,
            max_absolute_difference=0.0,
            mean_absolute_difference=0.0,
        )

    def test_counts_only_the_elements_that_differ(self) -> None:
        """The count distinguishes a localised artefact from a systematic shift."""
        record = compare_observations([1.0, 2.0, 3.0, 4.0], [1.0, 2.5, 3.0, 4.5])
        assert record["differing_elements"] == 2

    def test_reports_the_largest_absolute_difference(self) -> None:
        """The maximum is over every element, differing or not."""
        record = compare_observations([1.0, 2.0, 3.0], [1.0, 2.5, 3.0])
        assert record["max_absolute_difference"] == 0.5

    def test_the_mean_is_over_differing_elements_only(self) -> None:
        """Averaging over agreeing elements would dilute the figure to nothing.

        Two elements differ by 0.5 and 1.5 among four. The mean over the
        differing pair is 1.0; over all four it would be 0.5, which describes no
        element and shrinks as the observation grows.
        """
        record = compare_observations([1.0, 2.0, 3.0, 4.0], [1.0, 2.5, 3.0, 5.5])
        assert record["mean_absolute_difference"] == 1.0

    def test_difference_is_absolute_not_signed(self) -> None:
        """Opposite-signed differences do not cancel."""
        record = compare_observations([0.0, 0.0], [1.0, -1.0])
        assert record["mean_absolute_difference"] == 1.0

    def test_rejects_observations_of_different_lengths(self) -> None:
        """Different shapes cannot be lined up element-wise."""
        with pytest.raises(DivergenceError) as caught:
            compare_observations([1.0, 2.0], [1.0])
        assert caught.value.code == "NP-DIVERGE-001"

    def test_rejects_empty_observations(self) -> None:
        """A comparison over nothing would report agreement over no evidence."""
        with pytest.raises(DivergenceError) as caught:
            compare_observations([], [])
        assert caught.value.code == "NP-DIVERGE-002"


class TestMeasureDivergence:
    """Tests for :func:`measure_divergence`."""

    def test_two_identical_configurations_agree_exactly(self) -> None:
        """The positive control: same configuration twice, zero difference."""
        record = measure_divergence(
            LinearSimulatorFactory(world_count=2), LinearSimulatorFactory(world_count=2), 7, 5
        )
        assert record["differing_elements"] == 0
        assert record["max_absolute_difference"] == 0.0

    def test_records_the_observation_length(self) -> None:
        """Width is carried, so a differing count can be read as a fraction."""
        record = measure_divergence(
            LinearSimulatorFactory(world_count=3), LinearSimulatorFactory(world_count=3), 7, 5
        )
        assert record["observation_length"] == 3

    def test_two_different_configurations_diverge_by_a_computable_amount(self) -> None:
        """The negative control: two different simulators, one seed.

        The linear simulator folds the seed into its observation and the
        drifting one does not, so at seed 7 after five steps they sit exactly
        7000 apart. Asserting the computed value rather than "nonzero" pins that
        the comparison is element-wise and unsigned.
        """
        record = measure_divergence(
            LinearSimulatorFactory(world_count=1),
            DriftingSimulatorFactory(world_count=1, diverge_at_step=0),
            7,
            5,
        )
        assert record["max_absolute_difference"] == 7000.0

    def test_every_element_can_differ(self) -> None:
        """A systematic difference shows up as every element differing."""
        record = measure_divergence(
            LinearSimulatorFactory(world_count=3),
            DriftingSimulatorFactory(world_count=3, diverge_at_step=0),
            7,
            5,
        )
        assert record["differing_elements"] == record["observation_length"] == 3

    def test_the_order_of_the_two_sides_does_not_change_the_magnitude(self) -> None:
        """Differences are absolute, so swapping the arguments changes nothing."""
        left = measure_divergence(
            LinearSimulatorFactory(world_count=2),
            DriftingSimulatorFactory(world_count=2, diverge_at_step=0),
            7,
            5,
        )
        right = measure_divergence(
            DriftingSimulatorFactory(world_count=2, diverge_at_step=0),
            LinearSimulatorFactory(world_count=2),
            7,
            5,
        )
        assert left == right

    def test_rejects_configurations_of_different_widths(self) -> None:
        """Two configurations observing different things are not comparable."""
        with pytest.raises(DivergenceError) as caught:
            measure_divergence(
                LinearSimulatorFactory(world_count=2),
                LinearSimulatorFactory(world_count=3),
                7,
                5,
            )
        assert caught.value.code == "NP-DIVERGE-001"

    def test_rejects_a_configuration_that_observes_nothing(self) -> None:
        """An empty observation is refused rather than compared."""
        with pytest.raises(DivergenceError) as caught:
            measure_divergence(
                EmptyWorldSimulatorFactory(world_count=0),
                EmptyWorldSimulatorFactory(world_count=0),
                7,
                5,
            )
        assert caught.value.code == "NP-DIVERGE-002"

    def test_rejects_an_observation_containing_nan(self) -> None:
        """NaN fails the comparison rather than producing a NaN magnitude.

        A NaN difference compares false against every threshold, so an
        unguarded comparison would answer "below tolerance" with yes.
        """
        with pytest.raises(CanonicalEncodingError) as caught:
            measure_divergence(
                NaNSimulatorFactory(world_count=2, nan_at_step=2),
                LinearSimulatorFactory(world_count=2),
                7,
                3,
            )
        assert caught.value.code == "NP-CANON-001"

    def test_propagates_a_widening_factory(self) -> None:
        """A factory whose builds differ in width fails against a fixed one."""
        with pytest.raises(DivergenceError) as caught:
            measure_divergence(
                WideningSimulatorFactory(first_world_count=2),
                LinearSimulatorFactory(world_count=5),
                7,
                5,
            )
        assert caught.value.code == "NP-DIVERGE-001"
