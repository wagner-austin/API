"""Tests for measuring how far apart repeated rollouts end up.

Driven against real simulators. A dispersion measurement whose spread came from
a stand-in would report the stand-in's spread, and the whole reason this layer
exists is that bit-equality does not say how much things differ.
"""

from __future__ import annotations

import pytest

from navprobe.canonical import CanonicalEncodingError
from navprobe.dispersion import DispersionError, final_observation, measure_dispersion
from navprobe.experiment import MINIMUM_REPETITIONS
from tests.factories import (
    DriftingSimulatorFactory,
    EmptyWorldSimulatorFactory,
    LinearSimulatorFactory,
    NaNSimulatorFactory,
    WideningSimulatorFactory,
)
from tests.simulators import LinearSimulator


class TestFinalObservation:
    """Tests for :func:`final_observation`."""

    def test_returns_the_last_step_not_the_first(self) -> None:
        """The observation is the one after the final step.

        Compared against a single-step rollout of the same simulator, so this
        asserts which step was returned rather than merely that one was.
        """
        one_step = final_observation(LinearSimulator(world_count=1), 1, 1)
        three_steps = final_observation(LinearSimulator(world_count=1), 1, 3)
        assert list(three_steps) != list(one_step)

    def test_is_reproducible_at_one_seed(self) -> None:
        """Two drives of the same seed end in the same place."""
        left = final_observation(LinearSimulator(world_count=2), 7, 4)
        right = final_observation(LinearSimulator(world_count=2), 7, 4)
        assert list(left) == list(right)

    def test_rejects_a_rollout_with_no_steps(self) -> None:
        """Zero steps produces no observation to disperse."""
        with pytest.raises(DispersionError) as caught:
            final_observation(LinearSimulator(world_count=1), 1, 0)
        assert caught.value.code == "NP-DISP-001"


class TestMeasureDispersion:
    """Tests for :func:`measure_dispersion`."""

    def test_a_deterministic_factory_disperses_by_exactly_zero(self) -> None:
        """The positive control has no spread at all.

        Exact zero rather than a tolerance: a deterministic simulator's
        repetitions are bit-identical, so any non-zero spread would mean the
        measurement itself introduced one.
        """
        record = measure_dispersion(LinearSimulatorFactory(world_count=2), 7, 5, 4)
        assert record["max_spread"] == 0.0
        assert record["mean_spread"] == 0.0

    def test_records_the_repetition_count(self) -> None:
        """The record states how many rollouts it summarises."""
        record = measure_dispersion(LinearSimulatorFactory(world_count=2), 7, 5, 4)
        assert record["repetitions"] == 4

    def test_records_the_observation_length(self) -> None:
        """Observation width is carried, so a spread can be read in context."""
        record = measure_dispersion(LinearSimulatorFactory(world_count=3), 7, 5, 2)
        assert record["observation_length"] == 3

    def test_a_diverging_factory_disperses_by_its_offset(self) -> None:
        """The negative control's spread is the offset it was built to have.

        The drifting factory offsets each simulator by its build index, so four
        repetitions span offsets zero to three and the widest element-wise range
        is exactly three.
        """
        factory = DriftingSimulatorFactory(world_count=1, diverge_at_step=0)
        record = measure_dispersion(factory, 7, 5, 4)
        assert record["max_spread"] == 3.0

    def test_mean_spread_averages_over_every_element(self) -> None:
        """The mean is over observation elements, not over repetitions.

        Every element of this observation diverges by the same amount, so the
        mean equals the maximum — which pins that the divisor is the element
        count rather than something else that happens to be four.
        """
        factory = DriftingSimulatorFactory(world_count=3, diverge_at_step=0)
        record = measure_dispersion(factory, 7, 5, 4)
        assert record["mean_spread"] == record["max_spread"] == 3.0

    def test_rejects_a_single_repetition(self) -> None:
        """One rollout has nothing to be spread against."""
        with pytest.raises(DispersionError) as caught:
            measure_dispersion(LinearSimulatorFactory(world_count=2), 7, 5, 1)
        assert caught.value.code == "NP-DISP-002"

    def test_accepts_exactly_the_minimum(self) -> None:
        """The bound is inclusive."""
        record = measure_dispersion(
            LinearSimulatorFactory(world_count=2), 7, 5, MINIMUM_REPETITIONS
        )
        assert record["repetitions"] == MINIMUM_REPETITIONS

    def test_rejects_an_empty_observation(self) -> None:
        """A simulator that observes nothing would report zero over no evidence."""
        with pytest.raises(DispersionError) as caught:
            measure_dispersion(EmptyWorldSimulatorFactory(world_count=0), 7, 5, 2)
        assert caught.value.code == "NP-DISP-004"

    def test_rejects_observations_of_different_widths(self) -> None:
        """Repetitions of different shapes cannot be compared element-wise.

        Comparing them anyway would line up position three of one rollout
        against position three of a differently shaped one and report the
        difference as a spread.
        """
        with pytest.raises(DispersionError) as caught:
            measure_dispersion(WideningSimulatorFactory(first_world_count=2), 7, 5, 3)
        assert caught.value.code == "NP-DISP-003"

    def test_names_the_repetition_whose_width_differed(self) -> None:
        """The message identifies which repetition broke the comparison."""
        with pytest.raises(DispersionError) as caught:
            measure_dispersion(WideningSimulatorFactory(first_world_count=2), 7, 5, 3)
        assert "repetition 1 produced 3 values but repetition 0 produced 2" in (
            caught.value.message
        )

    def test_builds_a_fresh_simulator_per_repetition(self) -> None:
        """Each repetition starts from a newly constructed simulator."""
        factory = LinearSimulatorFactory(world_count=2)
        measure_dispersion(factory, 7, 5, 3)
        assert factory.built == 3

    def test_rejects_an_observation_containing_nan(self) -> None:
        """NaN fails the measurement rather than dispersing to NaN.

        This is the sharpest reason the check exists. NaN propagates silently
        through ``max`` and ``min``, so an unguarded measurement returns a NaN
        spread — which compares false against every threshold, meaning a caller
        asking "is the spread below tolerance" is told yes.
        """
        factory = NaNSimulatorFactory(world_count=2, nan_at_step=2)
        with pytest.raises(CanonicalEncodingError) as caught:
            measure_dispersion(factory, 7, 3, 2)
        assert caught.value.code == "NP-CANON-001"
