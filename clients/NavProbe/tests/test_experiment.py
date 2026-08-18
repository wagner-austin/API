"""Tests for the determinism trial.

The service is driven against real simulator factories, never a stand-in. A
trial's whole job is to reach a verdict about a simulator, so a trial validated
against something that reports whatever the test asked for would establish
nothing about the trial.
"""

from __future__ import annotations

import pytest

from navprobe.canonical import CanonicalEncodingError
from navprobe.comparison import ComparisonError
from navprobe.experiment import (
    MINIMUM_REPETITIONS,
    ProbeService,
    TrialError,
    repetition_label,
)
from navprobe.records import TrialSpec
from navprobe.rollout import RolloutError, roll_out
from tests.factories import (
    DriftingSimulatorFactory,
    EmptyWorldSimulatorFactory,
    LinearSimulatorFactory,
    NaNSimulatorFactory,
)
from tests.simulators import DriftingSimulator, LinearSimulator


def _spec(repetitions: int = 3, step_count: int = 6, seed: int = 7) -> TrialSpec:
    """Build a trial spec.

    Args:
        repetitions: Number of repetitions to compare.
        step_count: Steps per repetition.
        seed: The seed every repetition is pinned to.

    Returns:
        The spec.
    """
    return TrialSpec(seed=seed, step_count=step_count, repetitions=repetitions)


class TestRepetitionLabel:
    """Tests for :func:`repetition_label`."""

    def test_labels_are_derived_from_the_index(self) -> None:
        """The label is a function of position, so trials compare by content."""
        assert [repetition_label(index) for index in range(3)] == [
            "repetition-0",
            "repetition-1",
            "repetition-2",
        ]


class TestRollOutRepetitions:
    """Tests for :meth:`ProbeService.roll_out_repetitions`."""

    def test_produces_one_run_per_repetition(self) -> None:
        """Every repetition asked for appears in the result."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        assert len(service.roll_out_repetitions(_spec(repetitions=4))) == 4

    def test_builds_a_fresh_simulator_for_each_repetition(self) -> None:
        """The factory is called once per repetition, not once per trial.

        Reusing one instance would measure whether ``reset`` restores state,
        which is a weaker question wearing the same name.
        """
        factory = LinearSimulatorFactory(world_count=2)
        ProbeService(factory).roll_out_repetitions(_spec(repetitions=4))
        assert factory.built == 4

    def test_every_run_carries_its_derived_label(self) -> None:
        """Run labels come from the repetition index."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        runs = service.roll_out_repetitions(_spec(repetitions=3))
        assert [run["spec"]["label"] for run in runs] == [
            "repetition-0",
            "repetition-1",
            "repetition-2",
        ]

    def test_every_run_carries_the_trial_seed(self) -> None:
        """One seed throughout is the design, so every run records it."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        runs = service.roll_out_repetitions(_spec(repetitions=3, seed=11))
        assert [run["spec"]["seed"] for run in runs] == [11, 11, 11]

    def test_rejects_a_single_repetition(self) -> None:
        """One rollout has nothing to disagree with."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        with pytest.raises(TrialError) as caught:
            service.roll_out_repetitions(_spec(repetitions=1))
        assert caught.value.code == "NP-TRIAL-001"

    def test_accepts_exactly_the_minimum(self) -> None:
        """The bound is inclusive, so two repetitions run."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        runs = service.roll_out_repetitions(_spec(repetitions=MINIMUM_REPETITIONS))
        assert len(runs) == MINIMUM_REPETITIONS

    def test_propagates_an_unusable_world_count(self) -> None:
        """A simulator reporting no worlds fails the trial rather than passing it."""
        service = ProbeService(EmptyWorldSimulatorFactory(world_count=0))
        with pytest.raises(RolloutError) as caught:
            service.roll_out_repetitions(_spec())
        assert caught.value.code == "NP-ROLLOUT-002"

    def test_propagates_an_unencodable_observation(self) -> None:
        """NaN fails the trial rather than being recorded as divergence."""
        service = ProbeService(NaNSimulatorFactory(world_count=2, nan_at_step=1))
        with pytest.raises(CanonicalEncodingError) as caught:
            service.roll_out_repetitions(_spec())
        assert caught.value.code == "NP-CANON-001"


class TestCompareAgainstReference:
    """Tests for :meth:`ProbeService.compare_against_reference`."""

    def test_produces_one_comparison_per_later_repetition(self) -> None:
        """The reference is not compared against itself."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        runs = service.roll_out_repetitions(_spec(repetitions=4))
        assert len(service.compare_against_reference(runs)) == 3

    def test_every_comparison_names_the_reference_on_the_left(self) -> None:
        """A shared origin is what makes the divergence points comparable."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        runs = service.roll_out_repetitions(_spec(repetitions=3))
        comparisons = service.compare_against_reference(runs)
        assert [comparison["left_label"] for comparison in comparisons] == [
            "repetition-0",
            "repetition-0",
        ]

    def test_rejects_fewer_runs_than_the_minimum(self) -> None:
        """A single run cannot be compared against anything."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        runs = service.roll_out_repetitions(_spec(repetitions=2))
        with pytest.raises(TrialError) as caught:
            service.compare_against_reference(runs[:1])
        assert caught.value.code == "NP-TRIAL-002"


class TestSummarise:
    """Tests for :meth:`ProbeService.summarise`."""

    def test_deterministic_factory_is_reported_as_deterministic(self) -> None:
        """The positive control passes."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        spec = _spec()
        record = service.summarise(spec, service.roll_out_repetitions(spec))
        assert record["deterministic"] is True

    def test_a_deterministic_trial_reports_no_divergence(self) -> None:
        """Agreement is an absence, not a sentinel index."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        spec = _spec()
        record = service.summarise(spec, service.roll_out_repetitions(spec))
        assert record["first_divergent_step"] is None

    def test_carries_the_reference_digest(self) -> None:
        """The digest every repetition was compared against is recorded."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        spec = _spec()
        runs = service.roll_out_repetitions(spec)
        assert service.summarise(spec, runs)["reference_digest"] == runs[0]["digest"]

    def test_carries_the_world_count(self) -> None:
        """Batch width is the variable under test in a sweep, so it is recorded."""
        service = ProbeService(LinearSimulatorFactory(world_count=8))
        spec = _spec()
        record = service.summarise(spec, service.roll_out_repetitions(spec))
        assert record["world_count"] == 8

    def test_carries_the_spec(self) -> None:
        """The record states the design it came from."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        spec = _spec()
        record = service.summarise(spec, service.roll_out_repetitions(spec))
        assert record["spec"] == spec

    def test_reports_the_earliest_divergence_across_repetitions(self) -> None:
        """The earliest departure is reported, not the first one encountered.

        The repetitions are ordered so that the later-diverging one is compared
        first, which is what distinguishes a minimum from a first hit.
        """
        reference = roll_out(
            DriftingSimulator(world_count=2, diverge_at_step=5, offset=0),
            repetition_label(0),
            7,
            6,
        )
        late = roll_out(
            DriftingSimulator(world_count=2, diverge_at_step=4, offset=1),
            repetition_label(1),
            7,
            6,
        )
        early = roll_out(
            DriftingSimulator(world_count=2, diverge_at_step=2, offset=1),
            repetition_label(2),
            7,
            6,
        )
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        record = service.summarise(_spec(), (reference, late, early))
        assert record["first_divergent_step"] == 2

    def test_rejects_a_run_produced_at_another_seed(self) -> None:
        """A summary must not describe runs it did not come from.

        The seed mismatch is caught by the comparison layer first, which is the
        correct order: two runs at different seeds cannot be compared at all.
        """
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        matching = roll_out(LinearSimulator(world_count=2), repetition_label(0), 7, 6)
        other_seed = roll_out(LinearSimulator(world_count=2), repetition_label(1), 9, 6)
        with pytest.raises(ComparisonError) as caught:
            service.summarise(_spec(), (matching, other_seed))
        assert caught.value.code == "NP-COMPARE-001"

    def test_rejects_runs_taken_for_a_different_step_count(self) -> None:
        """A trial declaring six steps cannot be summarised from four-step runs."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        runs = tuple(
            roll_out(LinearSimulator(world_count=2), repetition_label(index), 7, 4)
            for index in range(2)
        )
        with pytest.raises(TrialError) as caught:
            service.summarise(_spec(step_count=6), runs)
        assert caught.value.code == "NP-TRIAL-003"


class TestRunTrial:
    """Tests for :meth:`ProbeService.run_trial`, the composed entry point."""

    def test_deterministic_factory_passes_end_to_end(self) -> None:
        """The positive control produces a clean verdict from the spec alone.

        The expected reference digest is computed from a separately driven
        simulator rather than read back off the record, so this asserts what the
        trial produced rather than that it equals itself.
        """
        expected_digest = roll_out(LinearSimulator(world_count=2), repetition_label(0), 7, 6)[
            "digest"
        ]
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        assert service.run_trial(_spec()) == {
            "spec": _spec(),
            "world_count": 2,
            "reference_digest": expected_digest,
            "deterministic": True,
            "first_divergent_step": None,
        }

    def test_drifting_factory_is_reported_as_non_deterministic(self) -> None:
        """The negative control fails, localised to the step it departed at."""
        service = ProbeService(DriftingSimulatorFactory(world_count=2, diverge_at_step=3))
        record = service.run_trial(_spec())
        assert record["deterministic"] is False
        assert record["first_divergent_step"] == 3

    def test_a_trial_is_reproducible(self) -> None:
        """Two trials of one design against one factory kind agree completely.

        This is the instrument checking itself: if the trial were sensitive to
        anything beyond the spec, these two records would differ.
        """
        first = ProbeService(LinearSimulatorFactory(world_count=2)).run_trial(_spec())
        second = ProbeService(LinearSimulatorFactory(world_count=2)).run_trial(_spec())
        assert first == second

    def test_world_count_changes_the_reference_digest(self) -> None:
        """Batch width is part of the observation, so a sweep sees it."""
        narrow = ProbeService(LinearSimulatorFactory(world_count=1)).run_trial(_spec())
        wide = ProbeService(LinearSimulatorFactory(world_count=4)).run_trial(_spec())
        assert narrow["reference_digest"] != wide["reference_digest"]

    def test_rejects_a_single_repetition(self) -> None:
        """The design check runs before any simulator is built."""
        service = ProbeService(LinearSimulatorFactory(world_count=2))
        with pytest.raises(TrialError) as caught:
            service.run_trial(_spec(repetitions=1))
        assert caught.value.code == "NP-TRIAL-001"

    def test_a_zero_step_trial_is_deterministic_by_construction(self) -> None:
        """Zero steps is the base case: nothing observed, nothing to disagree on."""
        service = ProbeService(DriftingSimulatorFactory(world_count=2, diverge_at_step=0))
        record = service.run_trial(_spec(step_count=0))
        assert record["deterministic"] is True
        assert record["first_divergent_step"] is None
