"""Tests for driving a simulator to a run record."""

from __future__ import annotations

import pytest

from navprobe.canonical import CanonicalEncodingError
from navprobe.digest import digest_run, digest_step
from navprobe.records import StepRecord
from navprobe.rollout import RolloutError, roll_out
from tests.simulators import (
    DriftingSimulator,
    EmptyWorldSimulator,
    LinearSimulator,
    NaNSimulator,
)


class TestRollOutShape:
    """The structure of a produced run record."""

    def test_records_the_requested_step_count(self) -> None:
        """Every requested step appears in the record."""
        record = roll_out(LinearSimulator(world_count=2), "same-process", 7, 5)
        assert len(record["steps"]) == 5

    def test_step_indices_are_contiguous_and_ordered(self) -> None:
        """Steps carry their own position, in order from zero."""
        record = roll_out(LinearSimulator(world_count=2), "same-process", 7, 4)
        assert [step["step_index"] for step in record["steps"]] == [0, 1, 2, 3]

    def test_spec_carries_the_condition(self) -> None:
        """The label, seed, step count, and world count are all recorded."""
        record = roll_out(LinearSimulator(world_count=3), "fresh-process", 11, 2)
        assert record["spec"] == {
            "label": "fresh-process",
            "seed": 11,
            "step_count": 2,
            "world_count": 3,
        }

    def test_steps_are_an_immutable_tuple(self) -> None:
        """A later stage cannot append to a rollout.

        Compared against a tuple by value: a list of the same steps compares
        unequal to a tuple in Python, so this asserts the container type as
        well as the contents.
        """
        record = roll_out(LinearSimulator(world_count=1), "same-process", 1, 2)
        assert record["steps"] == (
            StepRecord(step_index=0, digest=digest_step(0, [1000.0])),
            StepRecord(step_index=1, digest=digest_step(1, [1010.0])),
        )

    def test_run_digest_folds_the_recorded_steps(self) -> None:
        """The run digest is derived from exactly the steps stored."""
        record = roll_out(LinearSimulator(world_count=2), "same-process", 7, 3)
        assert record["digest"] == digest_run([step["digest"] for step in record["steps"]])

    def test_step_digests_match_the_observations_emitted(self) -> None:
        """Each step digest is the digest of that step's observation.

        Computed here from a second, independently driven simulator rather
        than from values the test invented, so this asserts the rollout
        digested what the simulator produced.
        """
        expected_source = LinearSimulator(world_count=2)
        expected_source.reset(7)
        expected = [digest_step(index, expected_source.advance()) for index in range(3)]
        record = roll_out(LinearSimulator(world_count=2), "same-process", 7, 3)
        assert [step["digest"] for step in record["steps"]] == expected

    def test_zero_steps_produces_an_empty_record(self) -> None:
        """A zero-step rollout is the base case, not an error."""
        record = roll_out(LinearSimulator(world_count=1), "same-process", 1, 0)
        assert record["steps"] == ()

    def test_zero_step_run_digest_is_the_empty_fold(self) -> None:
        """The empty run still carries a well-defined digest."""
        record = roll_out(LinearSimulator(world_count=1), "same-process", 1, 0)
        assert record["digest"] == digest_run([])


class TestRollOutDeterminism:
    """What the instrument reports about real simulators."""

    def test_deterministic_simulator_reproduces_its_run_digest(self) -> None:
        """The positive control agrees with itself across rollouts."""
        first = roll_out(LinearSimulator(world_count=2), "run-a", 7, 6)
        second = roll_out(LinearSimulator(world_count=2), "run-b", 7, 6)
        assert first["digest"] == second["digest"]

    def test_seed_changes_the_run_digest(self) -> None:
        """A different seed produces a different rollout."""
        first = roll_out(LinearSimulator(world_count=2), "run-a", 7, 4)
        second = roll_out(LinearSimulator(world_count=2), "run-b", 8, 4)
        assert first["digest"] != second["digest"]

    def test_world_count_changes_the_run_digest(self) -> None:
        """Batching width is part of the observation, so it is part of the digest."""
        narrow = roll_out(LinearSimulator(world_count=1), "run-a", 7, 4)
        wide = roll_out(LinearSimulator(world_count=4), "run-b", 7, 4)
        assert narrow["digest"] != wide["digest"]

    def test_matching_offsets_reproduce_the_run_digest(self) -> None:
        """Two identically offset instances agree, so the offset is the variable."""
        first = roll_out(DriftingSimulator(world_count=2, diverge_at_step=3, offset=0), "a", 7, 6)
        second = roll_out(DriftingSimulator(world_count=2, diverge_at_step=3, offset=0), "b", 7, 6)
        assert first["digest"] == second["digest"]

    def test_drifting_simulator_is_reported_as_divergent(self) -> None:
        """The negative control disagrees with a differently offset peer."""
        first = roll_out(DriftingSimulator(world_count=2, diverge_at_step=3, offset=0), "a", 7, 6)
        second = roll_out(DriftingSimulator(world_count=2, diverge_at_step=3, offset=1), "b", 7, 6)
        assert first["digest"] != second["digest"]

    def test_drift_leaves_the_shared_prefix_intact(self) -> None:
        """Steps before the divergence still agree, which localises the fault."""
        first = roll_out(DriftingSimulator(world_count=2, diverge_at_step=3, offset=0), "a", 7, 6)
        second = roll_out(DriftingSimulator(world_count=2, diverge_at_step=3, offset=1), "b", 7, 6)
        shared = [
            left["digest"] == right["digest"]
            for left, right in zip(first["steps"], second["steps"], strict=True)
        ]
        assert shared == [True, True, True, False, False, False]


class TestRollOutRejections:
    """Inputs the rollout refuses."""

    def test_rejects_a_negative_step_count(self) -> None:
        """A negative step count is refused by its own code."""
        with pytest.raises(RolloutError) as caught:
            roll_out(LinearSimulator(world_count=1), "same-process", 1, -1)
        assert caught.value.code == "NP-ROLLOUT-001"

    def test_rejects_a_simulator_with_no_worlds(self) -> None:
        """Zero worlds would report determinism over zero evidence."""
        with pytest.raises(RolloutError) as caught:
            roll_out(EmptyWorldSimulator(world_count=0), "same-process", 1, 3)
        assert caught.value.code == "NP-ROLLOUT-002"

    def test_propagates_nan_from_an_observation(self) -> None:
        """A NaN observation fails the rollout rather than digesting."""
        with pytest.raises(CanonicalEncodingError) as caught:
            roll_out(NaNSimulator(world_count=2, nan_at_step=1), "same-process", 1, 3)
        assert caught.value.code == "NP-CANON-001"
