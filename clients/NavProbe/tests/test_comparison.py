"""Tests for the determinism verdict."""

from __future__ import annotations

import pytest

from navprobe.comparison import ComparisonError, compare_runs, find_first_divergent_step
from navprobe.records import RunRecord
from navprobe.rollout import roll_out
from tests.simulators import DriftingSimulator, LinearSimulator


class TestFindFirstDivergentStep:
    """Tests for :func:`find_first_divergent_step`."""

    def test_returns_none_when_every_step_agrees(self) -> None:
        """Agreement is reported as an absence, not a sentinel index."""
        left = roll_out(LinearSimulator(world_count=2), "run-a", 7, 5)
        right = roll_out(LinearSimulator(world_count=2), "run-b", 7, 5)
        assert find_first_divergent_step(left, right) is None

    def test_returns_the_earliest_differing_index(self) -> None:
        """The first divergence is reported, not the last."""
        left = roll_out(DriftingSimulator(world_count=2, diverge_at_step=3, offset=0), "a", 7, 6)
        right = roll_out(DriftingSimulator(world_count=2, diverge_at_step=3, offset=1), "b", 7, 6)
        assert find_first_divergent_step(left, right) == 3

    def test_compares_only_the_common_prefix(self) -> None:
        """A longer run that agrees throughout the prefix reports no divergence."""
        left = roll_out(LinearSimulator(world_count=2), "run-a", 7, 3)
        right = roll_out(LinearSimulator(world_count=2), "run-b", 7, 6)
        assert find_first_divergent_step(left, right) is None


class TestCompareRuns:
    """Tests for :func:`compare_runs`."""

    def test_agreeing_runs_report_matching_digests(self) -> None:
        """The positive control is reported as deterministic."""
        left = roll_out(LinearSimulator(world_count=2), "same-process", 7, 5)
        right = roll_out(LinearSimulator(world_count=2), "fresh-process", 7, 5)
        assert compare_runs(left, right) == {
            "left_label": "same-process",
            "right_label": "fresh-process",
            "digests_match": True,
            "first_divergent_step": None,
            "compared_step_count": 5,
        }

    def test_diverging_runs_report_the_divergence_point(self) -> None:
        """The negative control is reported with its divergence localised."""
        left = roll_out(
            DriftingSimulator(world_count=2, diverge_at_step=2, offset=0), "run-a", 7, 5
        )
        right = roll_out(
            DriftingSimulator(world_count=2, diverge_at_step=2, offset=1), "run-b", 7, 5
        )
        assert compare_runs(left, right) == {
            "left_label": "run-a",
            "right_label": "run-b",
            "digests_match": False,
            "first_divergent_step": 2,
            "compared_step_count": 5,
        }

    def test_shorter_run_sets_the_compared_step_count(self) -> None:
        """Comparison covers the common prefix and says how long it was."""
        left = roll_out(LinearSimulator(world_count=2), "run-a", 7, 3)
        right = roll_out(LinearSimulator(world_count=2), "run-b", 7, 6)
        assert compare_runs(left, right)["compared_step_count"] == 3

    def test_different_lengths_do_not_match_despite_agreeing_prefix(self) -> None:
        """A truncated run is a different run, even where it agrees."""
        left = roll_out(LinearSimulator(world_count=2), "run-a", 7, 3)
        right = roll_out(LinearSimulator(world_count=2), "run-b", 7, 6)
        result = compare_runs(left, right)
        assert result["digests_match"] is False
        assert result["first_divergent_step"] is None

    def test_rejects_comparison_across_seeds(self) -> None:
        """Different seeds cannot produce evidence about determinism."""
        left = roll_out(LinearSimulator(world_count=2), "run-a", 7, 3)
        right = roll_out(LinearSimulator(world_count=2), "run-b", 8, 3)
        with pytest.raises(ComparisonError) as caught:
            compare_runs(left, right)
        assert caught.value.code == "NP-COMPARE-001"

    def test_rejects_a_record_whose_digest_contradicts_its_steps(self) -> None:
        """A run digest not derived from the recorded steps is refused.

        Reachable only by constructing the contradiction directly, which is the
        point: the check exists so a hand-edited or truncated record cannot
        pass as agreement.
        """
        left = roll_out(
            DriftingSimulator(world_count=2, diverge_at_step=1, offset=0), "run-a", 7, 3
        )
        right = roll_out(
            DriftingSimulator(world_count=2, diverge_at_step=1, offset=1), "run-b", 7, 3
        )
        forged = RunRecord(spec=right["spec"], steps=right["steps"], digest=left["digest"])
        with pytest.raises(ComparisonError) as caught:
            compare_runs(left, forged)
        assert caught.value.code == "NP-COMPARE-002"
