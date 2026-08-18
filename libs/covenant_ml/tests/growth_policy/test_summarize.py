"""Tests for reducing per-seed results to per-arm means."""

from __future__ import annotations

import pytest

from covenant_ml.growth_policy.summarize import summarize_arms
from covenant_ml.growth_policy.types import ERR_NO_RESULTS

from .factories import make_arm_result


class TestSummarizeArms:
    """Grouping, averaging and ordering."""

    def test_averages_every_field_across_seeds(self) -> None:
        """Two results scaled 1x and 3x should average to 2x on every field."""
        results = [
            make_arm_result(arm="arm-a", seed=42, scale=1.0),
            make_arm_result(arm="arm-a", seed=43, scale=3.0),
        ]

        summaries = summarize_arms(results)

        assert len(summaries) == 1
        assert summaries[0]["seed_count"] == 2
        assert summaries[0]["fit_seconds"] == 2.0
        assert summaries[0]["auc_roc"] == 1.0
        assert summaries[0]["auc_pr"] == 0.5
        assert summaries[0]["log_loss"] == 0.25
        assert summaries[0]["mean_leaves"] == 8.0

    def test_groups_by_arm(self) -> None:
        """Each arm should get exactly one summary."""
        results = [
            make_arm_result(arm="arm-a", seed=42),
            make_arm_result(arm="arm-b", seed=42),
            make_arm_result(arm="arm-a", seed=43),
        ]

        summaries = summarize_arms(results)

        assert [summary["arm"] for summary in summaries] == ["arm-a", "arm-b"]
        assert [summary["seed_count"] for summary in summaries] == [2, 1]

    def test_keeps_first_appearance_order(self) -> None:
        """Arms must not be sorted, so a table reads as the specified progression."""
        results = [
            make_arm_result(arm="zeta", seed=42),
            make_arm_result(arm="alpha", seed=42),
        ]

        assert [summary["arm"] for summary in summarize_arms(results)] == [
            "zeta",
            "alpha",
        ]

    def test_rejects_an_empty_result_set(self) -> None:
        """An empty table presented as a finished experiment must fail instead."""
        with pytest.raises(ValueError, match=ERR_NO_RESULTS):
            summarize_arms([])
