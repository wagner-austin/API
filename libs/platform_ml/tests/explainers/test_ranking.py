"""Ranking importances, which three modules used to do for themselves.

Two of the three copies were in this package's own explainers directory, side
by side. Tested directly here so the ordering rules are stated once, rather
than being implied by whichever explainer's test happened to exercise them.
"""

from __future__ import annotations

from platform_ml.explainers.ranking import rank_importances

from .array_helpers import make_float64_1d, make_float64_2d


class TestOrdering:
    def test_most_important_comes_first(self) -> None:
        scores = rank_importances(["a", "b", "c"], make_float64_1d([0.1, 0.9, 0.5]))

        assert [s["name"] for s in scores] == ["b", "c", "a"]

    def test_ranks_are_one_based_and_contiguous(self) -> None:
        """Callers render `rank` to a user, so a zero-based or gapped
        sequence would show up as 'feature #0'."""
        scores = rank_importances(["a", "b", "c"], make_float64_1d([0.1, 0.9, 0.5]))

        assert [s["rank"] for s in scores] == [1, 2, 3]

    def test_the_importance_travels_with_its_name(self) -> None:
        """The pairing is the whole job: sorting the values and leaving the
        names in place would produce a plausible, wrong answer."""
        scores = rank_importances(["a", "b"], make_float64_1d([0.25, 0.75]))

        assert scores == [
            {"name": "b", "importance": 0.75, "rank": 1},
            {"name": "a", "importance": 0.25, "rank": 2},
        ]

    def test_equal_importances_still_get_contiguous_ranks(self) -> None:
        """Ties have no defined order, but every feature must still get its
        own rank -- two features sharing rank 1, or a gap, would render as a
        broken list to whoever reads it."""
        scores = rank_importances(["x", "y", "z"], make_float64_1d([0.5, 0.5, 0.5]))

        assert sorted(s["rank"] for s in scores) == [1, 2, 3]
        assert sorted(s["name"] for s in scores) == ["x", "y", "z"]

    def test_negative_importances_sort_below_positive_ones(self) -> None:
        """Gradient-based explainers produce signed values."""
        scores = rank_importances(["a", "b", "c"], make_float64_1d([-0.5, 0.1, -0.9]))

        assert [s["name"] for s in scores] == ["b", "a", "c"]


class TestArrayShapes:
    def test_a_flat_array_is_ranked(self) -> None:
        scores = rank_importances(["a", "b"], make_float64_1d([0.1, 0.2]))

        assert [s["name"] for s in scores] == ["b", "a"]

    def test_a_row_vector_is_ranked_the_same_way(self) -> None:
        """It reads through `.flat` because the callers produce both shapes --
        a permutation explainer gives (n,) and a gradient one gives (1, n)."""
        flat = rank_importances(["a", "b"], make_float64_1d([0.1, 0.2]))
        row = rank_importances(["a", "b"], make_float64_2d([[0.1, 0.2]]))

        assert row == flat

    def test_a_single_feature_yields_one_score(self) -> None:
        assert rank_importances(["only"], make_float64_1d([0.4])) == [
            {"name": "only", "importance": 0.4, "rank": 1}
        ]

    def test_no_features_yields_no_scores(self) -> None:
        assert rank_importances([], make_float64_1d([])) == []


__all__ = ["TestArrayShapes", "TestOrdering"]
