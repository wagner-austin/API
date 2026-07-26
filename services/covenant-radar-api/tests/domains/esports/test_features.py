"""Tests for esports win-probability feature extraction.

Every feature is a pure function of one snapshot, so each value is exactly
predictable from the input and the assertions here are equalities rather
than bounds. The two cases worth naming are the ones that only occur at the
first tick of a game: an empty scoreline, where a share has no denominator,
and zero elapsed time, where a per-minute rate has no divisor.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_radar_api.domains.esports.features import (
    ESPORTS_FEATURE_NAMES,
    EsportsFeatureExtractor,
)
from covenant_radar_api.domains.esports.schemas import MatchEventV1
from tests.domains.esports._test_esports_fixtures import make_snapshot


def _values(array: NDArray[np.float64]) -> list[float]:
    """Read a 1D float64 array out as plain floats.

    Args:
        array: Feature vector to read.

    Returns:
        The values in order, so two vectors compare as ordinary lists.
    """
    return [float(array.flat[index]) for index in range(int(array.shape[0]))]


def _named(event: MatchEventV1) -> dict[str, float]:
    """Extract features and label each with its name.

    Args:
        event: Match snapshot to featurise.

    Returns:
        Mapping of feature name to value. The width is asserted against the
        declared names, so a dropped or extra feature fails here rather
        than silently shifting every later column.
    """
    extractor = EsportsFeatureExtractor()
    values: NDArray[np.float64] = extractor.extract(event)
    assert int(values.shape[0]) == len(ESPORTS_FEATURE_NAMES)
    return {name: float(values.flat[index]) for index, name in enumerate(ESPORTS_FEATURE_NAMES)}


class TestFeatureNames:
    """The declared names are the contract the model column order rests on."""

    def test_extractor_reports_the_module_names(self) -> None:
        """The property is the same tuple, not a second list that can drift."""
        assert EsportsFeatureExtractor().feature_names == ESPORTS_FEATURE_NAMES

    def test_names_are_unique(self) -> None:
        """A duplicated name would make two columns indistinguishable."""
        assert len(set(ESPORTS_FEATURE_NAMES)) == len(ESPORTS_FEATURE_NAMES)

    def test_twelve_features(self) -> None:
        """The width is fixed; a model trained on it cannot accept another."""
        assert len(ESPORTS_FEATURE_NAMES) == 12


class TestDifferences:
    """Every difference is blue minus red, so the sign names the leader."""

    def test_blue_ahead_gives_positive_differences(self) -> None:
        """Blue leading on every count produces positive differences."""
        named = _named(
            make_snapshot(
                blue_kills=12,
                red_kills=5,
                blue_gold=45000,
                red_gold=38000,
                blue_towers=6,
                red_towers=2,
                blue_dragons=3,
                red_dragons=1,
                blue_barons=1,
                red_barons=0,
            )
        )

        assert named["kill_diff"] == pytest.approx(7.0)
        assert named["gold_diff"] == pytest.approx(7000.0)
        assert named["tower_diff"] == pytest.approx(4.0)
        assert named["dragon_diff"] == pytest.approx(2.0)
        assert named["baron_diff"] == pytest.approx(1.0)

    def test_red_ahead_gives_negative_differences(self) -> None:
        """Red leading flips every sign; nothing is clamped at zero."""
        named = _named(
            make_snapshot(
                blue_kills=3,
                red_kills=11,
                blue_gold=31000,
                red_gold=40000,
                blue_towers=1,
                red_towers=5,
                blue_dragons=0,
                red_dragons=3,
                blue_barons=0,
                red_barons=2,
            )
        )

        assert named["kill_diff"] == pytest.approx(-8.0)
        assert named["gold_diff"] == pytest.approx(-9000.0)
        assert named["tower_diff"] == pytest.approx(-4.0)
        assert named["dragon_diff"] == pytest.approx(-3.0)
        assert named["baron_diff"] == pytest.approx(-2.0)

    def test_even_match_gives_zero_differences(self) -> None:
        """Identical scorelines leave every difference at zero."""
        named = _named(
            make_snapshot(
                blue_kills=7,
                red_kills=7,
                blue_gold=40000,
                red_gold=40000,
                blue_towers=3,
                red_towers=3,
            )
        )

        assert named["kill_diff"] == pytest.approx(0.0)
        assert named["gold_diff"] == pytest.approx(0.0)
        assert named["tower_diff"] == pytest.approx(0.0)


class TestGoldRate:
    """A gold lead is read against how long it took to build."""

    def test_rate_divides_by_elapsed_minutes(self) -> None:
        """Ten minutes and a 5000 lead is 500 gold per minute."""
        named = _named(make_snapshot(game_time_seconds=600, blue_gold=45000, red_gold=40000))

        assert named["game_time_minutes"] == pytest.approx(10.0)
        assert named["gold_diff_per_minute"] == pytest.approx(500.0)

    def test_the_same_lead_later_is_a_smaller_rate(self) -> None:
        """This is the whole point of the feature: timing changes meaning.

        A 5000 lead at ten minutes is decisive; the same lead at forty is
        close to even, and gold_diff alone cannot tell them apart.
        """
        early = _named(make_snapshot(game_time_seconds=600, blue_gold=45000, red_gold=40000))
        late = _named(make_snapshot(game_time_seconds=2400, blue_gold=45000, red_gold=40000))

        assert early["gold_diff"] == late["gold_diff"]
        assert late["gold_diff_per_minute"] == pytest.approx(125.0)
        assert late["gold_diff_per_minute"] < early["gold_diff_per_minute"]

    def test_no_elapsed_time_gives_no_rate(self) -> None:
        """At the first tick no rate exists, and zero is the honest value.

        Any non-zero number here would assert a trend from a single
        instant, and dividing would be a division by zero.
        """
        named = _named(make_snapshot(game_time_seconds=0, blue_gold=0, red_gold=0))

        assert named["game_time_minutes"] == pytest.approx(0.0)
        assert named["gold_diff_per_minute"] == pytest.approx(0.0)

    def test_zero_elapsed_time_with_a_lead_still_gives_no_rate(self) -> None:
        """The guard is on elapsed time, not on the lead being zero."""
        named = _named(make_snapshot(game_time_seconds=0, blue_gold=500, red_gold=0))

        assert named["gold_diff"] == pytest.approx(500.0)
        assert named["gold_diff_per_minute"] == pytest.approx(0.0)

    def test_negative_lead_gives_a_negative_rate(self) -> None:
        """Red ahead produces a negative rate, not an absolute value."""
        named = _named(make_snapshot(game_time_seconds=1200, blue_gold=38000, red_gold=42000))

        assert named["gold_diff_per_minute"] == pytest.approx(-200.0)


class TestShares:
    """Shares put each side's total on a common [0, 1] scale."""

    def test_share_is_the_fraction_of_the_total(self) -> None:
        """Twelve of sixteen kills is a share of 0.75."""
        named = _named(make_snapshot(blue_kills=12, red_kills=4, blue_gold=30000, red_gold=10000))

        assert named["blue_kill_ratio"] == pytest.approx(0.75)
        assert named["blue_gold_ratio"] == pytest.approx(0.75)

    def test_empty_scoreline_is_an_even_share(self) -> None:
        """Before either side scores, neither holds any of the total.

        Zero would read as red holding everything, which is the opposite of
        what an empty scoreline means, and is the value a naive guard
        against division by zero would produce.
        """
        named = _named(make_snapshot(blue_kills=0, red_kills=0, blue_gold=0, red_gold=0))

        assert named["blue_kill_ratio"] == pytest.approx(0.5)
        assert named["blue_gold_ratio"] == pytest.approx(0.5)

    def test_one_sided_scoreline_reaches_the_bounds(self) -> None:
        """A shutout is a share of one, and its mirror is zero."""
        blue = _named(make_snapshot(blue_kills=5, red_kills=0))
        red = _named(make_snapshot(blue_kills=0, red_kills=5))

        assert blue["blue_kill_ratio"] == pytest.approx(1.0)
        assert red["blue_kill_ratio"] == pytest.approx(0.0)

    def test_shares_stay_within_the_unit_interval(self) -> None:
        """Any scoreline keeps both shares in [0, 1]."""
        named = _named(make_snapshot(blue_kills=1, red_kills=37, blue_gold=1, red_gold=99999))

        assert 0.0 <= named["blue_kill_ratio"] <= 1.0
        assert 0.0 <= named["blue_gold_ratio"] <= 1.0


class TestObjectives:
    """Objectives sum the three structures that decide a game."""

    def test_objectives_sum_towers_dragons_and_barons(self) -> None:
        """Each side's total is the sum of its three counts."""
        named = _named(
            make_snapshot(
                blue_towers=5,
                blue_dragons=3,
                blue_barons=1,
                red_towers=2,
                red_dragons=1,
                red_barons=0,
            )
        )

        assert named["blue_objectives"] == pytest.approx(9.0)
        assert named["red_objectives"] == pytest.approx(3.0)

    def test_objective_diff_is_the_difference_of_the_sums(self) -> None:
        """The difference is derived, not a fourth independent count."""
        named = _named(
            make_snapshot(
                blue_towers=5,
                blue_dragons=3,
                blue_barons=1,
                red_towers=2,
                red_dragons=1,
                red_barons=0,
            )
        )

        assert named["objective_diff"] == pytest.approx(
            named["blue_objectives"] - named["red_objectives"]
        )

    def test_no_objectives_taken_is_all_zero(self) -> None:
        """The opening state carries no objectives for either side."""
        named = _named(make_snapshot())

        assert named["blue_objectives"] == pytest.approx(0.0)
        assert named["red_objectives"] == pytest.approx(0.0)
        assert named["objective_diff"] == pytest.approx(0.0)


class TestVectorContract:
    """The array handed to the model is well-formed."""

    def test_dtype_is_float64(self) -> None:
        """The model contract is float64; another dtype would be coerced."""
        assert EsportsFeatureExtractor().extract(make_snapshot()).dtype == np.float64

    def test_every_value_is_finite(self) -> None:
        """A non-finite feature would poison the model input silently."""
        values = EsportsFeatureExtractor().extract(
            make_snapshot(game_time_seconds=0, blue_gold=99999, red_gold=0)
        )

        finite: NDArray[np.bool_] = np.isfinite(values)
        assert int(np.count_nonzero(finite)) == int(values.size)

    def test_shape_is_one_dimensional(self) -> None:
        """A column vector would broadcast wrongly against the model."""
        assert EsportsFeatureExtractor().extract(make_snapshot()).shape == (12,)


class TestStatelessness:
    """No fitted state means one extractor serves every match at once."""

    def test_extraction_does_not_depend_on_call_order(self) -> None:
        """Feeding another match between two calls cannot change the result.

        Weather carries a fitted state; this extractor carries none, so one
        instance is safe to share across concurrent matches.
        """
        extractor = EsportsFeatureExtractor()
        snapshot = make_snapshot(blue_kills=9, red_kills=3, blue_gold=44000, red_gold=39000)

        first: NDArray[np.float64] = extractor.extract(snapshot)
        extractor.extract(make_snapshot(blue_kills=0, red_kills=25, game_time_seconds=1))
        second: NDArray[np.float64] = extractor.extract(snapshot)

        assert _values(first) == _values(second)

    def test_two_extractors_agree(self) -> None:
        """Separate instances produce identical output for identical input."""
        snapshot = make_snapshot(blue_kills=4, red_kills=4, blue_gold=41000, red_gold=40000)

        first: NDArray[np.float64] = EsportsFeatureExtractor().extract(snapshot)
        second: NDArray[np.float64] = EsportsFeatureExtractor().extract(snapshot)

        assert _values(first) == _values(second)
