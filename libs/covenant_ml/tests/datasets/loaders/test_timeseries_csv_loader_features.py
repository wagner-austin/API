"""Tests for TimeSeriesCSVLoader."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from covenant_ml.datasets.loaders.timeseries_csv_loader import (
    TimeSeriesCSVLoader,
)
from tests.datasets.loaders._timeseries_fixtures import (
    _copy_fixture_to_temp,
    _make_timeseries_config,
)


class TestTimeSeriesRankDiffFeatures:
    """Tests for rank and diff feature computation in loader."""

    def test_load_with_rank_features(self, tmp_path: Path) -> None:
        """Load adds rank features when enabled.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features, rank adds 2 features (one per base feature)
        config = _make_timeseries_config(
            n_features_expected=4,
            include_rank_features=True,
            include_diff_features=False,
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 4
        # Check feature names include rank suffix
        feature_names = result["meta"]["feature_names"]
        assert "feature_1_rank" in feature_names
        assert "feature_2_rank" in feature_names

    def test_load_with_diff_features(self, tmp_path: Path) -> None:
        """Load adds diff features when enabled.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features, diff adds 10 features (5 aggs per base feature)
        config = _make_timeseries_config(
            n_features_expected=12,
            include_rank_features=False,
            include_diff_features=True,
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 12
        # Check feature names include diff suffixes
        feature_names = result["meta"]["feature_names"]
        assert "feature_1_diff_mean" in feature_names
        assert "feature_2_diff_last" in feature_names

    def test_load_with_both_rank_and_diff_features(self, tmp_path: Path) -> None:
        """Load adds both rank and diff features when both enabled.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features, rank adds 2, diff adds 10 -> total 14
        config = _make_timeseries_config(
            n_features_expected=14,
            include_rank_features=True,
            include_diff_features=True,
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 14
        feature_names = result["meta"]["feature_names"]
        # Both rank and diff features present
        assert "feature_1_rank" in feature_names
        assert "feature_1_diff_mean" in feature_names

    def test_rank_features_values_in_range(self, tmp_path: Path) -> None:
        """Rank features are in [0, 1] range.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """

        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            include_rank_features=True,
            include_diff_features=False,
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        # Rank features are columns 2 and 3 (after base features 0 and 1)
        x_array = result["x"]
        rank_cols = x_array[:, 2:]
        assert np.all(rank_cols >= 0.0)
        assert np.all(rank_cols <= 1.0)


class TestTimeSeriesWindowFeatures:
    """Tests for window feature computation in loader."""

    def test_load_with_window_features(self, tmp_path: Path) -> None:
        """Load adds window features when enabled.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features, window size 2 adds 2*4=8 features
        config = _make_timeseries_config(
            n_features_expected=10,
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=True,
            window_sizes=(2,),
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 10
        # Check feature names include window suffix
        feature_names = result["meta"]["feature_names"]
        assert "feature_1_last2_mean" in feature_names
        assert "feature_2_last2_max" in feature_names

    def test_load_with_multiple_window_sizes(self, tmp_path: Path) -> None:
        """Load adds window features for multiple window sizes.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features, window sizes (2, 3) adds 2*4*2=16 features
        config = _make_timeseries_config(
            n_features_expected=18,
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=True,
            window_sizes=(2, 3),
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 18
        feature_names = result["meta"]["feature_names"]
        # Both window sizes should be present
        assert "feature_1_last2_mean" in feature_names
        assert "feature_1_last3_mean" in feature_names

    def test_load_with_all_feature_types(self, tmp_path: Path) -> None:
        """Load adds rank, diff, and window features when all enabled.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        # Base: 2 features
        # Rank: 2 features
        # Diff: 2*5=10 features
        # Window (size 2): 2*4=8 features
        # Total: 2 + 2 + 10 + 8 = 22
        config = _make_timeseries_config(
            n_features_expected=22,
            include_rank_features=True,
            include_diff_features=True,
            include_window_features=True,
            window_sizes=(2,),
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 22
        feature_names = result["meta"]["feature_names"]
        # All feature types present
        assert "feature_1_rank" in feature_names
        assert "feature_1_diff_mean" in feature_names
        assert "feature_1_last2_mean" in feature_names

    def test_load_window_features_disabled_with_empty_sizes(self, tmp_path: Path) -> None:
        """Load skips window features when flag True but sizes empty.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            n_features_expected=2,
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=True,
            window_sizes=(),  # Empty sizes
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        # No window features added because sizes is empty
        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2

    def test_load_window_features_disabled_flag_false(self, tmp_path: Path) -> None:
        """Load skips window features when flag is False.

        Args:
            tmp_path: Pytest temp directory for isolated testing.
        """
        loader = TimeSeriesCSVLoader()
        config = _make_timeseries_config(
            n_features_expected=2,
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=False,
            window_sizes=(2, 3),  # Sizes provided but flag is False
        )
        fixtures_dir = _copy_fixture_to_temp(tmp_path, "timeseries_simple")

        result = loader.load(config, fixtures_dir)

        # No window features added because flag is False
        assert result["meta"]["n_samples"] == 3
        assert result["meta"]["n_features"] == 2
