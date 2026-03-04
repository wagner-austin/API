"""Tests for temporal feature preset and engineer_features temporal integration."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.features import (
    FeatureEngineeringConfig,
    engineer_features,
    get_feature_config_for_preset,
)


def _make_array(rows: int, cols: int, values: tuple[float, ...]) -> NDArray[np.float64]:
    """Create a float64 array from flat values."""
    arr: NDArray[np.float64] = np.zeros((rows, cols), dtype=np.float64)
    for i, v in enumerate(values):
        arr[i // cols, i % cols] = v
    return arr


def _get_value(arr: NDArray[np.float64], row: int, col: int) -> float:
    """Extract a scalar value from array at given position."""
    idx = row * int(arr.shape[1]) + col
    for i, val in enumerate(arr.flat):
        if i == idx:
            return float(val)
    raise IndexError(f"Invalid index ({row}, {col}) for array shape {arr.shape}")


class TestTemporalPreset:
    """Tests for the 'temporal' feature preset."""

    def test_temporal_preset_config(self) -> None:
        """'temporal' preset enables only temporal features."""
        config = get_feature_config_for_preset("temporal")

        assert config["use_ratios"] is False
        assert config["use_products"] is False
        assert config["use_log_transforms"] is False
        assert config["use_temporal"] is True
        assert config["max_ratio_features"] == 0
        assert config["max_product_features"] == 0

    def test_temporal_preset_has_all_keys(self) -> None:
        """'temporal' preset returns all required FeatureEngineeringConfig keys."""
        config = get_feature_config_for_preset("temporal")

        assert "use_ratios" in config
        assert "use_products" in config
        assert "use_log_transforms" in config
        assert "use_temporal" in config
        assert "max_ratio_features" in config
        assert "max_product_features" in config


class TestEngineerFeaturesWithTemporal:
    """Tests for engineer_features with temporal features."""

    def test_temporal_features_appended(self) -> None:
        """Temporal features are appended when use_temporal is True."""
        x = _make_array(3, 2, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        names = ["A", "B"]
        temporal = _make_array(3, 2, (10.0, 20.0, 30.0, 40.0, 50.0, 60.0))
        temporal_names = ("seasonal_max", "cum_excess_hot")
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": False,
            "use_temporal": True,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }

        result = engineer_features(x, names, config, temporal, temporal_names)

        # Original (2) + temporal (2) = 4
        assert result["x"].shape == (3, 4)
        assert result["n_original"] == 2
        assert result["n_temporal"] == 2
        assert result["n_ratios"] == 0
        assert result["n_products"] == 0
        assert result["n_log"] == 0
        assert result["feature_names"] == ["A", "B", "seasonal_max", "cum_excess_hot"]

    def test_temporal_values_correct(self) -> None:
        """Temporal feature values are correctly placed in output."""
        x = _make_array(2, 1, (1.0, 2.0))
        temporal = _make_array(2, 2, (10.0, 20.0, 30.0, 40.0))
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": False,
            "use_temporal": True,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }

        result = engineer_features(
            x,
            ["A"],
            config,
            temporal,
            ("t1", "t2"),
        )

        # Column 0 = original A, columns 1-2 = temporal
        assert _get_value(result["x"], 0, 0) == 1.0
        assert _get_value(result["x"], 0, 1) == 10.0
        assert _get_value(result["x"], 0, 2) == 20.0
        assert _get_value(result["x"], 1, 0) == 2.0
        assert _get_value(result["x"], 1, 1) == 30.0
        assert _get_value(result["x"], 1, 2) == 40.0

    def test_temporal_with_other_features(self) -> None:
        """Temporal features combine with log transforms."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        temporal = _make_array(2, 1, (10.0, 20.0))
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": True,
            "use_temporal": True,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }

        result = engineer_features(
            x,
            ["A", "B"],
            config,
            temporal,
            ("t1",),
        )

        # Original (2) + log (2) + temporal (1) = 5
        assert result["x"].shape == (2, 5)
        assert result["n_original"] == 2
        assert result["n_log"] == 2
        assert result["n_temporal"] == 1
        assert result["feature_names"] == [
            "A",
            "B",
            "log(A)",
            "log(B)",
            "t1",
        ]

    def test_temporal_with_all_features(self) -> None:
        """Temporal features combine with ratios, products, and logs."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        temporal = _make_array(2, 3, (10.0, 20.0, 30.0, 40.0, 50.0, 60.0))
        config: FeatureEngineeringConfig = {
            "use_ratios": True,
            "use_products": True,
            "use_log_transforms": True,
            "use_temporal": True,
            "max_ratio_features": 100,
            "max_product_features": 100,
        }

        result = engineer_features(
            x,
            ["A", "B"],
            config,
            temporal,
            ("t1", "t2", "t3"),
        )

        # Original (2) + ratios (2) + products (1) + log (2) + temporal (3) = 10
        assert result["x"].shape == (2, 10)
        assert result["n_original"] == 2
        assert result["n_ratios"] == 2
        assert result["n_products"] == 1
        assert result["n_log"] == 2
        assert result["n_temporal"] == 3

    def test_output_dtype_is_float64(self) -> None:
        """Output is float64 when temporal features are included."""
        x = _make_array(2, 1, (1.0, 2.0))
        temporal = _make_array(2, 1, (10.0, 20.0))
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": False,
            "use_temporal": True,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }

        result = engineer_features(x, ["A"], config, temporal, ("t1",))

        assert result["x"].dtype == np.float64

    def test_temporal_ignored_when_disabled(self) -> None:
        """Temporal features are ignored when use_temporal is False."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        temporal = _make_array(2, 1, (10.0, 20.0))
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": False,
            "use_temporal": False,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }

        result = engineer_features(
            x,
            ["A", "B"],
            config,
            temporal,
            ("t1",),
        )

        # Temporal features should be ignored
        assert result["x"].shape == (2, 2)
        assert result["n_temporal"] == 0
        assert result["feature_names"] == ["A", "B"]


class TestEngineerFeaturesTemporalErrors:
    """Tests for engineer_features temporal error paths."""

    def test_use_temporal_without_features_raises(self) -> None:
        """Raises ValueError when use_temporal is True but no features given."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": False,
            "use_temporal": True,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }

        with pytest.raises(ValueError, match="temporal_features was not provided"):
            engineer_features(x, ["A", "B"], config)

    def test_use_temporal_with_none_features_raises(self) -> None:
        """Raises ValueError when use_temporal is True and features is None."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": False,
            "use_temporal": True,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }

        with pytest.raises(ValueError, match="temporal_features was not provided"):
            engineer_features(x, ["A", "B"], config, None, ())

    def test_use_temporal_with_empty_names_raises(self) -> None:
        """Raises ValueError when use_temporal is True but names are empty."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        temporal = _make_array(2, 1, (10.0, 20.0))
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": False,
            "use_temporal": True,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }

        with pytest.raises(ValueError, match="temporal_feature_names is empty"):
            engineer_features(x, ["A", "B"], config, temporal, ())

    def test_temporal_sample_mismatch_raises(self) -> None:
        """Raises ValueError when temporal feature sample count mismatches."""
        x = _make_array(3, 2, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        temporal = _make_array(2, 1, (10.0, 20.0))  # 2 samples vs 3
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": False,
            "use_temporal": True,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }

        with pytest.raises(ValueError, match="temporal_features has 2 samples but x has 3"):
            engineer_features(
                x,
                ["A", "B"],
                config,
                temporal,
                ("t1",),
            )
