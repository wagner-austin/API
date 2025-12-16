"""Tests for feature engineering module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.features import (
    FeatureEngineeringConfig,
    FeaturePreset,
    compute_log_transforms,
    compute_pairwise_products,
    compute_pairwise_ratios,
    default_feature_config,
    engineer_features,
    get_feature_config_for_preset,
)


def _make_array(rows: int, cols: int, values: tuple[float, ...]) -> NDArray[np.float64]:
    """Create a float64 array from flat values."""
    arr: NDArray[np.float64] = np.zeros((rows, cols), dtype=np.float64)
    for i, v in enumerate(values):
        arr[i // cols, i % cols] = v
    return arr


def _all_finite(arr: NDArray[np.float64]) -> bool:
    """Check if all array values are finite."""
    finite_mask: NDArray[np.bool_] = np.isfinite(arr)
    # np.all returns np.bool_, which is not NDArray
    return bool(np.all(finite_mask))


def _get_value(arr: NDArray[np.float64], row: int, col: int) -> float:
    """Extract a scalar value from array at given position."""
    # Use flat iteration to get typed access
    idx = row * int(arr.shape[1]) + col
    for i, val in enumerate(arr.flat):
        if i == idx:
            return float(val)
    raise IndexError(f"Invalid index ({row}, {col}) for array shape {arr.shape}")


def _get_max_abs(arr: NDArray[np.float64]) -> float:
    """Get maximum absolute value in array."""
    max_val: float = 0.0
    for val in arr.flat:
        abs_val: float = abs(float(val))
        if abs_val > max_val:
            max_val = abs_val
    return max_val


def _get_n_cols(arr: NDArray[np.float64]) -> int:
    """Get number of columns in array."""
    return int(arr.shape[1])


class TestDefaultFeatureConfig:
    """Tests for default_feature_config."""

    def test_returns_typed_dict(self) -> None:
        """default_feature_config returns FeatureEngineeringConfig."""
        config = default_feature_config()
        assert config["use_ratios"] is True
        assert config["use_products"] is False
        assert config["use_log_transforms"] is True
        assert config["max_ratio_features"] == 500
        assert config["max_product_features"] == 200


class TestComputePairwiseRatios:
    """Tests for compute_pairwise_ratios."""

    def test_basic_ratios(self) -> None:
        """Computes basic pairwise ratios."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        names = ["A", "B"]

        ratios, ratio_names = compute_pairwise_ratios(x, names)

        assert ratios.shape == (2, 2)  # A/B and B/A
        assert len(ratio_names) == 2
        assert "A/B" in ratio_names
        assert "B/A" in ratio_names

    def test_handles_zero_denominator(self) -> None:
        """Handles division by zero gracefully."""
        x = _make_array(2, 2, (1.0, 0.0, 2.0, 0.0))
        names = ["A", "B"]

        ratios, _ = compute_pairwise_ratios(x, names)

        # Should not have inf/nan
        assert _all_finite(ratios)

    def test_max_features_limit(self) -> None:
        """Respects max_features limit."""
        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((10, 10)).astype(np.float64)
        names = [f"X{i}" for i in range(10)]

        ratios, ratio_names = compute_pairwise_ratios(x, names, max_features=5)

        assert _get_n_cols(ratios) <= 5
        assert len(ratio_names) <= 5

    def test_single_feature_returns_empty(self) -> None:
        """Returns empty for single feature."""
        x = _make_array(2, 1, (1.0, 2.0))
        names = ["A"]

        ratios, ratio_names = compute_pairwise_ratios(x, names)

        assert ratios.shape == (2, 0)
        assert len(ratio_names) == 0

    def test_clips_extreme_values(self) -> None:
        """Clips extreme ratio values."""
        x = _make_array(2, 2, (1e10, 1e-10, 1.0, 1.0))
        names = ["A", "B"]

        ratios, _ = compute_pairwise_ratios(x, names)

        # Should be clipped to reasonable range
        max_abs = _get_max_abs(ratios)
        assert max_abs <= 1e6


class TestComputePairwiseProducts:
    """Tests for compute_pairwise_products."""

    def test_basic_products(self) -> None:
        """Computes basic pairwise products."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        names = ["A", "B"]

        products, product_names = compute_pairwise_products(x, names)

        assert products.shape == (2, 1)  # Only A*B (symmetric)
        assert len(product_names) == 1
        assert "A*B" in product_names

        # Check values
        assert _get_value(products, 0, 0) == 2.0  # 1 * 2
        assert _get_value(products, 1, 0) == 12.0  # 3 * 4

    def test_three_features(self) -> None:
        """Computes products for three features."""
        x = _make_array(2, 3, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        names = ["A", "B", "C"]

        products, product_names = compute_pairwise_products(x, names)

        # Should have 3 products: A*B, A*C, B*C
        assert products.shape == (2, 3)
        assert len(product_names) == 3

    def test_max_features_limit(self) -> None:
        """Respects max_features limit."""
        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((10, 10)).astype(np.float64)
        names = [f"X{i}" for i in range(10)]

        products, product_names = compute_pairwise_products(x, names, max_features=5)

        assert _get_n_cols(products) <= 5
        assert len(product_names) <= 5

    def test_single_feature_returns_empty(self) -> None:
        """Returns empty for single feature."""
        x = _make_array(2, 1, (1.0, 2.0))
        names = ["A"]

        products, product_names = compute_pairwise_products(x, names)

        assert products.shape == (2, 0)
        assert len(product_names) == 0

    def test_handles_extreme_values(self) -> None:
        """Handles extreme product values."""
        x = _make_array(2, 2, (1e8, 1e8, 1.0, 1.0))
        names = ["A", "B"]

        products, _ = compute_pairwise_products(x, names)

        # Should be finite
        assert _all_finite(products)


class TestComputeLogTransforms:
    """Tests for compute_log_transforms."""

    def test_positive_values(self) -> None:
        """Computes log for positive values."""
        x = _make_array(2, 2, (1.0, 10.0, 100.0, 1000.0))
        names = ["A", "B"]

        logs, log_names = compute_log_transforms(x, names)

        assert logs.shape == x.shape
        assert len(log_names) == 2
        assert "log(A)" in log_names
        assert "log(B)" in log_names

        # Check values are reasonable
        # Use math.log1p for typed scalar computation
        import math

        expected_log_1: float = math.log1p(1.0)
        expected_log_10: float = math.log1p(10.0)
        assert _get_value(logs, 0, 0) == pytest.approx(expected_log_1)
        assert _get_value(logs, 0, 1) == pytest.approx(expected_log_10)

    def test_negative_values(self) -> None:
        """Handles negative values with signed log."""
        x = _make_array(2, 2, (-1.0, -10.0, 1.0, 10.0))
        names = ["A", "B"]

        logs, _ = compute_log_transforms(x, names)

        # Negative input should give negative output
        assert _get_value(logs, 0, 0) < 0
        assert _get_value(logs, 0, 1) < 0
        # Positive input should give positive output
        assert _get_value(logs, 1, 0) > 0
        assert _get_value(logs, 1, 1) > 0

    def test_zero_values(self) -> None:
        """Handles zero values."""
        x = _make_array(1, 2, (0.0, 0.0))
        names = ["A", "B"]

        logs, _ = compute_log_transforms(x, names)

        # log1p(0) = 0
        assert _get_value(logs, 0, 0) == 0.0
        assert _get_value(logs, 0, 1) == 0.0

    def test_all_finite(self) -> None:
        """All outputs are finite."""
        x = _make_array(2, 2, (1e-10, 1e10, -1e10, 0.0))
        names = ["A", "B"]

        logs, _ = compute_log_transforms(x, names)

        assert _all_finite(logs)


class TestEngineerFeatures:
    """Tests for engineer_features main function."""

    def test_no_engineering(self) -> None:
        """Returns original features when all options disabled."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        names = ["A", "B"]
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": False,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }

        result = engineer_features(x, names, config)

        assert result["x"].shape == (2, 2)
        assert result["feature_names"] == ["A", "B"]
        assert result["n_original"] == 2
        assert result["n_ratios"] == 0
        assert result["n_products"] == 0
        assert result["n_log"] == 0

    def test_log_only(self) -> None:
        """Adds only log transforms."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        names = ["A", "B"]
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": True,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }

        result = engineer_features(x, names, config)

        # Original (2) + log transforms (2)
        assert result["x"].shape == (2, 4)
        assert result["n_original"] == 2
        assert result["n_log"] == 2
        assert "log(A)" in result["feature_names"]
        assert "log(B)" in result["feature_names"]

    def test_ratios_only(self) -> None:
        """Adds only ratio features."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        names = ["A", "B"]
        config: FeatureEngineeringConfig = {
            "use_ratios": True,
            "use_products": False,
            "use_log_transforms": False,
            "max_ratio_features": 100,
            "max_product_features": 0,
        }

        result = engineer_features(x, names, config)

        # Original (2) + ratios (2: A/B, B/A)
        assert result["x"].shape == (2, 4)
        assert result["n_original"] == 2
        assert result["n_ratios"] == 2
        assert "A/B" in result["feature_names"]

    def test_products_only(self) -> None:
        """Adds only product features."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        names = ["A", "B"]
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": True,
            "use_log_transforms": False,
            "max_ratio_features": 0,
            "max_product_features": 100,
        }

        result = engineer_features(x, names, config)

        # Original (2) + products (1: A*B)
        assert result["x"].shape == (2, 3)
        assert result["n_original"] == 2
        assert result["n_products"] == 1
        assert "A*B" in result["feature_names"]

    def test_full_engineering(self) -> None:
        """Adds all feature types."""
        x = _make_array(2, 2, (1.0, 2.0, 3.0, 4.0))
        names = ["A", "B"]
        config: FeatureEngineeringConfig = {
            "use_ratios": True,
            "use_products": True,
            "use_log_transforms": True,
            "max_ratio_features": 100,
            "max_product_features": 100,
        }

        result = engineer_features(x, names, config)

        # Original (2) + ratios (2) + products (1) + log (2) = 7
        assert result["x"].shape == (2, 7)
        assert result["n_original"] == 2
        assert result["n_ratios"] == 2
        assert result["n_products"] == 1
        assert result["n_log"] == 2

    def test_output_is_float64(self) -> None:
        """Output array is float64."""
        x = _make_array(1, 2, (1.0, 2.0))
        names = ["A", "B"]
        config = default_feature_config()

        result = engineer_features(x, names, config)

        assert result["x"].dtype == np.float64


class TestGetFeatureConfigForPreset:
    """Tests for get_feature_config_for_preset."""

    def test_none_preset(self) -> None:
        """'none' preset disables all features."""
        config = get_feature_config_for_preset("none")

        assert config["use_ratios"] is False
        assert config["use_products"] is False
        assert config["use_log_transforms"] is False

    def test_log_only_preset(self) -> None:
        """'log_only' preset enables only log transforms."""
        config = get_feature_config_for_preset("log_only")

        assert config["use_ratios"] is False
        assert config["use_products"] is False
        assert config["use_log_transforms"] is True

    def test_ratios_only_preset(self) -> None:
        """'ratios_only' preset enables only ratios."""
        config = get_feature_config_for_preset("ratios_only")

        assert config["use_ratios"] is True
        assert config["use_products"] is False
        assert config["use_log_transforms"] is False
        assert config["max_ratio_features"] == 500

    def test_full_preset(self) -> None:
        """'full' preset enables everything."""
        config = get_feature_config_for_preset("full")

        assert config["use_ratios"] is True
        assert config["use_products"] is True
        assert config["use_log_transforms"] is True
        assert config["max_ratio_features"] == 500
        assert config["max_product_features"] == 200

    def test_all_presets_return_typed_config(self) -> None:
        """All presets return valid FeatureEngineeringConfig."""
        presets: list[FeaturePreset] = ["none", "log_only", "ratios_only", "full"]

        for preset in presets:
            config = get_feature_config_for_preset(preset)
            # All required keys present
            assert "use_ratios" in config
            assert "use_products" in config
            assert "use_log_transforms" in config
            assert "max_ratio_features" in config
            assert "max_product_features" in config


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_get_index_value_out_of_bounds(self) -> None:
        """_get_index_value raises IndexError for invalid index."""
        from covenant_ml.features import _get_index_value

        arr: NDArray[np.intp] = np.arange(3, dtype=np.intp)
        with pytest.raises(IndexError):
            _get_index_value(arr, 10)


class TestMaxFeaturesEarlyBreak:
    """Tests for max_features early break paths."""

    def test_products_max_features_breaks(self) -> None:
        """Products hits max_features limit and triggers breaks.

        With 3 features, n_pairs = C(3,2) = 3 products.
        Setting max_features = n_pairs avoids variance selection
        but still allows us to hit the breaks.
        Loop: A*B (1), A*C (2), B*C (3) -> break on 3rd.
        """
        x = _make_array(2, 3, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        names = ["A", "B", "C"]

        # max_features = n_pairs = 3, so no variance selection
        # but breaks trigger after generating 3 products
        products, product_names = compute_pairwise_products(x, names, max_features=3)

        assert _get_n_cols(products) == 3
        assert len(product_names) == 3


class TestEngineerFeaturesEdgeCases:
    """Tests for edge cases in engineer_features."""

    def test_single_feature_with_ratios_enabled(self) -> None:
        """Single feature with ratios enabled returns n_ratios=0."""
        x = _make_array(2, 1, (1.0, 2.0))
        names = ["A"]
        config: FeatureEngineeringConfig = {
            "use_ratios": True,
            "use_products": False,
            "use_log_transforms": False,
            "max_ratio_features": 100,
            "max_product_features": 0,
        }

        result = engineer_features(x, names, config)

        # Should only have original feature (no ratios possible with 1 feature)
        assert result["n_original"] == 1
        assert result["n_ratios"] == 0
        assert result["x"].shape == (2, 1)

    def test_single_feature_with_products_enabled(self) -> None:
        """Single feature with products enabled returns n_products=0."""
        x = _make_array(2, 1, (1.0, 2.0))
        names = ["A"]
        config: FeatureEngineeringConfig = {
            "use_ratios": False,
            "use_products": True,
            "use_log_transforms": False,
            "max_ratio_features": 0,
            "max_product_features": 100,
        }

        result = engineer_features(x, names, config)

        # Should only have original feature (no products possible with 1 feature)
        assert result["n_original"] == 1
        assert result["n_products"] == 0
        assert result["x"].shape == (2, 1)


class TestIntegration:
    """Integration tests with realistic data."""

    def test_many_features(self) -> None:
        """Handles many features efficiently."""
        n_samples = 100
        n_features = 50
        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((n_samples, n_features)).astype(np.float64)
        names = [f"X{i}" for i in range(n_features)]

        config: FeatureEngineeringConfig = {
            "use_ratios": True,
            "use_products": True,
            "use_log_transforms": True,
            "max_ratio_features": 200,
            "max_product_features": 100,
        }

        result = engineer_features(x, names, config)

        # Should have original + capped derived features
        assert result["n_original"] == 50
        assert result["n_ratios"] <= 200
        assert result["n_products"] <= 100
        assert result["n_log"] == 50

        # All values should be finite
        assert _all_finite(result["x"])

    def test_with_extreme_values(self) -> None:
        """Handles extreme values in data."""
        x = _make_array(2, 4, (1e-10, 1e10, 0.0, -1e5, 1e10, 1e-10, 1.0, 1e5))
        names = ["tiny", "huge", "zero", "neg"]

        config = get_feature_config_for_preset("full")
        result = engineer_features(x, names, config)

        # All values should be finite
        assert _all_finite(result["x"])
