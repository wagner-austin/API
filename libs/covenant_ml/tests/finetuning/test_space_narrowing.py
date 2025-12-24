"""Tests for search space narrowing utilities.

Tests cover:
- _narrow_int_range
- _narrow_float_range
- narrow_xgboost_space
- narrow_mlp_space
- narrow_lstm_space
- narrow_lightgbm_space
- narrow_search_space
"""

from __future__ import annotations

from covenant_ml.finetuning.space_narrowing import (
    narrow_lightgbm_space,
    narrow_lstm_space,
    narrow_mlp_space,
    narrow_search_space,
    narrow_xgboost_space,
)
from covenant_ml.optimizer.types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    CategoricalStringSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    XGBoostSearchSpace,
)

# =============================================================================
# Test Helpers
# =============================================================================


def _make_xgboost_space() -> XGBoostSearchSpace:
    """Create a standard XGBoost search space."""
    return XGBoostSearchSpace(
        max_depth=IntRangeSpec(param_type="int", low=3, high=10, log_scale=False),
        n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=10.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=10.0, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
    )


def _make_mlp_space() -> MLPSearchSpace:
    """Create a standard MLP search space."""
    return MLPSearchSpace(
        n_layers=IntRangeSpec(param_type="int", low=1, high=5, log_scale=False),
        hidden_size=IntRangeSpec(param_type="int", low=32, high=256, log_scale=False),
        batch_size=IntRangeSpec(param_type="int", low=16, high=128, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.0001, high=0.01, log_scale=True),
        dropout=FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False),
    )


def _make_lstm_space() -> LSTMSearchSpace:
    """Create a standard LSTM search space."""
    return LSTMSearchSpace(
        hidden_size=IntRangeSpec(param_type="int", low=32, high=256, log_scale=False),
        num_layers=IntRangeSpec(param_type="int", low=1, high=4, log_scale=False),
        batch_size=IntRangeSpec(param_type="int", low=16, high=128, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.0001, high=0.01, log_scale=True),
        dropout=FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False),
    )


def _make_lightgbm_space() -> LightGBMSearchSpace:
    """Create a standard LightGBM search space."""
    return LightGBMSearchSpace(
        n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
        num_leaves=IntRangeSpec(param_type="int", low=15, high=127, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=10.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=10.0, log_scale=True),
    )


# =============================================================================
# XGBoost Space Narrowing Tests
# =============================================================================


class TestNarrowXGBoostSpace:
    """Tests for narrow_xgboost_space."""

    def test_narrowing_reduces_range(self) -> None:
        """Narrowing reduces parameter ranges."""
        space = _make_xgboost_space()
        best_int = SampledIntParams(max_depth=6, n_estimators=100)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        best_string = SampledStringParams()

        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.5)

        # Check that ranges are narrower - use discriminated union
        orig_md = space["max_depth"]
        new_md = narrowed["max_depth"]
        if orig_md["param_type"] == "int" and new_md["param_type"] == "int":
            assert new_md["high"] - new_md["low"] <= orig_md["high"] - orig_md["low"]

    def test_narrowing_centered_on_best(self) -> None:
        """Narrowed range is centered on best value."""
        space = _make_xgboost_space()
        best_int = SampledIntParams(max_depth=7, n_estimators=150)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        best_string = SampledStringParams()

        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.5)

        # Best value should be within narrowed range - use discriminated union
        md = narrowed["max_depth"]
        ne = narrowed["n_estimators"]
        if md["param_type"] == "int" and ne["param_type"] == "int":
            assert md["low"] <= 7 <= md["high"]
            assert ne["low"] <= 150 <= ne["high"]

    def test_with_booster_string_param(self) -> None:
        """Handles booster string parameter."""
        space = _make_xgboost_space()
        space["booster"] = CategoricalStringSpec(
            param_type="categorical_str",
            choices=("gbtree", "dart"),
        )

        best_int = SampledIntParams(max_depth=6, n_estimators=100)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        best_string = SampledStringParams(booster="gbtree")

        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.5)

        # Booster should be fixed to best value
        assert narrowed["booster"]["choices"] == ("gbtree",)

    def test_with_dart_params(self) -> None:
        """Handles DART-specific parameters."""
        space = _make_xgboost_space()
        space["booster"] = CategoricalStringSpec(
            param_type="categorical_str",
            choices=("gbtree", "dart"),
        )
        space["rate_drop"] = FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False)
        space["skip_drop"] = FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False)

        best_int = SampledIntParams(max_depth=6, n_estimators=100)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            rate_drop=0.2,
            skip_drop=0.3,
        )
        best_string = SampledStringParams(booster="dart")

        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.5)

        assert "rate_drop" in narrowed
        assert "skip_drop" in narrowed


# =============================================================================
# MLP Space Narrowing Tests
# =============================================================================


class TestNarrowMLPSpace:
    """Tests for narrow_mlp_space."""

    def test_narrowing_reduces_range(self) -> None:
        """Narrowing reduces parameter ranges."""
        space = _make_mlp_space()
        best_int = SampledIntParams(n_layers=2, hidden_size=128, batch_size=32)
        best_float = SampledFloatParams(learning_rate=0.001, dropout=0.2)

        narrowed = narrow_mlp_space(space, best_int, best_float, 0.5)

        # Check ranges are narrower - use discriminated union
        orig_hs = space["hidden_size"]
        new_hs = narrowed["hidden_size"]
        if orig_hs["param_type"] == "int" and new_hs["param_type"] == "int":
            original_range = orig_hs["high"] - orig_hs["low"]
            new_range = new_hs["high"] - new_hs["low"]
            assert new_range <= original_range

    def test_narrowing_centered_on_best(self) -> None:
        """Narrowed range is centered on best value."""
        space = _make_mlp_space()
        best_int = SampledIntParams(n_layers=3, hidden_size=128, batch_size=64)
        best_float = SampledFloatParams(learning_rate=0.001, dropout=0.2)

        narrowed = narrow_mlp_space(space, best_int, best_float, 0.5)

        # Best values should be within narrowed ranges - use discriminated union
        nl = narrowed["n_layers"]
        hs = narrowed["hidden_size"]
        if nl["param_type"] == "int" and hs["param_type"] == "int":
            assert nl["low"] <= 3 <= nl["high"]
            assert hs["low"] <= 128 <= hs["high"]


# =============================================================================
# LSTM Space Narrowing Tests
# =============================================================================


class TestNarrowLSTMSpace:
    """Tests for narrow_lstm_space."""

    def test_narrowing_reduces_range(self) -> None:
        """Narrowing reduces parameter ranges."""
        space = _make_lstm_space()
        best_int = SampledIntParams(hidden_size=128, num_layers=2, batch_size=32)
        best_float = SampledFloatParams(learning_rate=0.001, dropout=0.2)

        narrowed = narrow_lstm_space(space, best_int, best_float, 0.5)

        # Use discriminated union narrowing
        orig_hs = space["hidden_size"]
        new_hs = narrowed["hidden_size"]
        if orig_hs["param_type"] == "int" and new_hs["param_type"] == "int":
            original_range = orig_hs["high"] - orig_hs["low"]
            new_range = new_hs["high"] - new_hs["low"]
            assert new_range <= original_range

    def test_narrowing_centered_on_best(self) -> None:
        """Narrowed range is centered on best value."""
        space = _make_lstm_space()
        best_int = SampledIntParams(hidden_size=128, num_layers=2, batch_size=64)
        best_float = SampledFloatParams(learning_rate=0.001, dropout=0.25)

        narrowed = narrow_lstm_space(space, best_int, best_float, 0.5)

        # Use discriminated union narrowing
        hs = narrowed["hidden_size"]
        nl = narrowed["num_layers"]
        if hs["param_type"] == "int" and nl["param_type"] == "int":
            assert hs["low"] <= 128 <= hs["high"]
            assert nl["low"] <= 2 <= nl["high"]


# =============================================================================
# LightGBM Space Narrowing Tests
# =============================================================================


class TestNarrowLightGBMSpace:
    """Tests for narrow_lightgbm_space."""

    def test_narrowing_reduces_range(self) -> None:
        """Narrowing reduces parameter ranges."""
        space = _make_lightgbm_space()
        best_int = SampledIntParams(n_estimators=100, num_leaves=31)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        best_string = SampledStringParams()

        narrowed = narrow_lightgbm_space(space, best_int, best_float, best_string, 0.5)

        # Use discriminated union narrowing
        orig_nl = space["num_leaves"]
        new_nl = narrowed["num_leaves"]
        if orig_nl["param_type"] == "int" and new_nl["param_type"] == "int":
            original_range = orig_nl["high"] - orig_nl["low"]
            new_range = new_nl["high"] - new_nl["low"]
            assert new_range <= original_range

    def test_with_boosting_type_param(self) -> None:
        """Handles boosting_type string parameter."""
        space = _make_lightgbm_space()
        space["boosting_type"] = CategoricalStringSpec(
            param_type="categorical_str",
            choices=("gbdt", "dart", "goss"),
        )

        best_int = SampledIntParams(n_estimators=100, num_leaves=31)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        best_string = SampledStringParams(boosting_type="gbdt")

        narrowed = narrow_lightgbm_space(space, best_int, best_float, best_string, 0.5)

        assert narrowed["boosting_type"]["choices"] == ("gbdt",)

    def test_with_dart_params(self) -> None:
        """Handles DART-specific parameters for LightGBM."""
        space = _make_lightgbm_space()
        space["boosting_type"] = CategoricalStringSpec(
            param_type="categorical_str",
            choices=("gbdt", "dart"),
        )
        space["drop_rate"] = FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False)
        space["skip_drop"] = FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False)
        space["feature_fraction"] = FloatRangeSpec(
            param_type="float", low=0.5, high=1.0, log_scale=False
        )

        best_int = SampledIntParams(n_estimators=100, num_leaves=31)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            drop_rate=0.2,
            skip_drop=0.3,
            feature_fraction=0.8,
        )
        best_string = SampledStringParams(boosting_type="dart")

        narrowed = narrow_lightgbm_space(space, best_int, best_float, best_string, 0.5)

        assert "drop_rate" in narrowed
        assert "skip_drop" in narrowed
        assert "feature_fraction" in narrowed


# =============================================================================
# Generic Space Narrowing Tests
# =============================================================================


class TestNarrowSearchSpace:
    """Tests for narrow_search_space generic function."""

    def test_xgboost_space(self) -> None:
        """Correctly identifies and narrows XGBoost space."""
        space = _make_xgboost_space()
        best_int = SampledIntParams(max_depth=6, n_estimators=100)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        best_string = SampledStringParams()

        narrowed = narrow_search_space(space, best_int, best_float, best_string, 0.5)

        assert "max_depth" in narrowed

    def test_mlp_space(self) -> None:
        """Correctly identifies and narrows MLP space."""
        space = _make_mlp_space()
        best_int = SampledIntParams(n_layers=2, hidden_size=128, batch_size=32)
        best_float = SampledFloatParams(learning_rate=0.001, dropout=0.2)
        best_string = SampledStringParams()

        narrowed = narrow_search_space(space, best_int, best_float, best_string, 0.5)

        assert "n_layers" in narrowed

    def test_lstm_space(self) -> None:
        """Correctly identifies and narrows LSTM space."""
        space = _make_lstm_space()
        best_int = SampledIntParams(hidden_size=128, num_layers=2, batch_size=32)
        best_float = SampledFloatParams(learning_rate=0.001, dropout=0.2)
        best_string = SampledStringParams()

        narrowed = narrow_search_space(space, best_int, best_float, best_string, 0.5)

        assert "num_layers" in narrowed

    def test_lightgbm_space(self) -> None:
        """Correctly identifies and narrows LightGBM space."""
        space = _make_lightgbm_space()
        best_int = SampledIntParams(n_estimators=100, num_leaves=31)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        best_string = SampledStringParams()

        narrowed = narrow_search_space(space, best_int, best_float, best_string, 0.5)

        assert "num_leaves" in narrowed


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in space narrowing."""

    def test_categorical_int_unchanged(self) -> None:
        """Categorical int specs are not narrowed."""
        space = _make_xgboost_space()
        # Replace max_depth with categorical
        space["max_depth"] = CategoricalIntSpec(
            param_type="categorical_int",
            choices=(3, 5, 7, 9),
        )

        best_int = SampledIntParams(max_depth=5, n_estimators=100)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        best_string = SampledStringParams()

        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.5)

        # Categorical should remain unchanged
        assert narrowed["max_depth"]["param_type"] == "categorical_int"
        assert narrowed["max_depth"]["choices"] == (3, 5, 7, 9)

    def test_categorical_float_unchanged(self) -> None:
        """Categorical float specs are not narrowed."""
        space = _make_xgboost_space()
        # Replace learning_rate with categorical
        space["learning_rate"] = CategoricalFloatSpec(
            param_type="categorical_float",
            choices=(0.01, 0.05, 0.1, 0.2),
        )

        best_int = SampledIntParams(max_depth=6, n_estimators=100)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        best_string = SampledStringParams()

        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.5)

        # Categorical should remain unchanged
        assert narrowed["learning_rate"]["param_type"] == "categorical_float"
        assert narrowed["learning_rate"]["choices"] == (0.01, 0.05, 0.1, 0.2)

    def test_very_tight_radius(self) -> None:
        """Very tight radius still produces valid range."""
        space = _make_xgboost_space()
        best_int = SampledIntParams(max_depth=6, n_estimators=100)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        best_string = SampledStringParams()

        # Very tight radius
        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.01)

        # Should still have valid ranges - use discriminated union narrowing
        max_depth_spec = narrowed["max_depth"]
        lr_spec = narrowed["learning_rate"]
        if max_depth_spec["param_type"] == "int" and lr_spec["param_type"] == "float":
            assert max_depth_spec["low"] <= max_depth_spec["high"]
            assert lr_spec["low"] <= lr_spec["high"]

    def test_best_at_boundary(self) -> None:
        """Works when best value is at boundary."""
        space = _make_xgboost_space()
        # Best at low boundary
        best_int = SampledIntParams(max_depth=3, n_estimators=50)
        best_float = SampledFloatParams(
            learning_rate=0.01,
            reg_alpha=0.001,
            reg_lambda=0.001,
            subsample=0.5,
            colsample_bytree=0.5,
        )
        best_string = SampledStringParams()

        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.5)

        # Should respect original bounds - use discriminated union narrowing
        narrowed_depth = narrowed["max_depth"]
        orig_depth = space["max_depth"]
        narrowed_lr = narrowed["learning_rate"]
        orig_lr = space["learning_rate"]
        if (
            narrowed_depth["param_type"] == "int"
            and orig_depth["param_type"] == "int"
            and narrowed_lr["param_type"] == "float"
            and orig_lr["param_type"] == "float"
        ):
            assert narrowed_depth["low"] >= orig_depth["low"]
            assert narrowed_lr["low"] >= orig_lr["low"]

    def test_int_range_fallback_when_too_narrow(self) -> None:
        """Int range uses fallback when narrowing would be invalid."""
        # Create a very narrow int range where narrowing would collapse
        space = XGBoostSearchSpace(
            max_depth=IntRangeSpec(param_type="int", low=5, high=6, log_scale=False),
            n_estimators=IntRangeSpec(param_type="int", low=100, high=100, log_scale=False),
            learning_rate=FloatRangeSpec(param_type="float", low=0.1, high=0.1, log_scale=False),
            reg_alpha=FloatRangeSpec(param_type="float", low=0.1, high=0.1, log_scale=False),
            reg_lambda=FloatRangeSpec(param_type="float", low=0.1, high=0.1, log_scale=False),
            subsample=FloatRangeSpec(param_type="float", low=0.8, high=0.8, log_scale=False),
            colsample_bytree=FloatRangeSpec(param_type="float", low=0.8, high=0.8, log_scale=False),
        )
        best_int = SampledIntParams(max_depth=5, n_estimators=100)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        best_string = SampledStringParams()

        # Extremely tight radius to trigger fallback
        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.001)

        # Should still have valid int range
        md = narrowed["max_depth"]
        if md["param_type"] == "int":
            assert md["low"] <= md["high"]

    def test_float_range_fallback_when_too_narrow(self) -> None:
        """Float range uses epsilon fallback when narrowing would be invalid."""
        # Create very narrow float ranges
        space = XGBoostSearchSpace(
            max_depth=IntRangeSpec(param_type="int", low=5, high=5, log_scale=False),
            n_estimators=IntRangeSpec(param_type="int", low=100, high=100, log_scale=False),
            learning_rate=FloatRangeSpec(param_type="float", low=0.1, high=0.1001, log_scale=False),
            reg_alpha=FloatRangeSpec(param_type="float", low=0.1, high=0.1001, log_scale=False),
            reg_lambda=FloatRangeSpec(param_type="float", low=0.1, high=0.1001, log_scale=False),
            subsample=FloatRangeSpec(param_type="float", low=0.8, high=0.8001, log_scale=False),
            colsample_bytree=FloatRangeSpec(
                param_type="float", low=0.8, high=0.8001, log_scale=False
            ),
        )
        best_int = SampledIntParams(max_depth=5, n_estimators=100)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        best_string = SampledStringParams()

        # Extremely tight radius
        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.0001)

        # Should still have valid float ranges
        lr = narrowed["learning_rate"]
        if lr["param_type"] == "float":
            assert lr["low"] <= lr["high"]

    def test_xgboost_dart_params_narrowed(self) -> None:
        """XGBoost DART params (rate_drop, skip_drop) are narrowed."""
        space = XGBoostSearchSpace(
            max_depth=IntRangeSpec(param_type="int", low=3, high=10, log_scale=False),
            n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
            learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
            reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
            reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
            subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
            colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
            booster=CategoricalStringSpec(param_type="categorical_str", choices=("gbtree", "dart")),
            rate_drop=FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False),
            skip_drop=FloatRangeSpec(param_type="float", low=0.0, high=0.7, log_scale=False),
        )
        best_int = SampledIntParams(max_depth=6, n_estimators=100)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            rate_drop=0.1,
            skip_drop=0.3,
        )
        best_string = SampledStringParams(booster="dart")

        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.5)

        # Should have narrowed DART params with proper ranges
        rate_drop_spec = narrowed["rate_drop"]
        assert rate_drop_spec["param_type"] == "float"
        skip_drop_spec = narrowed["skip_drop"]
        assert skip_drop_spec["param_type"] == "float"
        # Booster should be fixed to "dart"
        assert narrowed["booster"]["choices"] == ("dart",)

    def test_lightgbm_dart_params_narrowed(self) -> None:
        """LightGBM DART params (drop_rate, skip_drop, feature_fraction) are narrowed."""
        space = LightGBMSearchSpace(
            n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
            num_leaves=IntRangeSpec(param_type="int", low=10, high=100, log_scale=False),
            learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
            subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
            colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
            reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
            reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
            boosting_type=CategoricalStringSpec(
                param_type="categorical_str", choices=("gbdt", "dart")
            ),
            drop_rate=FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False),
            skip_drop=FloatRangeSpec(param_type="float", low=0.0, high=0.7, log_scale=False),
            feature_fraction=FloatRangeSpec(
                param_type="float", low=0.02, high=0.2, log_scale=False
            ),
        )
        best_int = SampledIntParams(n_estimators=100, num_leaves=31)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            drop_rate=0.1,
            skip_drop=0.3,
            feature_fraction=0.05,
        )
        best_string = SampledStringParams(boosting_type="dart")

        narrowed = narrow_lightgbm_space(space, best_int, best_float, best_string, 0.5)

        # Should have narrowed DART params with proper ranges
        drop_rate_spec = narrowed["drop_rate"]
        assert drop_rate_spec["param_type"] == "float"
        skip_drop_spec = narrowed["skip_drop"]
        assert skip_drop_spec["param_type"] == "float"
        feature_fraction_spec = narrowed["feature_fraction"]
        assert feature_fraction_spec["param_type"] == "float"
        # boosting_type should be fixed to "dart"
        assert narrowed["boosting_type"]["choices"] == ("dart",)

    def test_xgboost_dart_without_dart_params_in_space(self) -> None:
        """XGBoost DART without DART params in search space."""
        # Space has booster but no rate_drop or skip_drop
        space = XGBoostSearchSpace(
            max_depth=IntRangeSpec(param_type="int", low=3, high=10, log_scale=False),
            n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
            learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
            reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
            reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
            subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
            colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
            booster=CategoricalStringSpec(param_type="categorical_str", choices=("gbtree", "dart")),
            # No rate_drop or skip_drop
        )
        best_int = SampledIntParams(max_depth=6, n_estimators=100)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        best_string = SampledStringParams(booster="dart")

        narrowed = narrow_xgboost_space(space, best_int, best_float, best_string, 0.5)

        # Should still have booster fixed to dart
        assert narrowed["booster"]["choices"] == ("dart",)
        # Should NOT have rate_drop or skip_drop
        assert "rate_drop" not in narrowed
        assert "skip_drop" not in narrowed

    def test_lightgbm_dart_without_dart_params_in_space(self) -> None:
        """LightGBM DART without DART params in search space."""
        # Space has boosting_type but no DART-specific params
        space = LightGBMSearchSpace(
            n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
            num_leaves=IntRangeSpec(param_type="int", low=10, high=100, log_scale=False),
            learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
            subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
            colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
            reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
            reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
            boosting_type=CategoricalStringSpec(
                param_type="categorical_str", choices=("gbdt", "dart")
            ),
            # No drop_rate, skip_drop, or feature_fraction
        )
        best_int = SampledIntParams(n_estimators=100, num_leaves=31)
        best_float = SampledFloatParams(
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
        )
        best_string = SampledStringParams(boosting_type="dart")

        narrowed = narrow_lightgbm_space(space, best_int, best_float, best_string, 0.5)

        # Should still have boosting_type fixed to dart
        assert narrowed["boosting_type"]["choices"] == ("dart",)
        # Should NOT have DART params
        assert "drop_rate" not in narrowed
        assert "skip_drop" not in narrowed
        assert "feature_fraction" not in narrowed
