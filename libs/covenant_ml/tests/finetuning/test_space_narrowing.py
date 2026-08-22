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
    CategoricalStringSpec,
    FloatRangeSpec,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from tests.finetuning._narrowing_fixtures import (
    _make_lightgbm_space,
    _make_lstm_space,
    _make_mlp_space,
    _make_xgboost_space,
)


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
