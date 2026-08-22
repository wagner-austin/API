"""Tests for worker/optimize_regression_types.py encode/decode round-trip validation.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
All encode/decode/require functions are round-trip tested.
"""

from __future__ import annotations

import pytest
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from platform_core.json_utils import JSONTypeError, JSONValue

from covenant_radar_api.worker.optimize_regression_results import (
    UnifiedRegressionOptimizationResult,
    decode_unified_regression_optimization_result,
    encode_unified_regression_optimization_result,
    require_unified_regression_optimization_result,
)
from tests._optimize_regression_types_fixtures import (
    _make_regression_optimization_result,
)


class TestRegressionOptimizationResultEncode:
    """Tests for encode_unified_regression_optimization_result."""

    def test_round_trip(self) -> None:
        """Encode then decode produces identical result."""
        original = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(original)
        decoded = decode_unified_regression_optimization_result(encoded)
        assert decoded == original

    def test_round_trip_lightgbm_reg(self) -> None:
        """Round-trip works for lightgbm_reg backend."""
        original = _make_regression_optimization_result(backend="lightgbm_reg")
        encoded = encode_unified_regression_optimization_result(original)
        decoded = decode_unified_regression_optimization_result(encoded)
        assert decoded == original

    def test_encode_contains_all_fields(self) -> None:
        """Encoded result contains all expected keys."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        expected_keys = {
            "backend",
            "status",
            "dataset",
            "n_samples",
            "n_features",
            "feature_preset",
            "n_trials_complete",
            "n_trials_pruned",
            "n_trials_failed",
            "best_trial_number",
            "best_value",
            "best_int_params",
            "best_float_params",
            "best_string_params",
            "duration_seconds",
        }
        assert set(encoded.keys()) == expected_keys

    def test_round_trip_with_dart_string_params(self) -> None:
        """Round-trip works with DART boosting string params."""
        result = UnifiedRegressionOptimizationResult(
            backend="lightgbm_reg",
            status="complete",
            dataset="us_bankruptcy",
            n_samples=500,
            n_features=10,
            feature_preset="log_only",
            n_trials_complete=25,
            n_trials_pruned=2,
            n_trials_failed=1,
            best_trial_number=20,
            best_value=-0.456,
            best_int_params=SampledIntParams(n_estimators=50, num_leaves=31),
            best_float_params=SampledFloatParams(
                learning_rate=0.1,
                reg_alpha=0.0,
                reg_lambda=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                drop_rate=0.1,
            ),
            best_string_params=SampledStringParams(boosting_type="dart"),
            duration_seconds=60.0,
        )
        encoded = encode_unified_regression_optimization_result(result)
        decoded = decode_unified_regression_optimization_result(encoded)
        assert decoded == result

    def test_round_trip_with_xgb_booster_params(self) -> None:
        """Round-trip works with XGBoost DART booster string params."""
        result = UnifiedRegressionOptimizationResult(
            backend="xgboost_reg",
            status="complete",
            dataset="us_bankruptcy",
            n_samples=500,
            n_features=10,
            feature_preset="none",
            n_trials_complete=25,
            n_trials_pruned=0,
            n_trials_failed=0,
            best_trial_number=15,
            best_value=-0.789,
            best_int_params=SampledIntParams(max_depth=4, n_estimators=80),
            best_float_params=SampledFloatParams(
                learning_rate=0.05,
                reg_alpha=0.0,
                reg_lambda=1.0,
                subsample=0.9,
                colsample_bytree=0.7,
                rate_drop=0.1,
                skip_drop=0.5,
            ),
            best_string_params=SampledStringParams(booster="dart"),
            duration_seconds=90.0,
        )
        encoded = encode_unified_regression_optimization_result(result)
        decoded = decode_unified_regression_optimization_result(encoded)
        assert decoded == result

    def test_round_trip_with_all_int_params(self) -> None:
        """Round-trip works with all regression-relevant int params."""
        result = UnifiedRegressionOptimizationResult(
            backend="lightgbm_reg",
            status="complete",
            dataset="us_bankruptcy",
            n_samples=500,
            n_features=10,
            feature_preset="none",
            n_trials_complete=25,
            n_trials_pruned=0,
            n_trials_failed=0,
            best_trial_number=15,
            best_value=-0.5,
            best_int_params=SampledIntParams(
                max_depth=6,
                n_estimators=100,
                num_leaves=31,
                min_child_samples=20,
                min_samples_split=5,
                min_samples_leaf=2,
            ),
            best_float_params=SampledFloatParams(learning_rate=0.1),
            best_string_params=SampledStringParams(),
            duration_seconds=30.0,
        )
        encoded = encode_unified_regression_optimization_result(result)
        decoded = decode_unified_regression_optimization_result(encoded)
        assert decoded == result

    def test_round_trip_with_all_float_params(self) -> None:
        """Round-trip works with all regression-relevant float params."""
        result = UnifiedRegressionOptimizationResult(
            backend="xgboost_reg",
            status="complete",
            dataset="us_bankruptcy",
            n_samples=500,
            n_features=10,
            feature_preset="full",
            n_trials_complete=25,
            n_trials_pruned=0,
            n_trials_failed=0,
            best_trial_number=15,
            best_value=-0.3,
            best_int_params=SampledIntParams(max_depth=4),
            best_float_params=SampledFloatParams(
                learning_rate=0.05,
                reg_alpha=0.1,
                reg_lambda=1.5,
                subsample=0.9,
                colsample_bytree=0.7,
                drop_rate=0.1,
                skip_drop=0.5,
                rate_drop=0.2,
                feature_fraction=0.8,
            ),
            best_string_params=SampledStringParams(),
            duration_seconds=45.0,
        )
        encoded = encode_unified_regression_optimization_result(result)
        decoded = decode_unified_regression_optimization_result(encoded)
        assert decoded == result

    def test_round_trip_empty_sampled_params(self) -> None:
        """Round-trip works with empty sampled params dicts."""
        result = UnifiedRegressionOptimizationResult(
            backend="xgboost_reg",
            status="complete",
            dataset="us_bankruptcy",
            n_samples=100,
            n_features=5,
            feature_preset="none",
            n_trials_complete=10,
            n_trials_pruned=0,
            n_trials_failed=0,
            best_trial_number=5,
            best_value=-1.0,
            best_int_params=SampledIntParams(),
            best_float_params=SampledFloatParams(),
            best_string_params=SampledStringParams(),
            duration_seconds=10.0,
        )
        encoded = encode_unified_regression_optimization_result(result)
        decoded = decode_unified_regression_optimization_result(encoded)
        assert decoded == result


class TestRegressionOptimizationResultDecode:
    """Tests for decode_unified_regression_optimization_result validation."""

    def test_invalid_status_raises(self) -> None:
        """Raises JSONTypeError when status is not 'complete'."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        encoded["status"] = "failed"
        with pytest.raises(JSONTypeError, match=r"status.*must be 'complete'"):
            decode_unified_regression_optimization_result(encoded)

    def test_missing_status_raises(self) -> None:
        """Raises JSONTypeError when status is missing."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        del encoded["status"]
        with pytest.raises(JSONTypeError, match="Missing required field 'status'"):
            decode_unified_regression_optimization_result(encoded)

    def test_invalid_backend_raises(self) -> None:
        """Raises JSONTypeError when backend is not a valid regressor."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        encoded["backend"] = "xgboost"
        with pytest.raises(JSONTypeError, match="must be one of"):
            decode_unified_regression_optimization_result(encoded)

    def test_missing_best_int_params_raises(self) -> None:
        """Raises JSONTypeError when best_int_params is missing."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        del encoded["best_int_params"]
        with pytest.raises(JSONTypeError, match="Missing required field 'best_int_params'"):
            decode_unified_regression_optimization_result(encoded)

    def test_non_object_best_int_params_raises(self) -> None:
        """Raises JSONTypeError when best_int_params is not a dict."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        encoded["best_int_params"] = "not a dict"
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_unified_regression_optimization_result(encoded)

    def test_non_int_in_int_params_raises(self) -> None:
        """Raises JSONTypeError when int param value is not an int."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        encoded["best_int_params"] = {"max_depth": "six"}
        with pytest.raises(JSONTypeError, match="must be an integer"):
            decode_unified_regression_optimization_result(encoded)

    def test_non_float_in_float_params_raises(self) -> None:
        """Raises JSONTypeError when float param value is not a number."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        encoded["best_float_params"] = {"learning_rate": "fast"}
        with pytest.raises(JSONTypeError, match="must be a number"):
            decode_unified_regression_optimization_result(encoded)

    def test_non_str_in_string_params_raises(self) -> None:
        """Raises JSONTypeError when string param value is not a string."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        encoded["best_string_params"] = {"boosting_type": 42}
        with pytest.raises(JSONTypeError, match="must be a string"):
            decode_unified_regression_optimization_result(encoded)

    def test_non_number_best_value_raises(self) -> None:
        """Raises JSONTypeError when best_value is not a number."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        encoded["best_value"] = "not a number"
        with pytest.raises(JSONTypeError, match="must be a number"):
            decode_unified_regression_optimization_result(encoded)

    def test_missing_n_samples_raises(self) -> None:
        """Raises JSONTypeError when n_samples is missing."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        del encoded["n_samples"]
        with pytest.raises(JSONTypeError, match="Missing required field 'n_samples'"):
            decode_unified_regression_optimization_result(encoded)

    def test_missing_duration_seconds_raises(self) -> None:
        """Raises JSONTypeError when duration_seconds (float field) is missing."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        del encoded["duration_seconds"]
        with pytest.raises(JSONTypeError, match="Missing required field 'duration_seconds'"):
            decode_unified_regression_optimization_result(encoded)

    def test_missing_feature_preset_raises(self) -> None:
        """Raises JSONTypeError when feature_preset is missing."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        del encoded["feature_preset"]
        with pytest.raises(JSONTypeError, match="Missing required field 'feature_preset'"):
            decode_unified_regression_optimization_result(encoded)

    def test_ratios_only_feature_preset_decodes(self) -> None:
        """Decodes feature_preset "ratios_only" to the literal."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        encoded["feature_preset"] = "ratios_only"
        decoded = decode_unified_regression_optimization_result(encoded)
        assert decoded["feature_preset"] == "ratios_only"

    def test_invalid_feature_preset_raises(self) -> None:
        """Raises JSONTypeError when feature_preset is not a regression preset."""
        result = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(result)
        encoded["feature_preset"] = "temporal"
        with pytest.raises(
            JSONTypeError, match="must be one of: none, log_only, ratios_only, full"
        ):
            decode_unified_regression_optimization_result(encoded)


class TestRegressionOptimizationResultRequire:
    """Tests for require_unified_regression_optimization_result."""

    def test_valid_dict(self) -> None:
        """Accepts a valid dict."""
        original = _make_regression_optimization_result()
        encoded = encode_unified_regression_optimization_result(original)
        raw_value: JSONValue = encoded
        decoded = require_unified_regression_optimization_result(raw_value)
        assert decoded == original

    def test_non_dict_raises(self) -> None:
        """Raises JSONTypeError for non-dict input."""
        raw_value: JSONValue = "not a dict"
        with pytest.raises(JSONTypeError, match="Expected a JSON object"):
            require_unified_regression_optimization_result(raw_value)

    def test_list_raises(self) -> None:
        """Raises JSONTypeError for list input."""
        raw_value: JSONValue = [1, 2, 3]
        with pytest.raises(JSONTypeError, match="Expected a JSON object"):
            require_unified_regression_optimization_result(raw_value)

    def test_int_raises(self) -> None:
        """Raises JSONTypeError for int input."""
        raw_value: JSONValue = 42
        with pytest.raises(JSONTypeError, match="Expected a JSON object"):
            require_unified_regression_optimization_result(raw_value)
