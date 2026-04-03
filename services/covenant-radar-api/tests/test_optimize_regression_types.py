"""Tests for worker/optimize_regression_types.py encode/decode round-trip validation.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
All encode/decode/require functions are round-trip tested.
"""

from __future__ import annotations

import pytest
from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.types import RegressorBackendName
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from covenant_radar_api.worker.optimize_regression_types import (
    UnifiedRegressionOptimizationResult,
    UnifiedRegressionOptimizeParseResult,
    decode_unified_regression_optimization_result,
    decode_unified_regression_optimize_parse_result,
    encode_unified_regression_optimization_result,
    encode_unified_regression_optimize_parse_result,
    require_unified_regression_optimization_result,
    require_unified_regression_optimize_parse_result,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_regression_parse_result(
    backend: RegressorBackendName = "xgboost_reg",
    dataset: str = "us_bankruptcy",
) -> UnifiedRegressionOptimizeParseResult:
    """Create a valid UnifiedRegressionOptimizeParseResult for testing.

    Args:
        backend: Regressor backend name.
        dataset: Dataset name.

    Returns:
        Valid UnifiedRegressionOptimizeParseResult.
    """
    return UnifiedRegressionOptimizeParseResult(
        backend=backend,
        dataset=dataset,
        n_trials=50,
        timeout_seconds=None,
        device="cpu",
        feature_preset="none",
        random_state=42,
        early_stopping_rounds=10,
        n_jobs=-1,
        precision="fp32",
        nn_optimizer="adamw",
        n_epochs=50,
        early_stopping_patience=10,
        sequence_length=5,
        bidirectional=False,
    )


def _make_regression_optimization_result(
    backend: RegressorBackendName = "xgboost_reg",
) -> UnifiedRegressionOptimizationResult:
    """Create a valid UnifiedRegressionOptimizationResult for testing.

    Args:
        backend: Regressor backend name.

    Returns:
        Valid UnifiedRegressionOptimizationResult.
    """
    return UnifiedRegressionOptimizationResult(
        backend=backend,
        status="complete",
        dataset="us_bankruptcy",
        n_samples=1000,
        n_features=18,
        feature_preset="none",
        n_trials_complete=50,
        n_trials_pruned=5,
        n_trials_failed=0,
        best_trial_number=37,
        best_value=-0.123,
        best_int_params=SampledIntParams(max_depth=6, n_estimators=100),
        best_float_params=SampledFloatParams(
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=1.5,
            subsample=0.9,
            colsample_bytree=0.7,
        ),
        best_string_params=SampledStringParams(),
        duration_seconds=120.5,
    )


# =============================================================================
# Tests: UnifiedRegressionOptimizeParseResult encode/decode
# =============================================================================


class TestRegressionParseResultEncode:
    """Tests for encode_unified_regression_optimize_parse_result."""

    def test_round_trip(self) -> None:
        """Encode then decode produces identical result."""
        original = _make_regression_parse_result()
        encoded = encode_unified_regression_optimize_parse_result(original)
        decoded = decode_unified_regression_optimize_parse_result(encoded)
        assert decoded == original

    def test_round_trip_lightgbm_reg(self) -> None:
        """Round-trip works for lightgbm_reg backend."""
        original = _make_regression_parse_result(backend="lightgbm_reg")
        encoded = encode_unified_regression_optimize_parse_result(original)
        decoded = decode_unified_regression_optimize_parse_result(encoded)
        assert decoded == original

    def test_round_trip_with_timeout(self) -> None:
        """Round-trip works with non-null timeout_seconds."""
        original = _make_regression_parse_result()
        original_with_timeout = UnifiedRegressionOptimizeParseResult(
            backend=original["backend"],
            dataset=original["dataset"],
            n_trials=original["n_trials"],
            timeout_seconds=300,
            device=original["device"],
            feature_preset=original["feature_preset"],
            random_state=original["random_state"],
            early_stopping_rounds=original["early_stopping_rounds"],
            n_jobs=original["n_jobs"],
            precision=original["precision"],
            nn_optimizer=original["nn_optimizer"],
            n_epochs=original["n_epochs"],
            early_stopping_patience=original["early_stopping_patience"],
            sequence_length=original["sequence_length"],
            bidirectional=original["bidirectional"],
        )
        encoded = encode_unified_regression_optimize_parse_result(original_with_timeout)
        decoded = decode_unified_regression_optimize_parse_result(encoded)
        assert decoded == original_with_timeout
        assert decoded["timeout_seconds"] == 300

    def test_encode_contains_all_fields(self) -> None:
        """Encoded result contains all expected keys."""
        result = _make_regression_parse_result()
        encoded = encode_unified_regression_optimize_parse_result(result)
        expected_keys = {
            "backend",
            "dataset",
            "n_trials",
            "timeout_seconds",
            "device",
            "feature_preset",
            "random_state",
            "early_stopping_rounds",
            "n_jobs",
            "precision",
            "nn_optimizer",
            "n_epochs",
            "early_stopping_patience",
            "sequence_length",
            "bidirectional",
        }
        assert set(encoded.keys()) == expected_keys

    def test_round_trip_all_backends(self) -> None:
        """Round-trip works for all regressor backends."""
        backends: list[RegressorBackendName] = [
            "xgboost_reg",
            "lightgbm_reg",
            "mlp_reg",
            "lstm_reg",
        ]
        for backend in backends:
            original = _make_regression_parse_result(backend=backend)
            encoded = encode_unified_regression_optimize_parse_result(original)
            decoded = decode_unified_regression_optimize_parse_result(encoded)
            assert decoded == original

    def test_round_trip_all_feature_presets(self) -> None:
        """Round-trip works for all feature presets."""
        presets: list[FeaturePreset] = ["none", "log_only", "ratios_only", "full"]
        for preset in presets:
            original = _make_regression_parse_result()
            original_with_preset = UnifiedRegressionOptimizeParseResult(
                backend=original["backend"],
                dataset=original["dataset"],
                n_trials=original["n_trials"],
                timeout_seconds=original["timeout_seconds"],
                device=original["device"],
                feature_preset=preset,
                random_state=original["random_state"],
                early_stopping_rounds=original["early_stopping_rounds"],
                n_jobs=original["n_jobs"],
                precision=original["precision"],
                nn_optimizer=original["nn_optimizer"],
                n_epochs=original["n_epochs"],
                early_stopping_patience=original["early_stopping_patience"],
                sequence_length=original["sequence_length"],
                bidirectional=original["bidirectional"],
            )
            encoded = encode_unified_regression_optimize_parse_result(original_with_preset)
            decoded = decode_unified_regression_optimize_parse_result(encoded)
            assert decoded == original_with_preset

    def test_round_trip_cuda_device(self) -> None:
        """Round-trip works with cuda device."""
        original = UnifiedRegressionOptimizeParseResult(
            backend="xgboost_reg",
            dataset="us_bankruptcy",
            n_trials=50,
            timeout_seconds=None,
            device="cuda",
            feature_preset="none",
            random_state=42,
            early_stopping_rounds=10,
            n_jobs=-1,
            precision="fp32",
            nn_optimizer="adamw",
            n_epochs=50,
            early_stopping_patience=10,
            sequence_length=5,
            bidirectional=False,
        )
        encoded = encode_unified_regression_optimize_parse_result(original)
        decoded = decode_unified_regression_optimize_parse_result(encoded)
        assert decoded == original
        assert decoded["device"] == "cuda"

    def test_round_trip_auto_device(self) -> None:
        """Round-trip works with auto device."""
        original = UnifiedRegressionOptimizeParseResult(
            backend="xgboost_reg",
            dataset="us_bankruptcy",
            n_trials=50,
            timeout_seconds=None,
            device="auto",
            feature_preset="none",
            random_state=42,
            early_stopping_rounds=10,
            n_jobs=-1,
            precision="fp32",
            nn_optimizer="adamw",
            n_epochs=50,
            early_stopping_patience=10,
            sequence_length=5,
            bidirectional=False,
        )
        encoded = encode_unified_regression_optimize_parse_result(original)
        decoded = decode_unified_regression_optimize_parse_result(encoded)
        assert decoded == original
        assert decoded["device"] == "auto"


class TestRegressionParseResultDecode:
    """Tests for decode_unified_regression_optimize_parse_result validation."""

    def test_missing_backend_raises(self) -> None:
        """Raises JSONTypeError when backend is missing."""
        raw: JSONObject = {
            "dataset": "us_bankruptcy",
            "n_trials": 50,
            "timeout_seconds": None,
            "device": "cpu",
            "feature_preset": "none",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
        }
        with pytest.raises(JSONTypeError, match="Missing required field 'backend'"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_invalid_backend_raises(self) -> None:
        """Raises JSONTypeError when backend is not a valid regressor backend."""
        raw: JSONObject = {
            "backend": "xgboost",
            "dataset": "us_bankruptcy",
            "n_trials": 50,
            "timeout_seconds": None,
            "device": "cpu",
            "feature_preset": "none",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
        }
        with pytest.raises(JSONTypeError, match="must be one of"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_non_string_backend_raises(self) -> None:
        """Raises JSONTypeError when backend is not a string."""
        raw: JSONObject = {
            "backend": 42,
            "dataset": "us_bankruptcy",
            "n_trials": 50,
            "timeout_seconds": None,
            "device": "cpu",
            "feature_preset": "none",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
        }
        with pytest.raises(JSONTypeError, match="must be a string"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_missing_dataset_raises(self) -> None:
        """Raises JSONTypeError when dataset is missing."""
        raw: JSONObject = {
            "backend": "xgboost_reg",
            "n_trials": 50,
            "timeout_seconds": None,
            "device": "cpu",
            "feature_preset": "none",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
        }
        with pytest.raises(JSONTypeError, match="Missing required field 'dataset'"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_invalid_device_raises(self) -> None:
        """Raises JSONTypeError when device is invalid."""
        raw: JSONObject = {
            "backend": "xgboost_reg",
            "dataset": "us_bankruptcy",
            "n_trials": 50,
            "timeout_seconds": None,
            "device": "tpu",
            "feature_preset": "none",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
        }
        with pytest.raises(JSONTypeError, match="device"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_missing_device_raises(self) -> None:
        """Raises JSONTypeError when device is missing."""
        raw: JSONObject = {
            "backend": "xgboost_reg",
            "dataset": "us_bankruptcy",
            "n_trials": 50,
            "timeout_seconds": None,
            "feature_preset": "none",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
        }
        with pytest.raises(JSONTypeError, match="Missing required field 'device'"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_invalid_feature_preset_raises(self) -> None:
        """Raises JSONTypeError when feature_preset is invalid."""
        raw: JSONObject = {
            "backend": "xgboost_reg",
            "dataset": "us_bankruptcy",
            "n_trials": 50,
            "timeout_seconds": None,
            "device": "cpu",
            "feature_preset": "mega",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
        }
        with pytest.raises(JSONTypeError, match="feature_preset"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_missing_feature_preset_raises(self) -> None:
        """Raises JSONTypeError when feature_preset is missing."""
        raw: JSONObject = {
            "backend": "xgboost_reg",
            "dataset": "us_bankruptcy",
            "n_trials": 50,
            "timeout_seconds": None,
            "device": "cpu",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
        }
        with pytest.raises(JSONTypeError, match="Missing required field 'feature_preset'"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_non_int_timeout_raises(self) -> None:
        """Raises JSONTypeError when timeout_seconds is not int or null."""
        raw: JSONObject = {
            "backend": "xgboost_reg",
            "dataset": "us_bankruptcy",
            "n_trials": 50,
            "timeout_seconds": "300",
            "device": "cpu",
            "feature_preset": "none",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
        }
        with pytest.raises(JSONTypeError, match="timeout_seconds"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_non_int_n_trials_raises(self) -> None:
        """Raises JSONTypeError when n_trials is not an integer."""
        raw: JSONObject = {
            "backend": "xgboost_reg",
            "dataset": "us_bankruptcy",
            "n_trials": "50",
            "timeout_seconds": None,
            "device": "cpu",
            "feature_preset": "none",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
        }
        with pytest.raises(JSONTypeError, match="n_trials"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_non_str_dataset_raises(self) -> None:
        """Raises JSONTypeError when dataset is not a string."""
        raw: JSONObject = {
            "backend": "xgboost_reg",
            "dataset": 123,
            "n_trials": 50,
            "timeout_seconds": None,
            "device": "cpu",
            "feature_preset": "none",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
        }
        with pytest.raises(JSONTypeError, match="must be a string"):
            decode_unified_regression_optimize_parse_result(raw)

    def _make_valid_raw(self) -> JSONObject:
        """Create a fully valid raw JSON object for decode tests.

        Returns:
            Complete valid JSON object for UnifiedRegressionOptimizeParseResult.
        """
        return {
            "backend": "xgboost_reg",
            "dataset": "us_bankruptcy",
            "n_trials": 50,
            "timeout_seconds": None,
            "device": "cpu",
            "feature_preset": "none",
            "random_state": 42,
            "early_stopping_rounds": 10,
            "n_jobs": -1,
            "precision": "fp32",
            "nn_optimizer": "adamw",
            "n_epochs": 50,
            "early_stopping_patience": 10,
            "sequence_length": 5,
            "bidirectional": False,
        }

    def test_missing_precision_raises(self) -> None:
        """Raises JSONTypeError when precision is missing."""
        raw = self._make_valid_raw()
        del raw["precision"]
        with pytest.raises(JSONTypeError, match="Missing required field 'precision'"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_invalid_precision_raises(self) -> None:
        """Raises JSONTypeError when precision is not a valid value."""
        raw = self._make_valid_raw()
        raw["precision"] = "fp64"
        with pytest.raises(JSONTypeError, match="precision"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_precision_fp16_accepted(self) -> None:
        """fp16 precision is accepted."""
        raw = self._make_valid_raw()
        raw["precision"] = "fp16"
        result = decode_unified_regression_optimize_parse_result(raw)
        assert result["precision"] == "fp16"

    def test_precision_bf16_accepted(self) -> None:
        """bf16 precision is accepted."""
        raw = self._make_valid_raw()
        raw["precision"] = "bf16"
        result = decode_unified_regression_optimize_parse_result(raw)
        assert result["precision"] == "bf16"

    def test_precision_auto_accepted(self) -> None:
        """auto precision is accepted."""
        raw = self._make_valid_raw()
        raw["precision"] = "auto"
        result = decode_unified_regression_optimize_parse_result(raw)
        assert result["precision"] == "auto"

    def test_missing_nn_optimizer_raises(self) -> None:
        """Raises JSONTypeError when nn_optimizer is missing."""
        raw = self._make_valid_raw()
        del raw["nn_optimizer"]
        with pytest.raises(JSONTypeError, match="Missing required field 'nn_optimizer'"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_invalid_nn_optimizer_raises(self) -> None:
        """Raises JSONTypeError when nn_optimizer is not valid."""
        raw = self._make_valid_raw()
        raw["nn_optimizer"] = "rmsprop"
        with pytest.raises(JSONTypeError, match="nn_optimizer"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_nn_optimizer_adam_accepted(self) -> None:
        """adam nn_optimizer is accepted."""
        raw = self._make_valid_raw()
        raw["nn_optimizer"] = "adam"
        result = decode_unified_regression_optimize_parse_result(raw)
        assert result["nn_optimizer"] == "adam"

    def test_nn_optimizer_sgd_accepted(self) -> None:
        """sgd nn_optimizer is accepted."""
        raw = self._make_valid_raw()
        raw["nn_optimizer"] = "sgd"
        result = decode_unified_regression_optimize_parse_result(raw)
        assert result["nn_optimizer"] == "sgd"

    def test_missing_bidirectional_raises(self) -> None:
        """Raises JSONTypeError when bidirectional is missing."""
        raw = self._make_valid_raw()
        del raw["bidirectional"]
        with pytest.raises(JSONTypeError, match="Missing required field 'bidirectional'"):
            decode_unified_regression_optimize_parse_result(raw)

    def test_non_bool_bidirectional_raises(self) -> None:
        """Raises JSONTypeError when bidirectional is not a boolean."""
        raw = self._make_valid_raw()
        raw["bidirectional"] = "yes"
        with pytest.raises(JSONTypeError, match="must be a boolean"):
            decode_unified_regression_optimize_parse_result(raw)


class TestRegressionParseResultRequire:
    """Tests for require_unified_regression_optimize_parse_result."""

    def test_valid_dict(self) -> None:
        """Accepts a valid dict."""
        original = _make_regression_parse_result()
        encoded = encode_unified_regression_optimize_parse_result(original)
        raw_value: JSONValue = encoded
        decoded = require_unified_regression_optimize_parse_result(raw_value)
        assert decoded == original

    def test_non_dict_raises(self) -> None:
        """Raises JSONTypeError for non-dict input."""
        raw_value: JSONValue = "not a dict"
        with pytest.raises(JSONTypeError, match="Expected a JSON object"):
            require_unified_regression_optimize_parse_result(raw_value)

    def test_list_raises(self) -> None:
        """Raises JSONTypeError for list input."""
        raw_value: JSONValue = [1, 2, 3]
        with pytest.raises(JSONTypeError, match="Expected a JSON object"):
            require_unified_regression_optimize_parse_result(raw_value)

    def test_none_raises(self) -> None:
        """Raises JSONTypeError for None input."""
        raw_value: JSONValue = None
        with pytest.raises(JSONTypeError, match="Expected a JSON object"):
            require_unified_regression_optimize_parse_result(raw_value)


# =============================================================================
# Tests: UnifiedRegressionOptimizationResult encode/decode
# =============================================================================


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
