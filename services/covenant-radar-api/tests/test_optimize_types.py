"""Tests for worker/optimize_types.py encode/decode round-trip validation.

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
from covenant_ml.types import BackendName
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from covenant_radar_api.worker.optimize_types import (
    UnifiedOptimizationResult,
    UnifiedOptimizeParseResult,
    decode_unified_optimization_result,
    decode_unified_optimize_parse_result,
    encode_unified_optimization_result,
    encode_unified_optimize_parse_result,
    require_unified_optimization_result,
    require_unified_optimize_parse_result,
)
from tests._optimize_types_fixtures import (
    _make_optimization_result,
    _make_parse_result,
)


class TestUnifiedOptimizeParseResultEncodeDecode:
    """Tests for UnifiedOptimizeParseResult encode/decode round-trip."""

    def test_encode_decode_round_trip(self) -> None:
        """Encoding then decoding produces identical result."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        decoded = decode_unified_optimize_parse_result(encoded)
        assert decoded == original

    def test_encode_decode_with_null_timeout(self) -> None:
        """Round-trip preserves None timeout_seconds."""
        original = _make_parse_result()
        original_with_none = UnifiedOptimizeParseResult(
            **{**original, "timeout_seconds": None},
        )
        encoded = encode_unified_optimize_parse_result(original_with_none)
        decoded = decode_unified_optimize_parse_result(encoded)
        assert decoded["timeout_seconds"] is None

    def test_encode_all_backends(self) -> None:
        """All 7 backends encode and decode correctly."""
        backends: tuple[BackendName, ...] = (
            "xgboost",
            "mlp",
            "lstm",
            "lightgbm",
            "cleargbm",
            "logreg",
            "random_forest",
        )
        for backend in backends:
            original = _make_parse_result(backend=backend)
            encoded = encode_unified_optimize_parse_result(original)
            decoded = decode_unified_optimize_parse_result(encoded)
            assert decoded["backend"] == backend

    def test_encode_all_devices(self) -> None:
        """All device values encode and decode correctly."""
        for device in ("cpu", "cuda", "auto"):
            original = _make_parse_result()
            updated = UnifiedOptimizeParseResult(**{**original, "device": device})
            encoded = encode_unified_optimize_parse_result(updated)
            decoded = decode_unified_optimize_parse_result(encoded)
            assert decoded["device"] == device

    def test_encode_all_feature_presets(self) -> None:
        """All feature preset values encode and decode correctly."""
        presets: tuple[FeaturePreset, ...] = ("none", "log_only", "ratios_only", "full", "temporal")
        for preset in presets:
            original = _make_parse_result()
            updated = UnifiedOptimizeParseResult(**{**original, "feature_preset": preset})
            encoded = encode_unified_optimize_parse_result(updated)
            decoded = decode_unified_optimize_parse_result(encoded)
            assert decoded["feature_preset"] == preset

    def test_encode_all_precisions(self) -> None:
        """All precision values encode and decode correctly."""
        for precision in ("fp32", "fp16", "bf16", "auto"):
            original = _make_parse_result()
            updated = UnifiedOptimizeParseResult(**{**original, "precision": precision})
            encoded = encode_unified_optimize_parse_result(updated)
            decoded = decode_unified_optimize_parse_result(encoded)
            assert decoded["precision"] == precision

    def test_encode_all_nn_optimizers(self) -> None:
        """All nn_optimizer values encode and decode correctly."""
        for opt in ("adamw", "adam", "sgd"):
            original = _make_parse_result()
            updated = UnifiedOptimizeParseResult(**{**original, "nn_optimizer": opt})
            encoded = encode_unified_optimize_parse_result(updated)
            decoded = decode_unified_optimize_parse_result(encoded)
            assert decoded["nn_optimizer"] == opt

    def test_encode_bidirectional_true(self) -> None:
        """bidirectional=True round-trips correctly."""
        original = _make_parse_result()
        updated = UnifiedOptimizeParseResult(**{**original, "bidirectional": True})
        encoded = encode_unified_optimize_parse_result(updated)
        decoded = decode_unified_optimize_parse_result(encoded)
        assert decoded["bidirectional"] is True

    def test_decode_missing_backend_raises(self) -> None:
        """Missing 'backend' field raises JSONTypeError."""
        raw: JSONObject = {"dataset": "taiwan", "n_trials": 10}
        with pytest.raises(JSONTypeError, match="backend"):
            decode_unified_optimize_parse_result(raw)

    def test_decode_invalid_backend_raises(self) -> None:
        """Invalid backend value raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        encoded["backend"] = "invalid"
        with pytest.raises(JSONTypeError, match="backend"):
            decode_unified_optimize_parse_result(encoded)

    def test_decode_missing_n_trials_raises(self) -> None:
        """Missing 'n_trials' field raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        del encoded["n_trials"]
        with pytest.raises(JSONTypeError, match="n_trials"):
            decode_unified_optimize_parse_result(encoded)

    def test_decode_invalid_device_raises(self) -> None:
        """Invalid device value raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        encoded["device"] = "tpu"
        with pytest.raises(JSONTypeError, match="device"):
            decode_unified_optimize_parse_result(encoded)

    def test_decode_invalid_feature_preset_raises(self) -> None:
        """Invalid feature_preset value raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        encoded["feature_preset"] = "garbage"
        with pytest.raises(JSONTypeError, match="feature_preset"):
            decode_unified_optimize_parse_result(encoded)

    def test_decode_invalid_precision_raises(self) -> None:
        """Invalid precision value raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        encoded["precision"] = "fp64"
        with pytest.raises(JSONTypeError, match="precision"):
            decode_unified_optimize_parse_result(encoded)

    def test_decode_invalid_nn_optimizer_raises(self) -> None:
        """Invalid nn_optimizer value raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        encoded["nn_optimizer"] = "rmsprop"
        with pytest.raises(JSONTypeError, match="nn_optimizer"):
            decode_unified_optimize_parse_result(encoded)

    def test_decode_non_int_n_trials_raises(self) -> None:
        """Non-integer n_trials raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        encoded["n_trials"] = "fifty"
        with pytest.raises(JSONTypeError, match="n_trials"):
            decode_unified_optimize_parse_result(encoded)

    def test_decode_non_bool_bidirectional_raises(self) -> None:
        """Non-boolean bidirectional raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        encoded["bidirectional"] = "yes"
        with pytest.raises(JSONTypeError, match="bidirectional"):
            decode_unified_optimize_parse_result(encoded)

    def test_decode_invalid_timeout_type_raises(self) -> None:
        """Non-integer timeout_seconds raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        encoded["timeout_seconds"] = "an_hour"
        with pytest.raises(JSONTypeError, match="timeout_seconds"):
            decode_unified_optimize_parse_result(encoded)


class TestRequireUnifiedOptimizeParseResult:
    """Tests for require_unified_optimize_parse_result validation."""

    def test_require_valid_json_object(self) -> None:
        """Valid JSON object passes validation."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        raw: JSONValue = encoded
        result = require_unified_optimize_parse_result(raw)
        assert result == original

    def test_require_non_dict_raises(self) -> None:
        """Non-dict value raises JSONTypeError."""
        raw: JSONValue = "not a dict"
        with pytest.raises(JSONTypeError, match="JSON object"):
            require_unified_optimize_parse_result(raw)

    def test_require_list_raises(self) -> None:
        """List value raises JSONTypeError."""
        raw: JSONValue = [1, 2, 3]
        with pytest.raises(JSONTypeError, match="JSON object"):
            require_unified_optimize_parse_result(raw)


class TestUnifiedOptimizationResultEncodeDecode:
    """Tests for UnifiedOptimizationResult encode/decode round-trip."""

    def test_encode_decode_round_trip(self) -> None:
        """Encoding then decoding produces identical result."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        decoded = decode_unified_optimization_result(encoded)
        assert decoded == original

    def test_encode_decode_empty_params(self) -> None:
        """Round-trip with empty sampled params works."""
        original = UnifiedOptimizationResult(
            backend="mlp",
            status="complete",
            dataset="us",
            n_samples=1000,
            n_features=50,
            feature_preset="log_only",
            n_trials_complete=20,
            n_trials_pruned=0,
            n_trials_failed=0,
            best_trial_number=10,
            best_value=0.92,
            best_int_params=SampledIntParams(),
            best_float_params=SampledFloatParams(),
            best_string_params=SampledStringParams(),
            duration_seconds=30.0,
        )
        encoded = encode_unified_optimization_result(original)
        decoded = decode_unified_optimization_result(encoded)
        assert decoded == original

    def test_encode_decode_all_backends(self) -> None:
        """All 7 backends round-trip correctly."""
        backends: tuple[BackendName, ...] = (
            "xgboost",
            "mlp",
            "lstm",
            "lightgbm",
            "cleargbm",
            "logreg",
            "random_forest",
        )
        for backend in backends:
            original = _make_optimization_result(backend=backend)
            encoded = encode_unified_optimization_result(original)
            decoded = decode_unified_optimization_result(encoded)
            assert decoded["backend"] == backend

    def test_decode_wrong_status_raises(self) -> None:
        """Status != 'complete' raises JSONTypeError."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        encoded["status"] = "running"
        with pytest.raises(JSONTypeError, match="status"):
            decode_unified_optimization_result(encoded)

    def test_decode_missing_best_value_raises(self) -> None:
        """Missing best_value raises JSONTypeError."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        del encoded["best_value"]
        with pytest.raises(JSONTypeError, match="best_value"):
            decode_unified_optimization_result(encoded)

    def test_decode_missing_best_int_params_raises(self) -> None:
        """Missing best_int_params raises JSONTypeError."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        del encoded["best_int_params"]
        with pytest.raises(JSONTypeError, match="best_int_params"):
            decode_unified_optimization_result(encoded)

    def test_decode_non_object_best_float_params_raises(self) -> None:
        """Non-object best_float_params raises JSONTypeError."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        encoded["best_float_params"] = "not_an_object"
        with pytest.raises(JSONTypeError, match="best_float_params"):
            decode_unified_optimization_result(encoded)

    def test_decode_non_int_in_int_params_raises(self) -> None:
        """Non-integer value in int_params raises JSONTypeError."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        int_params: JSONObject = {"max_depth": "six"}
        encoded["best_int_params"] = int_params
        with pytest.raises(JSONTypeError, match="max_depth"):
            decode_unified_optimization_result(encoded)

    def test_decode_non_number_in_float_params_raises(self) -> None:
        """Non-number value in float_params raises JSONTypeError."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        float_params: JSONObject = {"learning_rate": "fast"}
        encoded["best_float_params"] = float_params
        with pytest.raises(JSONTypeError, match="learning_rate"):
            decode_unified_optimization_result(encoded)

    def test_decode_non_string_in_string_params_raises(self) -> None:
        """Non-string value in string_params raises JSONTypeError."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        string_params: JSONObject = {"booster": 123}
        encoded["best_string_params"] = string_params
        with pytest.raises(JSONTypeError, match="booster"):
            decode_unified_optimization_result(encoded)

    def test_decode_int_as_float_param_converts(self) -> None:
        """Integer value in float_params is converted to float."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        float_params: JSONObject = {"learning_rate": 1}
        encoded["best_float_params"] = float_params
        decoded = decode_unified_optimization_result(encoded)
        assert decoded["best_float_params"]["learning_rate"] == 1.0
        # Verify the int was converted to float by checking repr
        assert type(decoded["best_float_params"]["learning_rate"]).__name__ == "float"


class TestRequireUnifiedOptimizationResult:
    """Tests for require_unified_optimization_result validation."""

    def test_require_valid_json_object(self) -> None:
        """Valid JSON object passes validation."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        raw: JSONValue = encoded
        result = require_unified_optimization_result(raw)
        assert result == original

    def test_require_non_dict_raises(self) -> None:
        """Non-dict value raises JSONTypeError."""
        raw: JSONValue = 42
        with pytest.raises(JSONTypeError, match="JSON object"):
            require_unified_optimization_result(raw)


class TestDecodeParseResultMissingFields:
    """Tests for missing required fields in decode_unified_optimize_parse_result."""

    def test_missing_device_raises(self) -> None:
        """Missing device field raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        del encoded["device"]
        with pytest.raises(JSONTypeError, match="device"):
            decode_unified_optimize_parse_result(encoded)

    def test_missing_feature_preset_raises(self) -> None:
        """Missing feature_preset field raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        del encoded["feature_preset"]
        with pytest.raises(JSONTypeError, match="feature_preset"):
            decode_unified_optimize_parse_result(encoded)

    def test_missing_precision_raises(self) -> None:
        """Missing precision field raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        del encoded["precision"]
        with pytest.raises(JSONTypeError, match="precision"):
            decode_unified_optimize_parse_result(encoded)

    def test_missing_nn_optimizer_raises(self) -> None:
        """Missing nn_optimizer field raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        del encoded["nn_optimizer"]
        with pytest.raises(JSONTypeError, match="nn_optimizer"):
            decode_unified_optimize_parse_result(encoded)

    def test_non_string_dataset_raises(self) -> None:
        """Non-string dataset field raises JSONTypeError."""
        original = _make_parse_result()
        encoded = encode_unified_optimize_parse_result(original)
        encoded["dataset"] = 123
        with pytest.raises(JSONTypeError, match="dataset"):
            decode_unified_optimize_parse_result(encoded)
