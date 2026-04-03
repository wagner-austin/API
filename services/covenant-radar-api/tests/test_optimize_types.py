"""Tests for worker/optimize_types.py encode/decode round-trip validation.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
All encode/decode/require functions are round-trip tested.
"""

from __future__ import annotations

import pytest
from covenant_ml.datasets.types import LoadPhase
from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.types import BackendName
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from covenant_radar_api.worker.optimize_types import (
    LoadingProgressInfo,
    OptimizePhase,
    PhaseProgressInfo,
    TrialProgressInfo,
    UnifiedOptimizationResult,
    UnifiedOptimizeParseResult,
    decode_loading_progress_info,
    decode_phase_progress_info,
    decode_trial_progress_info,
    decode_unified_optimization_result,
    decode_unified_optimize_parse_result,
    encode_loading_progress_info,
    encode_phase_progress_info,
    encode_trial_progress_info,
    encode_unified_optimization_result,
    encode_unified_optimize_parse_result,
    require_unified_optimization_result,
    require_unified_optimize_parse_result,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_parse_result(
    backend: BackendName = "xgboost",
    dataset: str = "taiwan",
) -> UnifiedOptimizeParseResult:
    """Create a valid UnifiedOptimizeParseResult for testing.

    Args:
        backend: Backend name.
        dataset: Dataset name.

    Returns:
        UnifiedOptimizeParseResult with all fields populated.
    """
    return UnifiedOptimizeParseResult(
        backend=backend,
        dataset=dataset,
        n_trials=50,
        timeout_seconds=3600,
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


def _make_optimization_result(
    backend: BackendName = "xgboost",
) -> UnifiedOptimizationResult:
    """Create a valid UnifiedOptimizationResult for testing.

    Args:
        backend: Backend name.

    Returns:
        UnifiedOptimizationResult with all fields populated.
    """
    return UnifiedOptimizationResult(
        backend=backend,
        status="complete",
        dataset="taiwan",
        n_samples=6819,
        n_features=95,
        feature_preset="none",
        n_trials_complete=50,
        n_trials_pruned=3,
        n_trials_failed=1,
        best_trial_number=42,
        best_value=0.8765,
        best_int_params=SampledIntParams(max_depth=6, n_estimators=200),
        best_float_params=SampledFloatParams(learning_rate=0.05, reg_alpha=0.1),
        best_string_params=SampledStringParams(booster="gbtree"),
        duration_seconds=120.5,
    )


# =============================================================================
# Tests for UnifiedOptimizeParseResult encode/decode
# =============================================================================


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


# =============================================================================
# Tests for UnifiedOptimizationResult encode/decode
# =============================================================================


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


# =============================================================================
# Tests for PhaseProgressInfo encode/decode
# =============================================================================


class TestPhaseProgressInfoEncodeDecode:
    """Tests for PhaseProgressInfo encode/decode round-trip."""

    def test_encode_decode_round_trip(self) -> None:
        """Encoding then decoding produces identical result."""
        original = PhaseProgressInfo(
            phase="loading_data",
            backend="xgboost",
            dataset="taiwan",
            n_samples=0,
            n_features=0,
        )
        encoded = encode_phase_progress_info(original)
        decoded = decode_phase_progress_info(encoded)
        assert decoded == original

    def test_all_phases_round_trip(self) -> None:
        """All phase values round-trip correctly."""
        phases: tuple[OptimizePhase, ...] = (
            "loading_data",
            "feature_engineering",
            "optimizing",
            "saving",
        )
        for phase in phases:
            original = PhaseProgressInfo(
                phase=phase,
                backend="mlp",
                dataset="us",
                n_samples=100,
                n_features=50,
            )
            encoded = encode_phase_progress_info(original)
            decoded = decode_phase_progress_info(encoded)
            assert decoded["phase"] == phase

    def test_decode_invalid_phase_raises(self) -> None:
        """Invalid phase raises JSONTypeError."""
        raw: JSONObject = {
            "phase": "unknown_phase",
            "backend": "xgboost",
            "dataset": "taiwan",
            "n_samples": 0,
            "n_features": 0,
        }
        with pytest.raises(JSONTypeError, match="phase"):
            decode_phase_progress_info(raw)

    def test_decode_missing_backend_raises(self) -> None:
        """Missing backend raises JSONTypeError."""
        raw: JSONObject = {
            "phase": "loading_data",
            "dataset": "taiwan",
            "n_samples": 0,
            "n_features": 0,
        }
        with pytest.raises(JSONTypeError, match="backend"):
            decode_phase_progress_info(raw)


# =============================================================================
# Tests for LoadingProgressInfo encode/decode
# =============================================================================


class TestLoadingProgressInfoEncodeDecode:
    """Tests for LoadingProgressInfo encode/decode round-trip."""

    def test_encode_decode_round_trip(self) -> None:
        """Encoding then decoding produces identical result."""
        original = LoadingProgressInfo(
            dataset="taiwan",
            phase="reading",
            percent_complete=50.0,
            rows_processed=500,
            rows_total=1000,
            message="Reading CSV",
        )
        encoded = encode_loading_progress_info(original)
        decoded = decode_loading_progress_info(encoded)
        assert decoded == original

    def test_all_load_phases_round_trip(self) -> None:
        """All load phase values round-trip correctly."""
        phases: tuple[LoadPhase, ...] = ("reading", "parsing", "encoding")
        for phase in phases:
            original = LoadingProgressInfo(
                dataset="us",
                phase=phase,
                percent_complete=75.0,
                rows_processed=750,
                rows_total=1000,
                message=f"Phase: {phase}",
            )
            encoded = encode_loading_progress_info(original)
            decoded = decode_loading_progress_info(encoded)
            assert decoded["phase"] == phase

    def test_decode_invalid_load_phase_raises(self) -> None:
        """Invalid load phase raises JSONTypeError."""
        raw: JSONObject = {
            "dataset": "taiwan",
            "phase": "transforming",
            "percent_complete": 50.0,
            "rows_processed": 500,
            "rows_total": 1000,
            "message": "test",
        }
        with pytest.raises(JSONTypeError, match="phase"):
            decode_loading_progress_info(raw)

    def test_decode_missing_message_raises(self) -> None:
        """Missing message field raises JSONTypeError."""
        raw: JSONObject = {
            "dataset": "taiwan",
            "phase": "reading",
            "percent_complete": 50.0,
            "rows_processed": 500,
            "rows_total": 1000,
        }
        with pytest.raises(JSONTypeError, match="message"):
            decode_loading_progress_info(raw)


# =============================================================================
# Tests for TrialProgressInfo encode/decode
# =============================================================================


class TestTrialProgressInfoEncodeDecode:
    """Tests for TrialProgressInfo encode/decode round-trip."""

    def test_encode_decode_round_trip(self) -> None:
        """Encoding then decoding produces identical result."""
        original = TrialProgressInfo(
            backend="lightgbm",
            trial_number=5,
            n_trials_total=50,
            current_value=0.82,
            best_value=0.85,
            best_trial=3,
            is_best=False,
        )
        encoded = encode_trial_progress_info(original)
        decoded = decode_trial_progress_info(encoded)
        assert decoded == original

    def test_is_best_true_round_trip(self) -> None:
        """is_best=True round-trips correctly."""
        original = TrialProgressInfo(
            backend="xgboost",
            trial_number=10,
            n_trials_total=100,
            current_value=0.90,
            best_value=0.90,
            best_trial=10,
            is_best=True,
        )
        encoded = encode_trial_progress_info(original)
        decoded = decode_trial_progress_info(encoded)
        assert decoded["is_best"] is True

    def test_decode_missing_trial_number_raises(self) -> None:
        """Missing trial_number raises JSONTypeError."""
        raw: JSONObject = {
            "backend": "xgboost",
            "n_trials_total": 50,
            "current_value": 0.82,
            "best_value": 0.85,
            "best_trial": 3,
            "is_best": False,
        }
        with pytest.raises(JSONTypeError, match="trial_number"):
            decode_trial_progress_info(raw)

    def test_decode_missing_is_best_raises(self) -> None:
        """Missing is_best raises JSONTypeError."""
        raw: JSONObject = {
            "backend": "xgboost",
            "trial_number": 5,
            "n_trials_total": 50,
            "current_value": 0.82,
            "best_value": 0.85,
            "best_trial": 3,
        }
        with pytest.raises(JSONTypeError, match="is_best"):
            decode_trial_progress_info(raw)


# =============================================================================
# Tests for all sampled param encode/decode branches
# =============================================================================


class TestSampledParamsBranchCoverage:
    """Tests ensuring ALL encode/decode branches for sampled params are covered.

    Each backend uses different param names. These tests exercise every
    individual param branch in _encode_*/_decode_* functions.
    """

    def test_lightgbm_int_params_round_trip(self) -> None:
        """LightGBM int params (num_leaves, min_child_samples) round-trip."""
        original = UnifiedOptimizationResult(
            backend="lightgbm",
            status="complete",
            dataset="taiwan",
            n_samples=1000,
            n_features=50,
            feature_preset="none",
            n_trials_complete=10,
            n_trials_pruned=0,
            n_trials_failed=0,
            best_trial_number=3,
            best_value=0.87,
            best_int_params=SampledIntParams(
                max_depth=-1, n_estimators=100, num_leaves=31, min_child_samples=20
            ),
            best_float_params=SampledFloatParams(
                learning_rate=0.1,
                reg_alpha=0.01,
                reg_lambda=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                feature_fraction=0.7,
            ),
            best_string_params=SampledStringParams(boosting_type="gbdt"),
            duration_seconds=10.0,
        )
        encoded = encode_unified_optimization_result(original)
        decoded = decode_unified_optimization_result(encoded)
        assert decoded == original
        assert decoded["best_int_params"]["num_leaves"] == 31
        assert decoded["best_int_params"]["min_child_samples"] == 20
        assert decoded["best_float_params"]["feature_fraction"] == 0.7
        assert decoded["best_string_params"]["boosting_type"] == "gbdt"

    def test_mlp_int_params_round_trip(self) -> None:
        """MLP int params (n_layers, hidden_size, batch_size) round-trip."""
        original = UnifiedOptimizationResult(
            backend="mlp",
            status="complete",
            dataset="us",
            n_samples=500,
            n_features=30,
            feature_preset="full",
            n_trials_complete=20,
            n_trials_pruned=2,
            n_trials_failed=1,
            best_trial_number=15,
            best_value=0.82,
            best_int_params=SampledIntParams(n_layers=3, hidden_size=128, batch_size=64),
            best_float_params=SampledFloatParams(learning_rate=0.001, dropout=0.2),
            best_string_params=SampledStringParams(),
            duration_seconds=30.0,
        )
        encoded = encode_unified_optimization_result(original)
        decoded = decode_unified_optimization_result(encoded)
        assert decoded == original
        assert decoded["best_int_params"]["n_layers"] == 3
        assert decoded["best_int_params"]["hidden_size"] == 128
        assert decoded["best_int_params"]["batch_size"] == 64
        assert decoded["best_float_params"]["dropout"] == 0.2

    def test_lstm_int_params_round_trip(self) -> None:
        """LSTM int params (num_layers, hidden_size, batch_size) round-trip."""
        original = UnifiedOptimizationResult(
            backend="lstm",
            status="complete",
            dataset="polish",
            n_samples=7000,
            n_features=64,
            feature_preset="log_only",
            n_trials_complete=15,
            n_trials_pruned=1,
            n_trials_failed=0,
            best_trial_number=10,
            best_value=0.80,
            best_int_params=SampledIntParams(num_layers=2, hidden_size=64, batch_size=32),
            best_float_params=SampledFloatParams(learning_rate=0.0005, dropout=0.3),
            best_string_params=SampledStringParams(),
            duration_seconds=45.0,
        )
        encoded = encode_unified_optimization_result(original)
        decoded = decode_unified_optimization_result(encoded)
        assert decoded == original
        assert decoded["best_int_params"]["num_layers"] == 2

    def test_cleargbm_int_params_round_trip(self) -> None:
        """ClearGBM int params (min_samples_split, min_samples_leaf, max_bins) round-trip."""
        original = UnifiedOptimizationResult(
            backend="cleargbm",
            status="complete",
            dataset="taiwan",
            n_samples=6819,
            n_features=95,
            feature_preset="none",
            n_trials_complete=25,
            n_trials_pruned=3,
            n_trials_failed=0,
            best_trial_number=20,
            best_value=0.84,
            best_int_params=SampledIntParams(
                max_depth=5,
                n_estimators=100,
                min_samples_split=10,
                min_samples_leaf=5,
                max_bins=64,
            ),
            best_float_params=SampledFloatParams(
                learning_rate=0.1, subsample=1.0, reg_alpha=0.0, reg_lambda=1.0
            ),
            best_string_params=SampledStringParams(),
            duration_seconds=20.0,
        )
        encoded = encode_unified_optimization_result(original)
        decoded = decode_unified_optimization_result(encoded)
        assert decoded == original
        assert decoded["best_int_params"]["min_samples_split"] == 10
        assert decoded["best_int_params"]["min_samples_leaf"] == 5
        assert decoded["best_int_params"]["max_bins"] == 64

    def test_logreg_params_round_trip(self) -> None:
        """LogReg params (max_iter, C, tol, l1_ratio, penalty, solver) round-trip."""
        original = UnifiedOptimizationResult(
            backend="logreg",
            status="complete",
            dataset="us",
            n_samples=78682,
            n_features=18,
            feature_preset="none",
            n_trials_complete=30,
            n_trials_pruned=0,
            n_trials_failed=0,
            best_trial_number=25,
            best_value=0.75,
            best_int_params=SampledIntParams(max_iter=1000),
            best_float_params=SampledFloatParams(C=1.0, tol=0.0001, l1_ratio=0.5),
            best_string_params=SampledStringParams(penalty="elasticnet", solver="saga"),
            duration_seconds=5.0,
        )
        encoded = encode_unified_optimization_result(original)
        decoded = decode_unified_optimization_result(encoded)
        assert decoded == original
        assert decoded["best_int_params"]["max_iter"] == 1000
        assert decoded["best_float_params"]["C"] == 1.0
        assert decoded["best_float_params"]["tol"] == 0.0001
        assert decoded["best_float_params"]["l1_ratio"] == 0.5
        assert decoded["best_string_params"]["penalty"] == "elasticnet"
        assert decoded["best_string_params"]["solver"] == "saga"

    def test_random_forest_params_round_trip(self) -> None:
        """RandomForest params (min_samples_split/leaf, max_features) round-trip."""
        original = UnifiedOptimizationResult(
            backend="random_forest",
            status="complete",
            dataset="polish",
            n_samples=7027,
            n_features=64,
            feature_preset="ratios_only",
            n_trials_complete=40,
            n_trials_pruned=0,
            n_trials_failed=0,
            best_trial_number=35,
            best_value=0.78,
            best_int_params=SampledIntParams(
                n_estimators=200,
                min_samples_split=5,
                min_samples_leaf=2,
            ),
            best_float_params=SampledFloatParams(max_features_float=0.7),
            best_string_params=SampledStringParams(max_features="sqrt"),
            duration_seconds=15.0,
        )
        encoded = encode_unified_optimization_result(original)
        decoded = decode_unified_optimization_result(encoded)
        assert decoded == original
        assert decoded["best_float_params"]["max_features_float"] == 0.7
        assert decoded["best_string_params"]["max_features"] == "sqrt"

    def test_xgboost_dart_params_round_trip(self) -> None:
        """XGBoost DART params (drop_rate, skip_drop, rate_drop) round-trip."""
        original = UnifiedOptimizationResult(
            backend="xgboost",
            status="complete",
            dataset="taiwan",
            n_samples=6819,
            n_features=95,
            feature_preset="none",
            n_trials_complete=50,
            n_trials_pruned=5,
            n_trials_failed=0,
            best_trial_number=42,
            best_value=0.88,
            best_int_params=SampledIntParams(max_depth=6, n_estimators=200),
            best_float_params=SampledFloatParams(
                learning_rate=0.05,
                reg_alpha=0.1,
                reg_lambda=0.5,
                subsample=0.9,
                colsample_bytree=0.7,
                drop_rate=0.1,
                skip_drop=0.5,
                rate_drop=0.05,
            ),
            best_string_params=SampledStringParams(booster="dart"),
            duration_seconds=60.0,
        )
        encoded = encode_unified_optimization_result(original)
        decoded = decode_unified_optimization_result(encoded)
        assert decoded == original
        assert decoded["best_float_params"]["drop_rate"] == 0.1
        assert decoded["best_float_params"]["skip_drop"] == 0.5
        assert decoded["best_float_params"]["rate_drop"] == 0.05

    def test_decode_non_string_backend_in_result_raises(self) -> None:
        """Non-string backend in encoded result raises JSONTypeError."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        encoded["backend"] = 123
        with pytest.raises(JSONTypeError, match="backend"):
            decode_unified_optimization_result(encoded)

    def test_decode_non_number_best_value_raises(self) -> None:
        """Non-number best_value raises JSONTypeError."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        encoded["best_value"] = "high"
        with pytest.raises(JSONTypeError, match="best_value"):
            decode_unified_optimization_result(encoded)

    def test_decode_non_number_duration_raises(self) -> None:
        """Non-number duration_seconds raises JSONTypeError."""
        original = _make_optimization_result()
        encoded = encode_unified_optimization_result(original)
        encoded["duration_seconds"] = "fast"
        with pytest.raises(JSONTypeError, match="duration_seconds"):
            decode_unified_optimization_result(encoded)


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
