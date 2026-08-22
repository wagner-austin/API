"""Tests for worker/optimize_types.py encode/decode round-trip validation.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
All encode/decode/require functions are round-trip tested.
"""

from __future__ import annotations

import pytest
from covenant_ml.datasets.types import LoadPhase
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from platform_core.json_utils import JSONObject, JSONTypeError

from covenant_radar_api.worker.optimize_types import (
    LoadingProgressInfo,
    OptimizePhase,
    PhaseProgressInfo,
    TrialProgressInfo,
    UnifiedOptimizationResult,
    decode_loading_progress_info,
    decode_phase_progress_info,
    decode_trial_progress_info,
    decode_unified_optimization_result,
    encode_loading_progress_info,
    encode_phase_progress_info,
    encode_trial_progress_info,
    encode_unified_optimization_result,
)
from tests._optimize_types_fixtures import (
    _make_optimization_result,
)


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
