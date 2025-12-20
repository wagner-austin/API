"""Tests for scripts/optimize runner functions.

Tests backend-specific optimization runners (XGBoost, MLP, LightGBM, LSTM).
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import scripts._test_hooks as _hooks
from scripts._test_hooks import (
    LightGBMOptimizationResult,
    LSTMOptimizationResult,
    MLPOptimizationResult,
    XGBoostOptimizationResult,
)
from scripts.optimize.runner import (
    get_project_root,
    run_lightgbm,
    run_lstm,
    run_mlp,
    run_xgboost,
)

from .conftest import (
    make_fake_lightgbm_result,
    make_fake_lstm_result,
    make_fake_mlp_result,
    make_fake_result,
)


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_returns_parent_of_scripts(self) -> None:
        """Test project root is parent of scripts directory."""
        root: Path = get_project_root()
        assert root.name == "covenant-radar-api"
        assert (root / "scripts").exists()


class TestRunXGBoost:
    """Tests for run_xgboost function."""

    def test_runs_optimization_with_hook(self, tmp_path: Path) -> None:
        """Test run_xgboost uses the xgboost_runner hook."""
        fake_result = make_fake_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            _ = progress_callback  # Available for progress reporting
            _ = phase_callback  # Available for phase reporting
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            result: XGBoostOptimizationResult = run_xgboost("taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "taiwan" in config_json
            assert "10" in config_json
            assert "full" in config_json
        finally:
            _hooks.xgboost_runner = original

    def test_includes_timeout_when_provided(self, tmp_path: Path) -> None:
        """Test run_xgboost includes timeout in config when provided."""
        fake_result = make_fake_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            _ = progress_callback  # Available for progress reporting
            _ = phase_callback  # Available for phase reporting
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            run_xgboost("taiwan", 10, "full", "cpu", 60)
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "timeout_seconds" in config_json
            assert "60" in config_json
        finally:
            _hooks.xgboost_runner = original


class TestRunMLP:
    """Tests for run_mlp function."""

    def test_runs_optimization_with_hook(self, tmp_path: Path) -> None:
        """Test run_mlp uses the mlp_runner hook."""
        fake_result = make_fake_mlp_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.MLPTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.MLPPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.MLPLoadingProgressCallbackProtocol | None = None,
        ) -> MLPOptimizationResult:
            _ = progress_callback  # Available for progress reporting
            _ = phase_callback  # Available for phase reporting
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.mlp_runner
        _hooks.mlp_runner = fake_runner
        try:
            result: MLPOptimizationResult = run_mlp("taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "taiwan" in config_json
            assert "10" in config_json
            assert "full" in config_json
            # MLP-specific config
            assert "precision" in config_json
            assert "adamw" in config_json
        finally:
            _hooks.mlp_runner = original

    def test_includes_timeout_when_provided(self, tmp_path: Path) -> None:
        """Test run_mlp includes timeout in config when provided."""
        fake_result = make_fake_mlp_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.MLPTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.MLPPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.MLPLoadingProgressCallbackProtocol | None = None,
        ) -> MLPOptimizationResult:
            _ = progress_callback  # Available for progress reporting
            _ = phase_callback  # Available for phase reporting
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.mlp_runner
        _hooks.mlp_runner = fake_runner
        try:
            run_mlp("taiwan", 10, "full", "cpu", 60)
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "timeout_seconds" in config_json
            assert "60" in config_json
        finally:
            _hooks.mlp_runner = original


class TestRunLightGBM:
    """Tests for run_lightgbm function."""

    def test_runs_optimization_with_hook(self, tmp_path: Path) -> None:
        """Test run_lightgbm uses the lightgbm_runner hook."""
        fake_result = make_fake_lightgbm_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LightGBMTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.LightGBMPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.LightGBMLoadingProgressCallbackProtocol | None = None,
        ) -> LightGBMOptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.lightgbm_runner
        _hooks.lightgbm_runner = fake_runner
        try:
            result: LightGBMOptimizationResult = run_lightgbm("taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "taiwan" in config_json
            assert "10" in config_json
            assert "full" in config_json
            # LightGBM-specific config
            assert "early_stopping_rounds" in config_json
        finally:
            _hooks.lightgbm_runner = original

    def test_includes_timeout_when_provided(self, tmp_path: Path) -> None:
        """Test run_lightgbm includes timeout in config when provided."""
        fake_result = make_fake_lightgbm_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LightGBMTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.LightGBMPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.LightGBMLoadingProgressCallbackProtocol | None = None,
        ) -> LightGBMOptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.lightgbm_runner
        _hooks.lightgbm_runner = fake_runner
        try:
            run_lightgbm("taiwan", 10, "full", "cpu", 60)
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "timeout_seconds" in config_json
            assert "60" in config_json
        finally:
            _hooks.lightgbm_runner = original


class TestRunLSTM:
    """Tests for run_lstm function."""

    def test_runs_optimization_with_hook(self, tmp_path: Path) -> None:
        """Test run_lstm uses the lstm_runner hook."""
        fake_result = make_fake_lstm_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LSTMTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.LSTMPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.LSTMLoadingProgressCallbackProtocol | None = None,
        ) -> LSTMOptimizationResult:
            _ = progress_callback  # Available for progress reporting
            _ = phase_callback  # Available for phase reporting
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.lstm_runner
        _hooks.lstm_runner = fake_runner
        try:
            result: LSTMOptimizationResult = run_lstm("taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "taiwan" in config_json
            assert "10" in config_json
            assert "full" in config_json
            # LSTM-specific config
            assert "precision" in config_json
            assert "sequence_length" in config_json
            assert "bidirectional" in config_json
        finally:
            _hooks.lstm_runner = original

    def test_includes_timeout_when_provided(self, tmp_path: Path) -> None:
        """Test run_lstm includes timeout in config when provided."""
        fake_result = make_fake_lstm_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LSTMTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.LSTMPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.LSTMLoadingProgressCallbackProtocol | None = None,
        ) -> LSTMOptimizationResult:
            _ = progress_callback  # Available for progress reporting
            _ = phase_callback  # Available for phase reporting
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.lstm_runner
        _hooks.lstm_runner = fake_runner
        try:
            run_lstm("taiwan", 10, "full", "cpu", 60)
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "timeout_seconds" in config_json
            assert "60" in config_json
        finally:
            _hooks.lstm_runner = original
