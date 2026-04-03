"""Integration tests for scripts/optimize entry point.

Tests main function, module entry point, and keyboard interrupt handling.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
import scripts._test_hooks as _hooks
from covenant_ml.types import BackendName
from scripts._test_hooks import (
    LoadingProgressCallbackProtocol,
    PhaseProgressCallbackProtocol,
    TrialProgressCallbackProtocol,
    TrialProgressInfo,
    UnifiedOptimizationResult,
)
from scripts.optimize import main

from .conftest import make_fake_lightgbm_result, make_fake_result


def _make_trial_info(
    trial_number: int = 1,
    n_trials_total: int = 5,
    current_value: float = 0.80,
    best_value: float = 0.80,
    best_trial: int = 1,
    is_best: bool = True,
    backend: BackendName = "xgboost",
) -> TrialProgressInfo:
    """Create a unified TrialProgressInfo for testing.

    Args:
        trial_number: Current trial number.
        n_trials_total: Total trials.
        current_value: Current trial AUC.
        best_value: Best AUC so far.
        best_trial: Best trial number.
        is_best: Whether current is best.
        backend: Backend name.

    Returns:
        TrialProgressInfo dict.
    """
    return TrialProgressInfo(
        backend=backend,
        trial_number=trial_number,
        n_trials_total=n_trials_total,
        current_value=current_value,
        best_value=best_value,
        best_trial=best_trial,
        is_best=is_best,
    )


class TestMain:
    """Tests for main function."""

    def test_main_with_defaults_runs_single(self) -> None:
        """Test main with no args runs single optimization and calls progress callback."""
        call_count = 0
        callback_calls = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            nonlocal call_count, callback_calls
            call_count += 1
            _ = phase_callback
            if progress_callback is not None:
                progress_callback(_make_trial_info(trial_number=1, is_best=True))
                callback_calls += 1
                progress_callback(
                    _make_trial_info(
                        trial_number=2, current_value=0.75, is_best=False, best_trial=1
                    )
                )
                callback_calls += 1
            return make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code: int = main(["-n", "5", "--no-save-model"])
            assert exit_code == 0
            assert call_count == 1
            assert callback_calls == 2
        finally:
            _hooks.optimization_runner = original

    def test_main_with_compare_presets(self) -> None:
        """Test main with -c runs compare presets."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            nonlocal call_count
            call_count += 1
            return make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code: int = main(["-c", "-n", "5", "--no-save-model"])
            assert exit_code == 0
            assert call_count == 4  # All four presets
        finally:
            _hooks.optimization_runner = original

    def test_main_with_all_datasets(self) -> None:
        """Test main with -a runs all datasets."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            nonlocal call_count
            call_count += 1
            return make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code: int = main(["-a", "-n", "5", "--no-save-model"])
            assert exit_code == 0
            assert call_count == 3  # All three datasets
        finally:
            _hooks.optimization_runner = original

    def test_main_with_verbose(self) -> None:
        """Test main with -v sets debug logging."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            nonlocal call_count
            call_count += 1
            return make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code: int = main(["-v", "-n", "5", "--no-save-model"])
            assert exit_code == 0
            assert call_count == 1
        finally:
            _hooks.optimization_runner = original

    def test_main_with_all_options(self) -> None:
        """Test main with multiple options."""
        configs_received: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            configs_received.append(config_json)
            return make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code: int = main(
                ["-d", "us", "-n", "25", "-f", "log_only", "--device", "cpu", "--no-save-model"]
            )
            assert exit_code == 0
            assert len(configs_received) == 1
            config = configs_received[0]
            assert '"us"' in config
            assert "25" in config
            assert "log_only" in config
            assert "cpu" in config
        finally:
            _hooks.optimization_runner = original

    def test_main_with_timeout(self) -> None:
        """Test main with timeout option."""
        configs_received: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            configs_received.append(config_json)
            return make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code: int = main(["-n", "5", "-t", "120", "--no-save-model"])
            assert exit_code == 0
            assert len(configs_received) == 1
            assert "timeout_seconds" in configs_received[0]
            assert "120" in configs_received[0]
        finally:
            _hooks.optimization_runner = original

    def test_main_help_exits_zero(self) -> None:
        """Test main with --help exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


class TestModuleEntry:
    """Tests for module entry point."""

    def test_module_main_entry_with_help(self) -> None:
        """Test __main__ entry point via runpy."""
        import runpy

        modules_to_clear = [k for k in sys.modules if k.startswith("scripts.optimize")]
        saved_modules: dict[str, ModuleType] = {}
        for mod in modules_to_clear:
            saved_modules[mod] = sys.modules.pop(mod)

        original_argv = sys.argv
        sys.argv = ["optimize", "--help"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module("scripts.optimize", run_name="__main__", alter_sys=True)
            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv
            sys.modules.update(saved_modules)

    def test_dunder_main_module_import_does_not_execute_main(self) -> None:
        """Test importing __main__.py directly doesn't execute main."""
        import importlib

        mod_name = "scripts.optimize.__main__"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        module: ModuleType = importlib.import_module(mod_name)
        assert module.__name__ == mod_name

    def test_main_with_multiple_backends(self) -> None:
        """Test main with multiple backends runs multi-backend mode."""
        backends_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if '"xgboost"' in config_json:
                backends_called.append("xgboost")
                return make_fake_result(dataset="taiwan", best_value=0.85)
            backends_called.append("lightgbm")
            return make_fake_lightgbm_result(dataset="taiwan", best_value=0.90)

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code = main(["-b", "xgboost,lightgbm", "-n", "5", "--no-save-model"])
            assert exit_code == 0
            assert len(backends_called) == 2
            assert "xgboost" in backends_called
            assert "lightgbm" in backends_called
        finally:
            _hooks.optimization_runner = original


class TestKeyboardInterrupt:
    """Tests for KeyboardInterrupt handling in main function."""

    def test_main_handles_keyboard_interrupt_gracefully(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test main exits gracefully on KeyboardInterrupt during execution."""

        def raising_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            raise KeyboardInterrupt()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = raising_runner
        try:
            exit_code: int = main(["-n", "1", "--no-save-model"])
            assert exit_code == 130

            captured = capsys.readouterr()
            output = captured.out + captured.err
            assert "Process Interrupted by User" in output
        finally:
            _hooks.optimization_runner = original
