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
from scripts._test_hooks import XGBoostOptimizationResult
from scripts.optimize import main

from .conftest import make_fake_result


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
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal call_count, callback_calls
            call_count += 1
            _ = phase_callback  # Available for phase reporting
            # Simulate calling progress callback as real runner would
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
                callback_calls += 1
                progress_callback(
                    {
                        "trial_number": 2,
                        "n_trials_total": 5,
                        "current_auc": 0.75,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": False,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
                callback_calls += 1
            return make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            exit_code: int = main(["-n", "5", "--no-save-model"])  # Small n_trials for test
            assert exit_code == 0
            assert call_count == 1
            assert callback_calls == 2  # Progress callback was invoked
        finally:
            _hooks.xgboost_runner = original

    def test_main_with_compare_presets(self) -> None:
        """Test main with -c runs compare presets."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal call_count
            call_count += 1
            _ = progress_callback  # Available for progress reporting
            _ = phase_callback  # Available for phase reporting
            return make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            exit_code: int = main(["-c", "-n", "5", "--no-save-model"])
            assert exit_code == 0
            assert call_count == 4  # All four presets
        finally:
            _hooks.xgboost_runner = original

    def test_main_with_all_datasets(self) -> None:
        """Test main with -a runs all datasets."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal call_count
            call_count += 1
            _ = progress_callback  # Available for progress reporting
            _ = phase_callback  # Available for phase reporting
            return make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            exit_code: int = main(["-a", "-n", "5", "--no-save-model"])
            assert exit_code == 0
            assert call_count == 3  # All three datasets
        finally:
            _hooks.xgboost_runner = original

    def test_main_with_verbose(self) -> None:
        """Test main with -v sets debug logging."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal call_count
            call_count += 1
            _ = progress_callback  # Available for progress reporting
            _ = phase_callback  # Available for phase reporting
            return make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            exit_code: int = main(["-v", "-n", "5", "--no-save-model"])
            assert exit_code == 0
            assert call_count == 1
        finally:
            _hooks.xgboost_runner = original

    def test_main_with_all_options(self) -> None:
        """Test main with multiple options."""
        configs_received: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            configs_received.append(config_json)
            return make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
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
            _hooks.xgboost_runner = original

    def test_main_with_timeout(self) -> None:
        """Test main with timeout option."""
        configs_received: list[str] = []

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
            configs_received.append(config_json)
            return make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            exit_code: int = main(["-n", "5", "-t", "120", "--no-save-model"])
            assert exit_code == 0
            assert len(configs_received) == 1
            assert "timeout_seconds" in configs_received[0]
            assert "120" in configs_received[0]
        finally:
            _hooks.xgboost_runner = original

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

        # Clear module from sys.modules to avoid runpy warning about
        # module already being imported
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
            # Restore modules
            sys.modules.update(saved_modules)

    def test_dunder_main_module_import_does_not_execute_main(self) -> None:
        """Test importing __main__.py directly doesn't execute main (covers False branch)."""
        import importlib

        # Clear the module from cache if present
        mod_name = "scripts.optimize.__main__"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        # Import the module - this covers the False branch of if __name__ == "__main__"
        module: ModuleType = importlib.import_module(mod_name)

        # Verify the module was imported but main wasn't called (no SystemExit)
        # The module should have the name "scripts.optimize.__main__", not "__main__"
        assert module.__name__ == mod_name
        # If we got here without SystemExit, the if __name__ == "__main__" was False


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
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            _ = config_json  # Unused
            _ = external_dir  # Unused
            _ = output_dir  # Unused
            _ = progress_callback  # Unused
            _ = phase_callback  # Unused
            raise KeyboardInterrupt()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = raising_runner
        try:
            exit_code: int = main(["-n", "1", "--no-save-model"])
            assert exit_code == 130

            # Verify output
            captured = capsys.readouterr()
            output = captured.out + captured.err
            assert "Process Interrupted by User" in output
        finally:
            _hooks.xgboost_runner = original
