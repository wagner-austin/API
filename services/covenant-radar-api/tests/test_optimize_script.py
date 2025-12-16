"""Tests for scripts/optimize.py CLI entry point.

Tests use dependency injection via scripts/_test_hooks to avoid real optimization runs.
All code paths are tested with strong assertions on actual behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import pytest
import scripts._test_hooks as _hooks
from covenant_ml.types import TrainConfig
from platform_core.logging import setup_rich_logging
from scripts.optimize import (
    PRESET_DESCRIPTIONS,
    FeaturePreset,
    OptimizeArgs,
    _compare_presets,
    _create_hyperparams_table,
    _create_result_table,
    _get_project_root,
    _handle_flag,
    _parse_args,
    _parse_dataset,
    _parse_preset,
    _print_config,
    _print_result,
    _run_all_datasets,
    _run_single,
    main,
)

from covenant_radar_api.worker.optimize_job import OptimizationResult

FeaturePresetLiteral = Literal["none", "log_only", "ratios_only", "full"]


@pytest.fixture(autouse=True)
def _setup_rich_logging_for_tests() -> None:
    """Set up rich logging before each test that needs it."""
    setup_rich_logging(level="WARNING", show_time=False)


def _make_fake_train_config() -> TrainConfig:
    """Create a fake TrainConfig for testing."""
    return TrainConfig(
        device="cpu",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        early_stopping_rounds=10,
        reg_alpha=0.01,
        reg_lambda=0.01,
    )


def _make_fake_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_val_auc: float = 0.85,
    n_features: int = 100,
) -> OptimizationResult:
    """Create a fake optimization result for testing."""
    return {
        "status": "complete",
        "dataset": dataset,
        "n_samples": 1000,
        "n_features": n_features,
        "feature_preset": feature_preset,
        "n_trials_complete": 10,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "best_trial_number": 5,
        "best_val_auc": best_val_auc,
        "best_max_depth": 6,
        "best_n_estimators": 100,
        "best_learning_rate": 0.1,
        "best_reg_alpha": 0.01,
        "best_reg_lambda": 0.01,
        "best_subsample": 0.8,
        "best_colsample_bytree": 0.8,
        "duration_seconds": 10.0,
        "recommended_config": _make_fake_train_config(),
    }


class TestOptimizeArgs:
    """Tests for OptimizeArgs initialization."""

    def test_default_values(self) -> None:
        """Test OptimizeArgs has correct defaults."""
        args = OptimizeArgs()
        assert args.dataset == "taiwan"
        assert args.n_trials == 300
        assert args.feature_preset == "full"
        assert args.device == "cuda"
        assert args.timeout is None
        assert args.compare_presets is False
        assert args.all_datasets is False
        assert args.verbose is False


class TestPresetDescriptions:
    """Tests for preset descriptions constant."""

    def test_all_presets_have_descriptions(self) -> None:
        """Verify all presets have descriptions."""
        presets: list[FeaturePreset] = ["none", "log_only", "ratios_only", "full"]
        for preset in presets:
            assert preset in PRESET_DESCRIPTIONS
            description: str = PRESET_DESCRIPTIONS[preset]
            # Description should contain meaningful content about features
            assert "features" in description.lower() or "original" in description.lower()


class TestParseDataset:
    """Tests for _parse_dataset function."""

    def test_parse_taiwan(self) -> None:
        """Test parsing taiwan dataset."""
        result = _parse_dataset("taiwan")
        assert result == "taiwan"

    def test_parse_us(self) -> None:
        """Test parsing us dataset."""
        result = _parse_dataset("us")
        assert result == "us"

    def test_parse_polish(self) -> None:
        """Test parsing polish dataset."""
        result = _parse_dataset("polish")
        assert result == "polish"

    def test_parse_invalid_raises_system_exit(self) -> None:
        """Test parsing invalid dataset raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_dataset("invalid")
        assert exc_info.value.code == 1


class TestParsePreset:
    """Tests for _parse_preset function."""

    def test_parse_none(self) -> None:
        """Test parsing none preset."""
        result = _parse_preset("none")
        assert result == "none"

    def test_parse_log_only(self) -> None:
        """Test parsing log_only preset."""
        result = _parse_preset("log_only")
        assert result == "log_only"

    def test_parse_ratios_only(self) -> None:
        """Test parsing ratios_only preset."""
        result = _parse_preset("ratios_only")
        assert result == "ratios_only"

    def test_parse_full(self) -> None:
        """Test parsing full preset."""
        result = _parse_preset("full")
        assert result == "full"

    def test_parse_invalid_raises_system_exit(self) -> None:
        """Test parsing invalid preset raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_preset("invalid")
        assert exc_info.value.code == 1


class TestHandleFlag:
    """Tests for _handle_flag function."""

    def test_compare_presets_short(self) -> None:
        """Test -c flag sets compare_presets."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "-c")
        assert handled is True
        assert args.compare_presets is True

    def test_compare_presets_long(self) -> None:
        """Test --compare-presets flag sets compare_presets."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "--compare-presets")
        assert handled is True
        assert args.compare_presets is True

    def test_all_datasets_short(self) -> None:
        """Test -a flag sets all_datasets."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "-a")
        assert handled is True
        assert args.all_datasets is True

    def test_all_datasets_long(self) -> None:
        """Test --all-datasets flag sets all_datasets."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "--all-datasets")
        assert handled is True
        assert args.all_datasets is True

    def test_verbose_short(self) -> None:
        """Test -v flag sets verbose."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "-v")
        assert handled is True
        assert args.verbose is True

    def test_verbose_long(self) -> None:
        """Test --verbose flag sets verbose."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "--verbose")
        assert handled is True
        assert args.verbose is True

    def test_help_short_raises_system_exit(self) -> None:
        """Test -h flag raises SystemExit(0)."""
        args = OptimizeArgs()
        with pytest.raises(SystemExit) as exc_info:
            _handle_flag(args, "-h")
        assert exc_info.value.code == 0

    def test_help_long_raises_system_exit(self) -> None:
        """Test --help flag raises SystemExit(0)."""
        args = OptimizeArgs()
        with pytest.raises(SystemExit) as exc_info:
            _handle_flag(args, "--help")
        assert exc_info.value.code == 0

    def test_unknown_flag_not_handled(self) -> None:
        """Test unknown flag returns False."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "--unknown")
        assert handled is False


class TestParseArgs:
    """Tests for _parse_args function."""

    def test_empty_args_uses_defaults(self) -> None:
        """Test empty args uses defaults."""
        args = _parse_args([])
        assert args.dataset == "taiwan"
        assert args.n_trials == 300
        assert args.feature_preset == "full"

    def test_dataset_short(self) -> None:
        """Test -d sets dataset."""
        args = _parse_args(["-d", "us"])
        assert args.dataset == "us"

    def test_dataset_long(self) -> None:
        """Test --dataset sets dataset."""
        args = _parse_args(["--dataset", "polish"])
        assert args.dataset == "polish"

    def test_n_trials_short(self) -> None:
        """Test -n sets n_trials."""
        args = _parse_args(["-n", "50"])
        assert args.n_trials == 50

    def test_n_trials_long(self) -> None:
        """Test --n-trials sets n_trials."""
        args = _parse_args(["--n-trials", "100"])
        assert args.n_trials == 100

    def test_feature_preset_short(self) -> None:
        """Test -f sets feature_preset."""
        args = _parse_args(["-f", "none"])
        assert args.feature_preset == "none"

    def test_feature_preset_long(self) -> None:
        """Test --feature-preset sets feature_preset."""
        args = _parse_args(["--feature-preset", "log_only"])
        assert args.feature_preset == "log_only"

    def test_device(self) -> None:
        """Test --device sets device."""
        args = _parse_args(["--device", "cpu"])
        assert args.device == "cpu"

    def test_timeout_short(self) -> None:
        """Test -t sets timeout."""
        args = _parse_args(["-t", "60"])
        assert args.timeout == 60

    def test_timeout_long(self) -> None:
        """Test --timeout sets timeout."""
        args = _parse_args(["--timeout", "120"])
        assert args.timeout == 120

    def test_verbose_flag(self) -> None:
        """Test -v flag."""
        args = _parse_args(["-v"])
        assert args.verbose is True

    def test_compare_presets_flag(self) -> None:
        """Test -c flag."""
        args = _parse_args(["-c"])
        assert args.compare_presets is True

    def test_all_datasets_flag(self) -> None:
        """Test -a flag."""
        args = _parse_args(["-a"])
        assert args.all_datasets is True

    def test_combined_args(self) -> None:
        """Test multiple args combined."""
        args = _parse_args(["-d", "us", "-n", "50", "-f", "none", "-v", "--device", "cpu"])
        assert args.dataset == "us"
        assert args.n_trials == 50
        assert args.feature_preset == "none"
        assert args.verbose is True
        assert args.device == "cpu"

    def test_unknown_args_ignored(self) -> None:
        """Test unknown args are ignored."""
        args = _parse_args(["--unknown", "value", "-x"])
        assert args.dataset == "taiwan"  # default


class TestGetProjectRoot:
    """Tests for _get_project_root function."""

    def test_returns_parent_of_scripts(self) -> None:
        """Test project root is parent of scripts directory."""
        root = _get_project_root()
        assert root.name == "covenant-radar-api"
        assert (root / "scripts").exists()


class TestCreateResultTable:
    """Tests for _create_result_table function."""

    def test_creates_table_with_data(self) -> None:
        """Test table is created with result data."""
        result = _make_fake_result()
        table = _create_result_table(result, 15.5)
        # Verify table has the expected protocol methods
        assert callable(table.add_column)
        assert callable(table.add_row)


class TestCreateHyperparamsTable:
    """Tests for _create_hyperparams_table function."""

    def test_creates_table_with_hyperparams(self) -> None:
        """Test table is created with hyperparameter data."""
        result = _make_fake_result()
        table = _create_hyperparams_table(result)
        # Verify table has the expected protocol methods
        assert callable(table.add_column)
        assert callable(table.add_row)


class TestRunSingle:
    """Tests for _run_single function."""

    def test_runs_optimization_with_hook(self, tmp_path: Path) -> None:
        """Test _run_single uses the optimization_runner hook."""
        fake_result = _make_fake_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
        ) -> OptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            result = _run_single("taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "taiwan" in config_json
            assert "10" in config_json
            assert "full" in config_json
        finally:
            _hooks.optimization_runner = original

    def test_includes_timeout_when_provided(self, tmp_path: Path) -> None:
        """Test _run_single includes timeout in config when provided."""
        fake_result = _make_fake_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
        ) -> OptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            _run_single("taiwan", 10, "full", "cpu", 60)
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "timeout_seconds" in config_json
            assert "60" in config_json
        finally:
            _hooks.optimization_runner = original


class TestPrintConfig:
    """Tests for _print_config function."""

    def test_prints_without_error(self) -> None:
        """Test _print_config runs without error."""
        # Just verify it doesn't raise
        _print_config("taiwan", 50, "full", "cuda")


class TestPrintResult:
    """Tests for _print_result function."""

    def test_prints_without_error(self) -> None:
        """Test _print_result runs without error."""
        result = _make_fake_result()
        _print_result(result, 10.5)


class TestComparePresets:
    """Tests for _compare_presets function."""

    def test_runs_all_presets(self) -> None:
        """Test _compare_presets runs all four presets."""
        presets_called: list[str] = []
        callback_calls = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
        ) -> OptimizationResult:
            nonlocal callback_calls
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
            # Extract preset from config
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
                return _make_fake_result(feature_preset="none", best_val_auc=0.75, n_features=20)
            if "log_only" in config_json:
                presets_called.append("log_only")
                return _make_fake_result(
                    feature_preset="log_only", best_val_auc=0.80, n_features=40
                )
            if "ratios_only" in config_json:
                presets_called.append("ratios_only")
                return _make_fake_result(
                    feature_preset="ratios_only", best_val_auc=0.82, n_features=500
                )
            presets_called.append("full")
            return _make_fake_result(feature_preset="full", best_val_auc=0.85, n_features=800)

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            _compare_presets("taiwan", 10, "cpu", None)
            assert len(presets_called) == 4
            assert "none" in presets_called
            assert "log_only" in presets_called
            assert "ratios_only" in presets_called
            assert "full" in presets_called
            assert callback_calls == 8  # 2 calls per preset * 4 presets
        finally:
            _hooks.optimization_runner = original


class TestRunAllDatasets:
    """Tests for _run_all_datasets function."""

    def test_runs_all_three_datasets(self) -> None:
        """Test _run_all_datasets runs on taiwan, us, and polish with varying AUCs."""
        datasets_called: list[str] = []
        callback_calls = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
        ) -> OptimizationResult:
            nonlocal callback_calls
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
            # Return different AUC values to cover both best and non-best formatting
            if "taiwan" in config_json:
                datasets_called.append("taiwan")
                return _make_fake_result(dataset="taiwan", best_val_auc=0.90)  # Best
            if '"us"' in config_json:
                datasets_called.append("us")
                return _make_fake_result(dataset="us", best_val_auc=0.85)  # Not best
            datasets_called.append("polish")
            return _make_fake_result(dataset="polish", best_val_auc=0.82)  # Not best

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            _run_all_datasets(10, "full", "cpu", None)
            assert len(datasets_called) == 3
            assert "taiwan" in datasets_called
            assert "us" in datasets_called
            assert "polish" in datasets_called
            assert callback_calls == 6  # 2 calls per dataset * 3 datasets
        finally:
            _hooks.optimization_runner = original


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
            progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
        ) -> OptimizationResult:
            nonlocal call_count, callback_calls
            call_count += 1
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
            return _make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code = main(["-n", "5"])  # Small n_trials for test
            assert exit_code == 0
            assert call_count == 1
            assert callback_calls == 2  # Progress callback was invoked
        finally:
            _hooks.optimization_runner = original

    def test_main_with_compare_presets(self) -> None:
        """Test main with -c runs compare presets."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
        ) -> OptimizationResult:
            nonlocal call_count
            call_count += 1
            return _make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code = main(["-c", "-n", "5"])
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
            progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
        ) -> OptimizationResult:
            nonlocal call_count
            call_count += 1
            return _make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code = main(["-a", "-n", "5"])
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
            progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
        ) -> OptimizationResult:
            nonlocal call_count
            call_count += 1
            return _make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code = main(["-v", "-n", "5"])
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
            progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
        ) -> OptimizationResult:
            configs_received.append(config_json)
            return _make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code = main(["-d", "us", "-n", "25", "-f", "log_only", "--device", "cpu"])
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
            progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
        ) -> OptimizationResult:
            configs_received.append(config_json)
            return _make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            exit_code = main(["-n", "5", "-t", "120"])
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
        from types import ModuleType

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
            progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
        ) -> OptimizationResult:
            raise KeyboardInterrupt()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = raising_runner
        try:
            exit_code = main(["-n", "1"])
            assert exit_code == 130

            # Verify output
            captured = capsys.readouterr()
            output = captured.out + captured.err
            assert "Process Interrupted by User" in output
        finally:
            _hooks.optimization_runner = original
