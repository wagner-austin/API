"""Tests for scripts/optimize CLI argument parsing.

Tests command-line argument parsing, validation, and flag handling.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import pytest
from scripts.optimize.cli import (
    ALL_STANDARD_DATASETS,
    ALL_TIMESERIES_DATASETS,
    PRESET_DESCRIPTIONS,
    FeaturePreset,
    OptimizeArgs,
    _handle_flag,
    _parse_backend,
    _parse_dataset,
    _parse_preset,
    is_timeseries_dataset,
    parse_args,
)


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


class TestParseBackend:
    """Tests for _parse_backend function."""

    def test_parse_xgboost(self) -> None:
        """Test parsing xgboost backend."""
        result: str = _parse_backend("xgboost")
        assert result == "xgboost"

    def test_parse_mlp(self) -> None:
        """Test parsing mlp backend."""
        result: str = _parse_backend("mlp")
        assert result == "mlp"

    def test_parse_lightgbm(self) -> None:
        """Test parsing lightgbm backend."""
        result: str = _parse_backend("lightgbm")
        assert result == "lightgbm"

    def test_parse_lstm(self) -> None:
        """Test parsing lstm backend."""
        result: str = _parse_backend("lstm")
        assert result == "lstm"

    def test_parse_invalid_raises_system_exit(self) -> None:
        """Test parsing invalid backend raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_backend("invalid")
        assert exc_info.value.code == 1


class TestParseDataset:
    """Tests for _parse_dataset function."""

    def test_parse_taiwan(self) -> None:
        """Test parsing taiwan dataset."""
        result: str = _parse_dataset("taiwan")
        assert result == "taiwan"

    def test_parse_us(self) -> None:
        """Test parsing us dataset."""
        result: str = _parse_dataset("us")
        assert result == "us"

    def test_parse_polish(self) -> None:
        """Test parsing polish dataset."""
        result: str = _parse_dataset("polish")
        assert result == "polish"

    def test_parse_kaggle_give_me_credit(self) -> None:
        """Test parsing kaggle_give_me_credit dataset."""
        result: str = _parse_dataset("kaggle_give_me_credit")
        assert result == "kaggle_give_me_credit"

    def test_parse_kaggle_amex_default_timeseries(self) -> None:
        """Test parsing kaggle_amex_default time-series dataset."""
        result: str = _parse_dataset("kaggle_amex_default")
        assert result == "kaggle_amex_default"

    def test_parse_invalid_raises_system_exit(self) -> None:
        """Test parsing invalid dataset raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_dataset("invalid")
        assert exc_info.value.code == 1


class TestIsTimeseriesDataset:
    """Tests for is_timeseries_dataset function."""

    def test_standard_datasets_return_false(self) -> None:
        """Test standard datasets return False."""
        for dataset in ALL_STANDARD_DATASETS:
            assert is_timeseries_dataset(dataset) is False

    def test_timeseries_datasets_return_true(self) -> None:
        """Test time-series datasets return True."""
        for dataset in ALL_TIMESERIES_DATASETS:
            assert is_timeseries_dataset(dataset) is True

    def test_kaggle_amex_default_is_timeseries(self) -> None:
        """Test kaggle_amex_default is a time-series dataset."""
        assert is_timeseries_dataset("kaggle_amex_default") is True

    def test_taiwan_is_not_timeseries(self) -> None:
        """Test taiwan is not a time-series dataset."""
        assert is_timeseries_dataset("taiwan") is False


class TestParsePreset:
    """Tests for _parse_preset function."""

    def test_parse_none(self) -> None:
        """Test parsing none preset."""
        result: str = _parse_preset("none")
        assert result == "none"

    def test_parse_log_only(self) -> None:
        """Test parsing log_only preset."""
        result: str = _parse_preset("log_only")
        assert result == "log_only"

    def test_parse_ratios_only(self) -> None:
        """Test parsing ratios_only preset."""
        result: str = _parse_preset("ratios_only")
        assert result == "ratios_only"

    def test_parse_full(self) -> None:
        """Test parsing full preset."""
        result: str = _parse_preset("full")
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
    """Tests for parse_args function."""

    def test_empty_args_uses_defaults(self) -> None:
        """Test empty args uses defaults."""
        args: OptimizeArgs = parse_args([])
        assert args.dataset == "taiwan"
        assert args.n_trials == 300
        assert args.feature_preset == "full"

    def test_backend_short(self) -> None:
        """Test -b sets backend."""
        args: OptimizeArgs = parse_args(["-b", "lightgbm"])
        assert args.backend == "lightgbm"

    def test_backend_long(self) -> None:
        """Test --backend sets backend."""
        args: OptimizeArgs = parse_args(["--backend", "lstm"])
        assert args.backend == "lstm"

    def test_dataset_short(self) -> None:
        """Test -d sets dataset."""
        args: OptimizeArgs = parse_args(["-d", "us"])
        assert args.dataset == "us"

    def test_dataset_long(self) -> None:
        """Test --dataset sets dataset."""
        args: OptimizeArgs = parse_args(["--dataset", "polish"])
        assert args.dataset == "polish"

    def test_n_trials_short(self) -> None:
        """Test -n sets n_trials."""
        args: OptimizeArgs = parse_args(["-n", "50"])
        assert args.n_trials == 50

    def test_n_trials_long(self) -> None:
        """Test --n-trials sets n_trials."""
        args: OptimizeArgs = parse_args(["--n-trials", "100"])
        assert args.n_trials == 100

    def test_feature_preset_short(self) -> None:
        """Test -f sets feature_preset."""
        args: OptimizeArgs = parse_args(["-f", "none"])
        assert args.feature_preset == "none"

    def test_feature_preset_long(self) -> None:
        """Test --feature-preset sets feature_preset."""
        args: OptimizeArgs = parse_args(["--feature-preset", "log_only"])
        assert args.feature_preset == "log_only"

    def test_device(self) -> None:
        """Test --device sets device."""
        args: OptimizeArgs = parse_args(["--device", "cpu"])
        assert args.device == "cpu"

    def test_timeout_short(self) -> None:
        """Test -t sets timeout."""
        args: OptimizeArgs = parse_args(["-t", "60"])
        assert args.timeout == 60

    def test_timeout_long(self) -> None:
        """Test --timeout sets timeout."""
        args: OptimizeArgs = parse_args(["--timeout", "120"])
        assert args.timeout == 120

    def test_verbose_flag(self) -> None:
        """Test -v flag."""
        args: OptimizeArgs = parse_args(["-v"])
        assert args.verbose is True

    def test_compare_presets_flag(self) -> None:
        """Test -c flag."""
        args: OptimizeArgs = parse_args(["-c"])
        assert args.compare_presets is True

    def test_all_datasets_flag(self) -> None:
        """Test -a flag."""
        args: OptimizeArgs = parse_args(["-a"])
        assert args.all_datasets is True

    def test_save_model_flag_short(self) -> None:
        """Test -s flag sets save_model to True."""
        args: OptimizeArgs = parse_args(["-s"])
        assert args.save_model is True

    def test_save_model_flag_long(self) -> None:
        """Test --save-model flag sets save_model to True."""
        args: OptimizeArgs = parse_args(["--save-model"])
        assert args.save_model is True

    def test_combined_args(self) -> None:
        """Test multiple args combined."""
        args: OptimizeArgs = parse_args(
            ["-d", "us", "-n", "50", "-f", "none", "-v", "--device", "cpu"]
        )
        assert args.dataset == "us"
        assert args.n_trials == 50
        assert args.feature_preset == "none"
        assert args.verbose is True
        assert args.device == "cpu"

    def test_unknown_args_ignored(self) -> None:
        """Test unknown args are ignored."""
        args: OptimizeArgs = parse_args(["--unknown", "value", "-x"])
        assert args.dataset == "taiwan"  # default
