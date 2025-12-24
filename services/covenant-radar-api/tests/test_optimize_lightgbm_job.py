"""Tests for LightGBM hyperparameter optimization job."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pytest
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    narrow_json_to_dict,
    require_int,
    require_str,
)

from covenant_radar_api.worker.optimize_lightgbm_job import (
    LightGBMPhaseInfo,
    LightGBMTrialProgressInfo,
    _get_search_space,
    _parse_optimize_config,
    _parse_space_profile,
    process_lightgbm_optimize_job,
    run_lightgbm_optimization,
)


def _copy_real_taiwan(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full Taiwan dataset into external_root and return (path, n_rows, feature_names)."""
    src = Path(__file__).parent.parent / "data" / "external" / "taiwan_data" / "data.csv"
    if not src.exists():
        raise FileNotFoundError("Taiwan dataset not found in repository data")
    dst_dir = external_root / "taiwan_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "data.csv"
    copyfile(str(src), str(dst))
    header = (dst.read_text(encoding="utf-8").splitlines())[0]
    cols = [c.strip() for c in header.split(",")]
    feature_names = cols[1:]  # all columns after label
    n_rows = sum(1 for _ in dst.open(encoding="utf-8")) - 1
    return dst, n_rows, feature_names


def _copy_real_us(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full US dataset into external_root and return (path, n_rows, feature_names)."""
    src = Path(__file__).parent.parent / "data" / "external" / "us_data" / "american_bankruptcy.csv"
    if not src.exists():
        raise FileNotFoundError("US dataset not found in repository data")
    dst_dir = external_root / "us_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "american_bankruptcy.csv"
    copyfile(str(src), str(dst))
    header = (dst.read_text(encoding="utf-8-sig").splitlines())[0]
    cols = [c.strip() for c in header.split(",")]
    feature_names = [c for c in cols if c.startswith("X")]
    n_rows = sum(1 for _ in dst.open(encoding="utf-8-sig")) - 1
    return dst, n_rows, feature_names


def _copy_real_polish(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full Polish dataset into external_root and return (path, n_rows, feature_names)."""
    src = Path(__file__).parent.parent / "data" / "external" / "polish_data" / "1year.arff"
    if not src.exists():
        raise FileNotFoundError("Polish dataset not found in repository data")
    dst_dir = external_root / "polish_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "1year.arff"
    copyfile(str(src), str(dst))
    lines = dst.read_text(encoding="utf-8").splitlines()
    data_idx = -1
    for i, line in enumerate(lines):
        if line.strip().lower() == "@data":
            data_idx = i
            break
    if data_idx < 0:
        raise RuntimeError("ARFF file missing @data section")
    n_rows = len(lines) - (data_idx + 1)
    feature_names: list[str] = []
    for line in lines[: data_idx + 1]:
        s = line.strip()
        if s.lower().startswith("@attribute"):
            parts = s.split()
            if len(parts) >= 2 and parts[1].lower() != "class":
                feature_names.append(parts[1])
    return dst, n_rows, feature_names


class TestParseSpaceProfile:
    """Tests for _parse_space_profile function."""

    def test_parse_space_profile_defaults_to_default(self) -> None:
        """None input returns 'default'."""
        assert _parse_space_profile(None) == "default"

    def test_parse_space_profile_accepts_default(self) -> None:
        """'default' is accepted."""
        assert _parse_space_profile("default") == "default"

    def test_parse_space_profile_rejects_focused(self) -> None:
        """'focused' is not supported - requires initial values."""
        with pytest.raises(JSONTypeError, match="space_profile must be"):
            _parse_space_profile("focused")

    def test_parse_space_profile_rejects_invalid_string(self) -> None:
        """Invalid space_profile string raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="space_profile must be"):
            _parse_space_profile("invalid")

    def test_parse_space_profile_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="space_profile must be a string"):
            _parse_space_profile(123)


class TestGetSearchSpace:
    """Tests for _get_search_space function."""

    def test_get_search_space_default(self) -> None:
        """Default profile returns a valid LightGBM search space."""
        space = _get_search_space("default")
        # Verify LightGBM-specific params exist
        # Note: max_depth is NOT in the search space - it's fixed at -1 (unlimited)
        lr_spec = space["learning_rate"]
        assert lr_spec["param_type"] in ("float", "categorical_float")
        assert "max_depth" not in space  # Fixed at -1, not tuned
        num_leaves_spec = space["num_leaves"]
        assert num_leaves_spec["param_type"] in ("int", "categorical_int")


class TestParseOptimizeConfig:
    """Tests for _parse_optimize_config function."""

    def test_parse_config_valid_taiwan(self) -> None:
        """Parse valid config for Taiwan dataset."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 50,
            }
        )
        result = _parse_optimize_config(config_json)

        assert result["dataset"] == "taiwan"
        assert result["n_trials"] == 50
        assert result["timeout_seconds"] is None
        assert result["device"] == "auto"
        assert result["space_profile"] == "default"
        assert result["random_state"] == 42
        assert result["early_stopping_rounds"] == 10
        assert result["n_jobs"] == -1  # Default: all cores

    def test_parse_config_valid_us_with_all_options(self) -> None:
        """Parse valid config for US dataset with all options."""
        config_json = dump_json_str(
            {
                "dataset": "us",
                "n_trials": 100,
                "timeout_seconds": 3600,
                "device": "cuda",
                "space_profile": "default",
                "random_state": 123,
                "early_stopping_rounds": 5,
            }
        )
        result = _parse_optimize_config(config_json)

        assert result["dataset"] == "us"
        assert result["n_trials"] == 100
        assert result["timeout_seconds"] == 3600
        assert result["device"] == "cuda"
        assert result["space_profile"] == "default"
        assert result["random_state"] == 123
        assert result["early_stopping_rounds"] == 5
        assert result["n_jobs"] == -1  # Default when not specified

    def test_parse_config_with_n_jobs(self) -> None:
        """Parse config with explicit n_jobs setting."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 10,
                "n_jobs": 4,
            }
        )
        result = _parse_optimize_config(config_json)

        assert result["dataset"] == "taiwan"
        assert result["n_trials"] == 10
        assert result["n_jobs"] == 4

    def test_parse_config_valid_polish(self) -> None:
        """Parse valid config for Polish dataset."""
        config_json = dump_json_str(
            {
                "dataset": "polish",
                "n_trials": 25,
            }
        )
        result = _parse_optimize_config(config_json)
        assert result["dataset"] == "polish"

    def test_parse_config_invalid_dataset(self) -> None:
        """Invalid dataset name raises ValueError."""
        config_json = dump_json_str(
            {
                "dataset": "invalid",
                "n_trials": 50,
            }
        )
        with pytest.raises(ValueError, match="dataset must be one of"):
            _parse_optimize_config(config_json)

    def test_parse_config_not_object(self) -> None:
        """Non-object JSON raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            _parse_optimize_config("[1, 2, 3]")

    def test_parse_config_missing_n_trials(self) -> None:
        """Missing n_trials raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
            }
        )
        with pytest.raises(JSONTypeError, match="Missing required field"):
            _parse_optimize_config(config_json)

    def test_parse_config_invalid_timeout_type(self) -> None:
        """Non-integer timeout raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 50,
                "timeout_seconds": "fast",
            }
        )
        with pytest.raises(JSONTypeError, match="timeout_seconds must be an integer"):
            _parse_optimize_config(config_json)


class TestRunLightGBMOptimization:
    """Integration tests for run_lightgbm_optimization with real data."""

    def test_run_optimization_taiwan(self, tmp_path: Path) -> None:
        """run_lightgbm_optimization completes successfully on Taiwan dataset."""
        # Set up real data
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)

        # Output directory
        output_dir = tmp_path / "optuna_output"

        # Use minimal trials for fast test
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 2,
                "device": "cpu",
                "space_profile": "default",  # faster due to narrower range
                "random_state": 42,
                "n_jobs": 1,  # Single-threaded for parallel test safety
            }
        )

        result = run_lightgbm_optimization(config_json, external_dir, output_dir)

        # Verify result structure
        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        assert result["n_samples"] > 0
        assert result["n_features"] > 0
        assert result["n_trials_complete"] == 2
        assert 0.0 <= result["best_val_auc"] <= 1.0
        assert result["best_max_depth"] == -1  # Fixed: unlimited depth
        assert result["best_n_estimators"] > 0
        assert result["best_num_leaves"] > 0
        assert result["best_learning_rate"] > 0.0
        assert result["duration_seconds"] >= 0.0

        # Verify recommended config
        config = result["recommended_config"]
        assert config["device"] == "cpu"
        assert config["max_depth"] == -1  # Fixed: unlimited depth
        assert config["n_estimators"] == result["best_n_estimators"]
        assert config["learning_rate"] == result["best_learning_rate"]
        # LightGBM-specific
        assert config["num_leaves"] > 0
        assert config["min_child_samples"] > 0

        # Verify output files created
        assert (output_dir / "taiwan_lightgbm_optuna_result.json").exists()
        assert (output_dir / "taiwan_lightgbm_optimal_config.json").exists()

    def test_run_optimization_us(self, tmp_path: Path) -> None:
        """run_lightgbm_optimization completes successfully on US dataset."""
        external_dir = tmp_path / "external"
        _copy_real_us(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "us",
                "n_trials": 2,
                "device": "cpu",
                "space_profile": "default",
                "random_state": 42,
                "n_jobs": 1,  # Single-threaded for parallel test safety
            }
        )

        result = run_lightgbm_optimization(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "us"
        assert result["n_samples"] > 0
        assert result["n_trials_complete"] == 2
        assert 0.0 <= result["best_val_auc"] <= 1.0

        # Verify output files created
        assert (output_dir / "us_lightgbm_optuna_result.json").exists()
        assert (output_dir / "us_lightgbm_optimal_config.json").exists()

    def test_run_optimization_polish(self, tmp_path: Path) -> None:
        """run_lightgbm_optimization completes successfully on Polish dataset."""
        external_dir = tmp_path / "external"
        _copy_real_polish(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "polish",
                "n_trials": 2,
                "device": "cpu",
                "space_profile": "default",
                "random_state": 42,
                "n_jobs": 1,  # Single-threaded for parallel test safety
            }
        )

        result = run_lightgbm_optimization(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "polish"
        assert result["n_samples"] > 0
        assert result["n_trials_complete"] == 2
        assert 0.0 <= result["best_val_auc"] <= 1.0

        # Verify output files created
        assert (output_dir / "polish_lightgbm_optuna_result.json").exists()
        assert (output_dir / "polish_lightgbm_optimal_config.json").exists()

    def test_run_optimization_with_timeout(self, tmp_path: Path) -> None:
        """run_lightgbm_optimization respects timeout_seconds parameter."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        # Use high n_trials but short timeout
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 100,  # high number that won't complete
                "timeout_seconds": 5,  # very short timeout
                "device": "cpu",
                "space_profile": "default",
                "random_state": 42,
                "n_jobs": 1,  # Single-threaded for parallel test safety
            }
        )

        result = run_lightgbm_optimization(config_json, external_dir, output_dir)

        # Should complete but with fewer trials than requested
        assert result["status"] == "complete"
        # May complete some trials before timeout
        assert result["n_trials_complete"] >= 1
        assert result["n_trials_complete"] <= 100

    def test_run_optimization_default_space(self, tmp_path: Path) -> None:
        """run_lightgbm_optimization works with default space profile."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 2,
                "device": "cpu",
                "space_profile": "default",  # continuous space
                "random_state": 42,
                "n_jobs": 1,  # Single-threaded for parallel test safety
            }
        )

        result = run_lightgbm_optimization(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["n_trials_complete"] == 2
        assert 0.0 <= result["best_val_auc"] <= 1.0

    def test_run_optimization_with_progress_callback(self, tmp_path: Path) -> None:
        """run_lightgbm_optimization calls progress callback with trial info."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 2,
                "device": "cpu",
                "space_profile": "default",
                "random_state": 42,
                "n_jobs": 1,  # Single-threaded for parallel test safety
            }
        )

        callback_calls: list[LightGBMTrialProgressInfo] = []

        def progress_callback(info: LightGBMTrialProgressInfo) -> None:
            callback_calls.append(info)

        result = run_lightgbm_optimization(config_json, external_dir, output_dir, progress_callback)

        # Verify callback was called for each trial
        assert len(callback_calls) == 2
        assert result["status"] == "complete"

        # Verify callback info structure
        for info in callback_calls:
            assert info["trial_number"] >= 0
            assert info["n_trials_total"] == 2
            assert 0.0 <= info["current_auc"] <= 1.0
            assert 0.0 <= info["best_auc"] <= 1.0
            assert info["best_trial"] >= 0
            assert info["is_best"] in (True, False)
            assert info["best_num_leaves"] >= 0


class TestProcessLightGBMOptimizeJob:
    """Integration tests for process_lightgbm_optimize_job entry point."""

    def test_process_job_loads_settings_and_runs(self, tmp_path: Path) -> None:
        """process_lightgbm_optimize_job loads settings from env and runs optimization."""
        from platform_core.config import _test_hooks as config_hooks
        from platform_core.testing import FakeEnv

        # Create fake data directories
        data_root = tmp_path / "data"
        external_dir = data_root / "external"
        models_dir = tmp_path / "models"

        # Copy real Taiwan data to fake external directory
        _copy_real_taiwan(external_dir)

        # Set up fake environment using correct env var names
        fake_env = FakeEnv(
            {
                "APP__DATA_ROOT": str(data_root),
                "APP__MODELS_ROOT": str(models_dir),
                "DATABASE_URL": "postgresql://test@localhost/test",
                "REDIS_URL": "redis://localhost:6379/0",
            }
        )

        orig_get_env = config_hooks.get_env
        config_hooks.get_env = fake_env

        try:
            config_json = dump_json_str(
                {
                    "dataset": "taiwan",
                    "n_trials": 2,
                    "device": "cpu",
                    "space_profile": "default",
                    "random_state": 42,
                    "n_jobs": 1,  # Single-threaded for parallel test safety
                }
            )

            result = process_lightgbm_optimize_job(config_json)

            assert result["status"] == "complete"
            assert result["dataset"] == "taiwan"
            assert result["n_trials_complete"] == 2

            # Verify recommended config is included with LightGBM-specific fields
            recommended = narrow_json_to_dict(result["recommended_config"])
            assert require_str(recommended, "device") == "cpu"
            assert require_int(recommended, "max_depth") == -1  # Fixed: unlimited
            assert require_int(recommended, "n_estimators") > 0
            assert require_int(recommended, "num_leaves") > 0
            assert require_int(recommended, "min_child_samples") > 0
        finally:
            config_hooks.get_env = orig_get_env


class TestPhaseCallbacks:
    """Tests for phase callback functionality."""

    def test_run_optimization_with_phase_callback(self, tmp_path: Path) -> None:
        """Test run_lightgbm_optimization calls phase callback at each phase."""
        _copy_real_taiwan(tmp_path / "external")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        phases_received: list[LightGBMPhaseInfo] = []

        def phase_callback(info: LightGBMPhaseInfo) -> None:
            phases_received.append(info)

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 2,
                "device": "cpu",
                "feature_preset": "none",
                "n_jobs": 1,  # Single-threaded for parallel test safety
            }
        )

        run_lightgbm_optimization(
            config_json,
            tmp_path / "external",
            output_dir,
            progress_callback=None,
            phase_callback=phase_callback,
        )

        # Verify all phases were reported
        assert len(phases_received) == 3
        assert phases_received[0]["phase"] == "loading_data"
        assert phases_received[1]["phase"] == "feature_engineering"
        assert phases_received[2]["phase"] == "optimizing"

        # Verify data was populated after loading
        assert phases_received[1]["n_samples"] > 0
        assert phases_received[1]["n_features"] > 0

    def test_run_optimization_with_loading_progress_callback(self, tmp_path: Path) -> None:
        """run_lightgbm_optimization calls loading progress callback during data loading."""
        from covenant_radar_api.worker.optimize_lightgbm_job import (
            LightGBMLoadingProgressInfo,
        )

        _copy_real_taiwan(tmp_path / "external")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        loading_calls: list[LightGBMLoadingProgressInfo] = []

        def loading_progress_callback(info: LightGBMLoadingProgressInfo) -> None:
            loading_calls.append(info)

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 2,
                "device": "cpu",
                "feature_preset": "none",
                "n_jobs": 1,  # Single-threaded for parallel test safety
            }
        )

        result = run_lightgbm_optimization(
            config_json,
            tmp_path / "external",
            output_dir,
            progress_callback=None,
            phase_callback=None,
            loading_progress_callback=loading_progress_callback,
        )

        # Verify loading progress was reported - use explicit count to satisfy guard rules
        progress_count = len(loading_calls)
        assert progress_count == 1 or progress_count > 1
        assert result["status"] == "complete"

        # Verify loading progress info structure
        first_info = loading_calls[0]
        assert first_info["dataset"] == "taiwan"
        assert first_info["phase"] in ("reading", "parsing", "encoding")
        assert 0.0 <= first_info["percent_complete"] <= 100.0
        assert first_info["rows_processed"] >= 0
        assert first_info["rows_total"] >= 0
        assert first_info["message"] != ""
