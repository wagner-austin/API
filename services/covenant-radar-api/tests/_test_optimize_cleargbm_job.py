"""Tests for ClearGBM hyperparameter optimization job.

Tests for parsing, configuration, and running ClearGBM optimization.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pytest
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    narrow_json_to_dict,
    require_int,
)

from covenant_radar_api.worker.optimize_cleargbm_job import (
    ClearGBMLoadingProgressInfo,
    ClearGBMPhaseInfo,
    ClearGBMTrialProgressInfo,
    _generate_cleargbm_config,
    _get_search_space,
    _parse_optimize_config,
    _parse_space_profile,
    process_cleargbm_optimize_job,
    run_cleargbm_optimization,
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


class TestParseSpaceProfile:
    """Tests for _parse_space_profile function."""

    def test_parse_space_profile_defaults_to_default(self) -> None:
        """None input returns 'default'."""
        assert _parse_space_profile(None) == "default"

    def test_parse_space_profile_accepts_default(self) -> None:
        """'default' is accepted."""
        assert _parse_space_profile("default") == "default"

    def test_parse_space_profile_rejects_invalid_string(self) -> None:
        """Invalid space_profile string raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="space_profile must be: default"):
            _parse_space_profile("invalid")

    def test_parse_space_profile_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="space_profile must be a string"):
            _parse_space_profile(123)


class TestGetSearchSpace:
    """Tests for _get_search_space function."""

    def test_get_search_space_default(self) -> None:
        """Default profile returns a valid search space."""
        space = _get_search_space("default")
        # Access keys directly to verify they exist and have expected types
        lr_spec = space["learning_rate"]
        assert lr_spec["param_type"] == "float"
        depth_spec = space["max_depth"]
        assert depth_spec["param_type"] == "int"
        est_spec = space["n_estimators"]
        assert est_spec["param_type"] == "int"


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
        assert result["space_profile"] == "default"
        assert result["random_state"] == 42
        assert result["feature_preset"] == "none"
        assert result["early_stopping_rounds"] == 10

    def test_parse_config_with_all_options(self) -> None:
        """Parse config with all options specified."""
        config_json = dump_json_str(
            {
                "dataset": "us",
                "n_trials": 100,
                "timeout_seconds": 3600,
                "space_profile": "default",
                "random_state": 123,
                "feature_preset": "full",
                "early_stopping_rounds": 20,
            }
        )
        result = _parse_optimize_config(config_json)

        assert result["dataset"] == "us"
        assert result["n_trials"] == 100
        assert result["timeout_seconds"] == 3600
        assert result["space_profile"] == "default"
        assert result["random_state"] == 123
        assert result["feature_preset"] == "full"
        assert result["early_stopping_rounds"] == 20

    def test_parse_config_not_object(self) -> None:
        """Non-object JSON raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            _parse_optimize_config("[1, 2, 3]")

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


class TestGenerateClearGBMConfig:
    """Tests for _generate_cleargbm_config function."""

    def test_generate_config_from_summary(self) -> None:
        """_generate_cleargbm_config creates correct ClearGBMConfig from summary."""
        from covenant_ml.optimizer import OptimizationSummary

        summary: OptimizationSummary = {
            "n_trials_total": 50,
            "n_trials_complete": 50,
            "n_trials_pruned": 0,
            "n_trials_failed": 0,
            "best_trial_number": 25,
            "best_value": 0.85,
            "best_int_params": {
                "max_depth": 5,
                "n_estimators": 100,
                "min_samples_split": 15,
                "min_samples_leaf": 8,
                "max_bins": 128,
            },
            "best_float_params": {
                "learning_rate": 0.1,
                "subsample": 0.9,
            },
            "best_string_params": {},
            "total_duration_seconds": 120.5,
        }

        config = _generate_cleargbm_config(summary)

        assert config["learning_rate"] == 0.1
        assert config["max_depth"] == 5
        assert config["n_estimators"] == 100
        assert config["min_samples_split"] == 15
        assert config["min_samples_leaf"] == 8
        assert config["max_bins"] == 128
        assert config["subsample"] == 0.9
        assert config["random_state"] == 42
        assert config["train_ratio"] == 0.7
        assert config["val_ratio"] == 0.15
        assert config["test_ratio"] == 0.15
        assert config["early_stopping_rounds"] == 10
        assert config["track_contributions"] is True

    def test_generate_config_uses_defaults_for_missing(self) -> None:
        """_generate_cleargbm_config uses defaults when params are missing."""
        from covenant_ml.optimizer import OptimizationSummary

        summary: OptimizationSummary = {
            "n_trials_total": 10,
            "n_trials_complete": 10,
            "n_trials_pruned": 0,
            "n_trials_failed": 0,
            "best_trial_number": 5,
            "best_value": 0.80,
            "best_int_params": {
                "max_depth": 3,
                "n_estimators": 50,
            },
            "best_float_params": {
                "learning_rate": 0.05,
            },
            "best_string_params": {},
            "total_duration_seconds": 60.0,
        }

        config = _generate_cleargbm_config(summary)

        # Verify defaults are used
        assert config["min_samples_split"] == 10
        assert config["min_samples_leaf"] == 5
        assert config["max_bins"] == 64
        assert config["subsample"] == 1.0


class TestRunClearGBMOptimization:
    """Integration tests for run_cleargbm_optimization with real data."""

    def test_run_optimization_taiwan(self, tmp_path: Path) -> None:
        """run_cleargbm_optimization completes successfully on Taiwan dataset."""
        # Set up real data
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)

        # Output directory
        output_dir = tmp_path / "optuna_output"

        # Use minimal trials for fast test
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 1,
                "random_state": 42,
            }
        )

        result = run_cleargbm_optimization(config_json, external_dir, output_dir)

        # Verify result structure
        assert result["backend"] == "cleargbm"
        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        assert result["n_samples"] > 0
        assert result["n_features"] > 0
        assert result["n_trials_complete"] == 1
        assert 0.0 <= result["best_val_auc"] <= 1.0
        assert result["best_max_depth"] > 0
        assert result["best_n_estimators"] > 0
        assert result["best_learning_rate"] > 0.0
        assert result["duration_seconds"] >= 0.0

        # Verify recommended config
        config = result["recommended_config"]
        assert config["max_depth"] == result["best_max_depth"]
        assert config["n_estimators"] == result["best_n_estimators"]
        assert config["learning_rate"] == result["best_learning_rate"]

        # Verify output files created
        assert (output_dir / "taiwan_cleargbm_optuna_result.json").exists()
        assert (output_dir / "taiwan_cleargbm_optimal_config.json").exists()

    def test_run_optimization_with_timeout(self, tmp_path: Path) -> None:
        """run_cleargbm_optimization respects timeout_seconds parameter."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        # Use high n_trials but short timeout
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 100,  # high number that won't complete
                "timeout_seconds": 5,  # very short timeout
                "random_state": 42,
            }
        )

        result = run_cleargbm_optimization(config_json, external_dir, output_dir)

        # Should complete but with fewer trials than requested
        assert result["status"] == "complete"
        # May complete some trials before timeout
        assert result["n_trials_complete"] >= 1
        assert result["n_trials_complete"] <= 100

    def test_run_optimization_with_progress_callback(self, tmp_path: Path) -> None:
        """run_cleargbm_optimization calls progress callback with trial info."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 1,
                "random_state": 42,
            }
        )

        callback_calls: list[ClearGBMTrialProgressInfo] = []

        def progress_callback(info: ClearGBMTrialProgressInfo) -> None:
            callback_calls.append(info)

        result = run_cleargbm_optimization(config_json, external_dir, output_dir, progress_callback)

        # Verify callback was called for each trial
        assert len(callback_calls) == 1
        assert result["status"] == "complete"

        # Verify callback info structure
        for info in callback_calls:
            assert info["trial_number"] >= 0
            assert info["n_trials_total"] == 1
            assert 0.0 <= info["current_auc"] <= 1.0
            assert 0.0 <= info["best_auc"] <= 1.0
            assert info["best_trial"] >= 0
            assert info["is_best"] in (True, False)

    def test_run_optimization_with_multiple_trials_covers_non_best(self, tmp_path: Path) -> None:
        """Test progress callback covers both is_best=True and is_best=False paths.

        With multiple trials, at least one trial won't be the best (unless all
        trials have identical AUC, which is statistically improbable).
        """
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 3,  # Three trials to ensure at least one is_best=False
                "random_state": 42,
            }
        )

        callback_calls: list[ClearGBMTrialProgressInfo] = []

        def progress_callback(info: ClearGBMTrialProgressInfo) -> None:
            callback_calls.append(info)

        result = run_cleargbm_optimization(config_json, external_dir, output_dir, progress_callback)

        # Verify all trials completed
        assert len(callback_calls) == 3
        assert result["status"] == "complete"

        # First trial should always be best (any AUC > 0.0 initial)
        assert callback_calls[0]["is_best"] is True

        # With 3 trials, at least one should NOT be best (covers is_best=False branch)
        non_best_count = sum(1 for info in callback_calls if not info["is_best"])
        assert non_best_count >= 1, "Expected at least one non-best trial for branch coverage"

    def test_run_optimization_with_phase_callback(self, tmp_path: Path) -> None:
        """Test phase callback for loading, feature engineering, optimizing."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 1,
                "random_state": 42,
            }
        )

        phase_calls: list[ClearGBMPhaseInfo] = []

        def phase_callback(info: ClearGBMPhaseInfo) -> None:
            phase_calls.append(info)

        result = run_cleargbm_optimization(
            config_json, external_dir, output_dir, None, phase_callback
        )

        # Verify all three phases were reported
        assert len(phase_calls) == 3
        assert result["status"] == "complete"

        # Verify phase sequence and info structure
        assert phase_calls[0]["phase"] == "loading_data"
        assert phase_calls[0]["dataset"] == "taiwan"
        assert phase_calls[0]["n_samples"] == 0  # Not yet loaded

        assert phase_calls[1]["phase"] == "feature_engineering"
        assert phase_calls[1]["dataset"] == "taiwan"
        assert phase_calls[1]["n_samples"] > 0  # Now loaded

        assert phase_calls[2]["phase"] == "optimizing"
        assert phase_calls[2]["dataset"] == "taiwan"
        assert phase_calls[2]["n_features"] > 0

    def test_run_optimization_with_loading_progress_callback(self, tmp_path: Path) -> None:
        """run_cleargbm_optimization calls loading progress callback during data loading."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 1,
                "random_state": 42,
            }
        )

        loading_calls: list[ClearGBMLoadingProgressInfo] = []

        def loading_progress_callback(info: ClearGBMLoadingProgressInfo) -> None:
            loading_calls.append(info)

        result = run_cleargbm_optimization(
            config_json, external_dir, output_dir, None, None, loading_progress_callback
        )

        # Verify loading progress was reported (at least one call for reading phase)
        assert result["status"] == "complete"
        # Use explicit count to verify callback was called
        progress_count = len(loading_calls)
        assert progress_count == 1 or progress_count > 1

        # Verify loading progress info structure
        first_info = loading_calls[0]
        assert first_info["dataset"] == "taiwan"
        assert first_info["phase"] in ("reading", "parsing", "encoding")
        assert 0.0 <= first_info["percent_complete"] <= 100.0
        assert first_info["rows_processed"] >= 0
        assert first_info["rows_total"] >= 0
        # Verify message is a non-empty string
        assert first_info["message"] != ""

    def test_run_optimization_with_feature_preset_none(self, tmp_path: Path) -> None:
        """run_cleargbm_optimization applies 'none' feature preset correctly."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        config = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 1,
                "feature_preset": "none",
                "random_state": 42,
            }
        )
        result = run_cleargbm_optimization(config, external_dir, output_dir)

        assert result["feature_preset"] == "none"
        assert result["n_features"] > 0
        assert result["status"] == "complete"

    def test_run_optimization_with_feature_preset_log_only(self, tmp_path: Path) -> None:
        """run_cleargbm_optimization applies 'log_only' feature preset correctly."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        config = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 1,
                "feature_preset": "log_only",
                "random_state": 42,
            }
        )
        result = run_cleargbm_optimization(config, external_dir, output_dir)

        assert result["feature_preset"] == "log_only"
        # log_only adds log-transformed features, so n_features should be > base
        assert result["n_features"] > 0
        assert result["status"] == "complete"


class TestProcessClearGBMOptimizeJob:
    """Integration tests for process_cleargbm_optimize_job entry point."""

    def test_process_job_loads_settings_and_runs(self, tmp_path: Path) -> None:
        """process_cleargbm_optimize_job loads settings from env and runs optimization."""
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
                    "n_trials": 1,
                    "random_state": 42,
                }
            )

            result = process_cleargbm_optimize_job(config_json)

            assert result["status"] == "complete"
            assert result["dataset"] == "taiwan"
            assert result["n_trials_complete"] == 1

            # Verify recommended config is included
            recommended = narrow_json_to_dict(result["recommended_config"])
            assert require_int(recommended, "max_depth") > 0
            assert require_int(recommended, "n_estimators") > 0
        finally:
            config_hooks.get_env = orig_get_env
