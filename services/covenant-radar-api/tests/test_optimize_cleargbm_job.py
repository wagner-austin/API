"""Tests for ClearGBM hyperparameter optimization job."""

from __future__ import annotations

import gc
from collections.abc import Generator
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
    _get_search_space,
    _parse_optimize_config,
    _parse_space_profile,
    process_cleargbm_optimize_job,
    run_cleargbm_optimization,
)

# Skip entire module - ClearGBM optimization tests are too slow for CI
pytestmark = pytest.mark.skip(reason="ClearGBM optimization tests disabled - too slow")


@pytest.fixture(autouse=True)
def _force_gc_around_test() -> Generator[None, None, None]:
    """Force garbage collection before and after each test.

    ClearGBM optimization tests train models which consume memory.
    When pytest-xdist runs multiple tests on the same worker, memory can accumulate
    and cause worker crashes. This fixture ensures cleanup before and after each test.
    """
    gc.collect()
    yield
    gc.collect()


def _copy_real_taiwan(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full Taiwan dataset into external_root and return (path, n_rows, feature_names).

    Args:
        external_root: Root directory for external datasets.

    Returns:
        Tuple of (dataset path, number of rows, list of feature names).

    Raises:
        FileNotFoundError: If Taiwan dataset not found in repository.
    """
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
        """Default profile returns a valid ClearGBM search space."""
        space = _get_search_space("default")
        # Verify ClearGBM-specific params exist
        lr_spec = space["learning_rate"]
        assert lr_spec["param_type"] in ("float", "categorical_float")
        max_depth_spec = space["max_depth"]
        assert max_depth_spec["param_type"] in ("int", "categorical_int")
        n_estimators_spec = space["n_estimators"]
        assert n_estimators_spec["param_type"] in ("int", "categorical_int")


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
        assert result["early_stopping_rounds"] == 10

    def test_parse_config_valid_with_all_options(self) -> None:
        """Parse valid config with all options specified."""
        config_json = dump_json_str(
            {
                "dataset": "us",
                "n_trials": 100,
                "timeout_seconds": 3600,
                "space_profile": "default",
                "random_state": 123,
                "early_stopping_rounds": 5,
                "feature_preset": "full",
            }
        )
        result = _parse_optimize_config(config_json)

        assert result["dataset"] == "us"
        assert result["n_trials"] == 100
        assert result["timeout_seconds"] == 3600
        assert result["space_profile"] == "default"
        assert result["random_state"] == 123
        assert result["early_stopping_rounds"] == 5
        assert result["feature_preset"] == "full"

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


@pytest.mark.xdist_group(name="cleargbm_heavy")
@pytest.mark.timeout(300)
class TestRunClearGBMOptimization:
    """Integration tests for run_cleargbm_optimization with real data.

    Uses xdist_group marker to ensure all ClearGBM tests run on the same worker.
    Timeout extended to 5 minutes per test for ClearGBM optimization.
    """

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
                "n_trials": 2,
                "space_profile": "default",
                "random_state": 42,
            }
        )

        result = run_cleargbm_optimization(config_json, external_dir, output_dir)

        # Verify result structure
        assert result["status"] == "complete"
        assert result["backend"] == "cleargbm"
        assert result["dataset"] == "taiwan"
        assert result["n_samples"] > 0
        assert result["n_features"] > 0
        assert result["n_trials_complete"] == 2
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
        # ClearGBM-specific
        assert config["min_samples_split"] > 0
        assert config["min_samples_leaf"] > 0
        assert config["max_bins"] > 0

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
                "space_profile": "default",
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
                "n_trials": 2,
                "space_profile": "default",
                "random_state": 42,
            }
        )

        callback_calls: list[ClearGBMTrialProgressInfo] = []

        def progress_callback(info: ClearGBMTrialProgressInfo) -> None:
            callback_calls.append(info)

        result = run_cleargbm_optimization(config_json, external_dir, output_dir, progress_callback)

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
            assert info["best_max_depth"] >= 0

    def test_run_optimization_with_phase_callback(self, tmp_path: Path) -> None:
        """run_cleargbm_optimization calls phase callback at each phase."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        phases_received: list[ClearGBMPhaseInfo] = []

        def phase_callback(info: ClearGBMPhaseInfo) -> None:
            phases_received.append(info)

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 2,
                "feature_preset": "none",
            }
        )

        run_cleargbm_optimization(
            config_json,
            external_dir,
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
        """run_cleargbm_optimization calls loading progress callback during data loading."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        loading_calls: list[ClearGBMLoadingProgressInfo] = []

        def loading_progress_callback(info: ClearGBMLoadingProgressInfo) -> None:
            loading_calls.append(info)

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 2,
                "feature_preset": "none",
            }
        )

        result = run_cleargbm_optimization(
            config_json,
            external_dir,
            output_dir,
            progress_callback=None,
            phase_callback=None,
            loading_progress_callback=loading_progress_callback,
        )

        # Verify loading progress was reported
        progress_count = len(loading_calls)
        assert progress_count >= 1
        assert result["status"] == "complete"

        # Verify loading progress info structure
        first_info = loading_calls[0]
        assert first_info["dataset"] == "taiwan"
        assert first_info["phase"] in ("reading", "parsing", "encoding")
        assert 0.0 <= first_info["percent_complete"] <= 100.0
        assert first_info["rows_processed"] >= 0
        assert first_info["rows_total"] >= 0
        assert first_info["message"] != ""

    @pytest.mark.timeout(600)
    def test_run_optimization_progress_includes_non_best_trial(self, tmp_path: Path) -> None:
        """run_cleargbm_optimization includes trials where is_best is False."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        # Run 3 trials - at least one must NOT be best since best can only improve
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 3,
                "space_profile": "default",
                "random_state": 123,  # Different seed from other tests
            }
        )

        callback_calls: list[ClearGBMTrialProgressInfo] = []

        def progress_callback(info: ClearGBMTrialProgressInfo) -> None:
            callback_calls.append(info)

        result = run_cleargbm_optimization(config_json, external_dir, output_dir, progress_callback)

        assert result["status"] == "complete"
        assert len(callback_calls) == 3

        # With 3 trials, at least one must have is_best=False
        best_count = sum(1 for info in callback_calls if info["is_best"])
        non_best_count = sum(1 for info in callback_calls if not info["is_best"])

        # At least one trial must not be best (covers the is_best=False branch)
        assert non_best_count >= 1, f"Expected at least 1 non-best trial, got {non_best_count}"
        # And at least one must be best (the first trial is always best)
        assert best_count >= 1, f"Expected at least 1 best trial, got {best_count}"


@pytest.mark.xdist_group(name="cleargbm_heavy")
@pytest.mark.timeout(300)
class TestProcessClearGBMOptimizeJob:
    """Integration tests for process_cleargbm_optimize_job entry point.

    Uses xdist_group marker to ensure this runs on the same worker as other ClearGBM tests.
    """

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
                    "n_trials": 2,
                    "space_profile": "default",
                    "random_state": 42,
                }
            )

            result = process_cleargbm_optimize_job(config_json)

            assert result["status"] == "complete"
            assert result["dataset"] == "taiwan"
            assert result["n_trials_complete"] == 2

            # Verify recommended config is included with ClearGBM-specific fields
            recommended = narrow_json_to_dict(result["recommended_config"])
            assert require_int(recommended, "max_depth") > 0
            assert require_int(recommended, "n_estimators") > 0
            assert require_int(recommended, "min_samples_split") > 0
            assert require_int(recommended, "min_samples_leaf") > 0
            assert require_int(recommended, "max_bins") > 0
        finally:
            config_hooks.get_env = orig_get_env
