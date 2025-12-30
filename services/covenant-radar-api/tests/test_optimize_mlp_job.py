"""Tests for MLP hyperparameter optimization job."""

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
    require_str,
)

from covenant_radar_api.worker.optimize_mlp_job import (
    MLPLoadingProgressInfo,
    MLPPhaseInfo,
    MLPTrialProgressInfo,
    _get_search_space,
    _parse_optimize_config,
    _parse_optimizer,
    _parse_precision,
    _parse_space_profile,
    process_mlp_optimize_job,
    run_mlp_optimization,
)


@pytest.fixture(autouse=True)
def _force_gc_after_test() -> Generator[None, None, None]:
    """Force garbage collection after each test to prevent memory accumulation.

    MLP optimization tests train PyTorch models which consume significant memory.
    When pytest-xdist runs multiple tests on the same worker, memory can accumulate
    and cause worker crashes. This fixture ensures cleanup after each test.
    """
    yield
    gc.collect()


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
    feature_names = cols[1:]
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


class TestParsePrecision:
    """Tests for _parse_precision function."""

    def test_parse_precision_defaults_to_fp32(self) -> None:
        """None input returns 'fp32'."""
        assert _parse_precision(None) == "fp32"

    def test_parse_precision_accepts_fp32(self) -> None:
        """'fp32' is accepted."""
        assert _parse_precision("fp32") == "fp32"

    def test_parse_precision_accepts_fp16(self) -> None:
        """'fp16' is accepted."""
        assert _parse_precision("fp16") == "fp16"

    def test_parse_precision_accepts_bf16(self) -> None:
        """'bf16' is accepted."""
        assert _parse_precision("bf16") == "bf16"

    def test_parse_precision_accepts_auto(self) -> None:
        """'auto' is accepted."""
        assert _parse_precision("auto") == "auto"

    def test_parse_precision_rejects_invalid_string(self) -> None:
        """Invalid precision string raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="precision must be one of"):
            _parse_precision("invalid")

    def test_parse_precision_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="precision must be a string"):
            _parse_precision(123)


class TestParseOptimizer:
    """Tests for _parse_optimizer function."""

    def test_parse_optimizer_defaults_to_adamw(self) -> None:
        """None input returns 'adamw'."""
        assert _parse_optimizer(None) == "adamw"

    def test_parse_optimizer_accepts_adamw(self) -> None:
        """'adamw' is accepted."""
        assert _parse_optimizer("adamw") == "adamw"

    def test_parse_optimizer_accepts_adam(self) -> None:
        """'adam' is accepted."""
        assert _parse_optimizer("adam") == "adam"

    def test_parse_optimizer_accepts_sgd(self) -> None:
        """'sgd' is accepted."""
        assert _parse_optimizer("sgd") == "sgd"

    def test_parse_optimizer_rejects_invalid_string(self) -> None:
        """Invalid optimizer string raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="optimizer must be one of"):
            _parse_optimizer("invalid")

    def test_parse_optimizer_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="optimizer must be a string"):
            _parse_optimizer(123)


class TestGetSearchSpace:
    """Tests for _get_search_space function."""

    def test_get_search_space_default(self) -> None:
        """Default profile returns a valid MLP search space."""
        space = _get_search_space("default")
        # Verify MLP-specific params exist
        lr_spec = space["learning_rate"]
        assert lr_spec["param_type"] in ("float", "categorical_float")
        n_layers_spec = space["n_layers"]
        assert n_layers_spec["param_type"] in ("int", "categorical_int")
        hidden_size_spec = space["hidden_size"]
        assert hidden_size_spec["param_type"] in ("int", "categorical_int")
        dropout_spec = space["dropout"]
        assert dropout_spec["param_type"] in ("float", "categorical_float")


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
        assert result["precision"] == "fp32"
        assert result["optimizer"] == "adamw"
        assert result["n_epochs"] == 50
        assert result["early_stopping_patience"] == 10

    def test_parse_config_valid_with_all_options(self) -> None:
        """Parse valid config with all options specified."""
        config_json = dump_json_str(
            {
                "dataset": "us",
                "n_trials": 100,
                "timeout_seconds": 3600,
                "device": "cuda",
                "space_profile": "default",
                "random_state": 123,
                "precision": "fp16",
                "optimizer": "adam",
                "n_epochs": 100,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_optimize_config(config_json)

        assert result["dataset"] == "us"
        assert result["n_trials"] == 100
        assert result["timeout_seconds"] == 3600
        assert result["device"] == "cuda"
        assert result["space_profile"] == "default"
        assert result["random_state"] == 123
        assert result["precision"] == "fp16"
        assert result["optimizer"] == "adam"
        assert result["n_epochs"] == 100
        assert result["early_stopping_patience"] == 5

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


@pytest.mark.xdist_group(name="mlp_heavy")
@pytest.mark.timeout(300)
class TestRunMLPOptimization:
    """Integration tests for run_mlp_optimization with real data.

    Uses xdist_group marker to ensure all MLP tests run on the same worker,
    preventing memory exhaustion from parallel PyTorch training.
    Timeout extended to 5 minutes per test for MLP training.
    """

    def test_run_optimization_taiwan(self, tmp_path: Path) -> None:
        """run_mlp_optimization completes successfully on Taiwan dataset."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        # Use minimal trials and epochs for fast test
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 2,
                "device": "cpu",
                "space_profile": "default",
                "random_state": 42,
                "n_epochs": 5,  # Very short for testing
                "early_stopping_patience": 2,
            }
        )

        result = run_mlp_optimization(config_json, external_dir, output_dir)

        # Verify result structure
        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        assert result["n_samples"] > 0
        assert result["n_features"] > 0
        assert result["n_trials_complete"] == 2
        assert 0.0 <= result["best_val_auc"] <= 1.0
        assert result["best_n_layers"] > 0
        assert result["best_hidden_size"] > 0
        assert result["best_learning_rate"] > 0.0
        assert result["best_dropout"] >= 0.0
        assert result["duration_seconds"] >= 0.0

        # Verify recommended config
        config = result["recommended_config"]
        assert config["device"] == "cpu"
        assert config["precision"] == "fp32"
        assert config["optimizer"] == "adamw"
        # hidden_sizes tuple length matches best_n_layers
        assert len(config["hidden_sizes"]) == result["best_n_layers"]

        # Verify output files created
        assert (output_dir / "taiwan_mlp_optuna_result.json").exists()
        assert (output_dir / "taiwan_mlp_optimal_config.json").exists()

    def test_run_optimization_with_timeout(self, tmp_path: Path) -> None:
        """run_mlp_optimization respects timeout_seconds parameter."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        # Use high n_trials but short timeout
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 100,
                "timeout_seconds": 5,
                "device": "cpu",
                "space_profile": "default",
                "random_state": 42,
                "n_epochs": 2,
            }
        )

        result = run_mlp_optimization(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["n_trials_complete"] >= 1
        assert result["n_trials_complete"] <= 100

    def test_run_optimization_with_progress_callback(self, tmp_path: Path) -> None:
        """run_mlp_optimization calls progress callback with trial info."""
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
                "n_epochs": 5,
            }
        )

        callback_calls: list[MLPTrialProgressInfo] = []

        def progress_callback(info: MLPTrialProgressInfo) -> None:
            callback_calls.append(info)

        result = run_mlp_optimization(config_json, external_dir, output_dir, progress_callback)

        assert len(callback_calls) == 2
        assert result["status"] == "complete"

        for info in callback_calls:
            assert info["trial_number"] >= 0
            assert info["n_trials_total"] == 2
            assert 0.0 <= info["current_auc"] <= 1.0
            assert 0.0 <= info["best_auc"] <= 1.0
            assert info["best_trial"] >= 0
            assert info["is_best"] in (True, False)

    def test_run_optimization_with_phase_callback(self, tmp_path: Path) -> None:
        """run_mlp_optimization calls phase callback for all phases."""
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
                "n_epochs": 5,
            }
        )

        phase_calls: list[MLPPhaseInfo] = []

        def phase_callback(info: MLPPhaseInfo) -> None:
            phase_calls.append(info)

        result = run_mlp_optimization(config_json, external_dir, output_dir, None, phase_callback)

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
        """run_mlp_optimization calls loading progress callback during data loading."""
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
                "n_epochs": 5,
            }
        )

        loading_calls: list[MLPLoadingProgressInfo] = []

        def loading_progress_callback(info: MLPLoadingProgressInfo) -> None:
            loading_calls.append(info)

        result = run_mlp_optimization(
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

    def test_run_optimization_progress_includes_non_best_trial(self, tmp_path: Path) -> None:
        """run_mlp_optimization includes trials where is_best is False."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        # Run 3 trials - at least one must NOT be best since best can only improve
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 3,
                "device": "cpu",
                "space_profile": "default",
                "random_state": 42,
                "n_epochs": 5,
            }
        )

        callback_calls: list[MLPTrialProgressInfo] = []

        def progress_callback(info: MLPTrialProgressInfo) -> None:
            callback_calls.append(info)

        result = run_mlp_optimization(config_json, external_dir, output_dir, progress_callback)

        assert result["status"] == "complete"
        assert len(callback_calls) == 3

        # With 3 trials, at least one must have is_best=False
        # (first trial is always best, subsequent trials may or may not improve)
        best_count = sum(1 for info in callback_calls if info["is_best"])
        non_best_count = sum(1 for info in callback_calls if not info["is_best"])

        # At least one trial must not be best (covers the is_best=False branch)
        assert non_best_count >= 1, f"Expected at least 1 non-best trial, got {non_best_count}"
        # And at least one must be best (the first trial is always best)
        assert best_count >= 1, f"Expected at least 1 best trial, got {best_count}"


@pytest.mark.xdist_group(name="mlp_heavy")
@pytest.mark.timeout(300)
class TestProcessMLPOptimizeJob:
    """Integration tests for process_mlp_optimize_job entry point.

    Uses xdist_group marker to ensure this runs on the same worker as other MLP tests.
    """

    def test_process_job_loads_settings_and_runs(self, tmp_path: Path) -> None:
        """process_mlp_optimize_job loads settings from env and runs optimization."""
        from platform_core.config import _test_hooks as config_hooks
        from platform_core.testing import FakeEnv

        data_root = tmp_path / "data"
        external_dir = data_root / "external"
        models_dir = tmp_path / "models"

        _copy_real_taiwan(external_dir)

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
                    "n_epochs": 5,
                }
            )

            result = process_mlp_optimize_job(config_json)

            assert result["status"] == "complete"
            assert result["dataset"] == "taiwan"
            assert result["n_trials_complete"] == 2

            recommended = narrow_json_to_dict(result["recommended_config"])
            assert require_str(recommended, "device") == "cpu"
            assert require_str(recommended, "precision") == "fp32"
            assert require_str(recommended, "optimizer") == "adamw"
            assert require_int(recommended, "batch_size") > 0
        finally:
            config_hooks.get_env = orig_get_env
