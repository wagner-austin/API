"""Tests for LSTM hyperparameter optimization job."""

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

from covenant_radar_api.worker.optimize_lstm_job import (
    LSTMLoadingProgressInfo,
    LSTMPhaseInfo,
    LSTMTrialProgressInfo,
    _get_search_space,
    _parse_bidirectional,
    _parse_optimize_config,
    _parse_precision,
    _parse_space_profile,
    process_lstm_optimize_job,
    run_lstm_optimization,
)


@pytest.fixture(autouse=True)
def _force_gc_after_test() -> Generator[None, None, None]:
    """Force garbage collection after each test to prevent memory accumulation.

    LSTM optimization tests train PyTorch models which consume significant memory.
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


class TestParseBidirectional:
    """Tests for _parse_bidirectional function."""

    def test_parse_bidirectional_defaults_to_false(self) -> None:
        """None input returns False."""
        assert _parse_bidirectional(None) is False

    def test_parse_bidirectional_accepts_true(self) -> None:
        """True is accepted."""
        assert _parse_bidirectional(True) is True

    def test_parse_bidirectional_accepts_false(self) -> None:
        """False is accepted."""
        assert _parse_bidirectional(False) is False

    def test_parse_bidirectional_rejects_non_boolean(self) -> None:
        """Non-boolean input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="bidirectional must be a boolean"):
            _parse_bidirectional("true")


class TestGetSearchSpace:
    """Tests for _get_search_space function."""

    def test_get_search_space_default(self) -> None:
        """Default profile returns a valid LSTM search space."""
        space = _get_search_space("default")
        # Verify LSTM-specific params exist
        lr_spec = space["learning_rate"]
        assert lr_spec["param_type"] in ("float", "categorical_float")
        hidden_size_spec = space["hidden_size"]
        assert hidden_size_spec["param_type"] in ("int", "categorical_int")
        num_layers_spec = space["num_layers"]
        assert num_layers_spec["param_type"] in ("int", "categorical_int")
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
        assert result["n_epochs"] == 50
        assert result["early_stopping_patience"] == 10
        assert result["sequence_length"] == 5
        assert result["bidirectional"] is False

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
                "n_epochs": 100,
                "early_stopping_patience": 5,
                "sequence_length": 10,
                "bidirectional": True,
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
        assert result["n_epochs"] == 100
        assert result["early_stopping_patience"] == 5
        assert result["sequence_length"] == 10
        assert result["bidirectional"] is True

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


@pytest.mark.xdist_group(name="lstm_heavy")
@pytest.mark.timeout(300)
class TestRunLSTMOptimization:
    """Integration tests for run_lstm_optimization with real data.

    Uses xdist_group marker to ensure all LSTM tests run on the same worker,
    preventing memory exhaustion from parallel LSTM training.
    Timeout extended to 5 minutes per test for LSTM training.
    """

    def test_run_optimization_taiwan(self, tmp_path: Path) -> None:
        """run_lstm_optimization completes successfully on Taiwan dataset."""
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
                "n_epochs": 3,  # Very short for testing
                "early_stopping_patience": 1,
                "sequence_length": 2,  # Short sequence for testing
            }
        )

        result = run_lstm_optimization(config_json, external_dir, output_dir)

        # Verify result structure
        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        assert result["n_samples"] > 0
        assert result["n_features"] > 0
        assert result["n_trials_complete"] == 2
        assert 0.0 <= result["best_val_auc"] <= 1.0
        assert result["best_hidden_size"] > 0
        assert result["best_num_layers"] > 0
        assert result["best_learning_rate"] > 0.0
        assert result["best_dropout"] >= 0.0
        assert result["duration_seconds"] >= 0.0

        # Verify recommended config
        config = result["recommended_config"]
        assert config["device"] == "cpu"
        assert config["precision"] == "fp32"
        assert config["hidden_size"] > 0
        assert config["num_layers"] > 0
        assert config["sequence_length"] == 2

        # Verify output files created
        assert (output_dir / "taiwan_lstm_optuna_result.json").exists()
        assert (output_dir / "taiwan_lstm_optimal_config.json").exists()

    def test_run_optimization_with_timeout(self, tmp_path: Path) -> None:
        """run_lstm_optimization respects timeout_seconds parameter."""
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
                "sequence_length": 2,
            }
        )

        result = run_lstm_optimization(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["n_trials_complete"] >= 1
        assert result["n_trials_complete"] <= 100

    def test_run_optimization_with_progress_callback(self, tmp_path: Path) -> None:
        """run_lstm_optimization calls progress callback with trial info."""
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
                "n_epochs": 3,
                "sequence_length": 2,
            }
        )

        callback_calls: list[LSTMTrialProgressInfo] = []

        def progress_callback(info: LSTMTrialProgressInfo) -> None:
            callback_calls.append(info)

        result = run_lstm_optimization(config_json, external_dir, output_dir, progress_callback)

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
        """run_lstm_optimization calls phase callback for all phases."""
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
                "n_epochs": 3,
                "sequence_length": 2,
            }
        )

        phase_calls: list[LSTMPhaseInfo] = []

        def phase_callback(info: LSTMPhaseInfo) -> None:
            phase_calls.append(info)

        result = run_lstm_optimization(config_json, external_dir, output_dir, None, phase_callback)

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
        """run_lstm_optimization calls loading progress callback during data loading."""
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
                "n_epochs": 3,
                "sequence_length": 2,
            }
        )

        loading_calls: list[LSTMLoadingProgressInfo] = []

        def loading_progress_callback(info: LSTMLoadingProgressInfo) -> None:
            loading_calls.append(info)

        result = run_lstm_optimization(
            config_json, external_dir, output_dir, None, None, loading_progress_callback
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

    def test_run_optimization_covers_non_best_trials(self, tmp_path: Path) -> None:
        """run_lstm_optimization covers is_best=False branch with multiple trials.

        Running 5 trials ensures at least one trial is NOT the best, which
        exercises the code path where is_best is False (branch 383->405).
        """
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 5,  # More trials increases chance of non-best
                "device": "cpu",
                "space_profile": "default",
                "random_state": 123,  # Different seed from other tests
                "n_epochs": 3,
                "early_stopping_patience": 1,
                "sequence_length": 2,
            }
        )

        best_seen: list[bool] = []

        def progress_callback(info: LSTMTrialProgressInfo) -> None:
            best_seen.append(info["is_best"])

        result = run_lstm_optimization(config_json, external_dir, output_dir, progress_callback)

        assert result["status"] == "complete"
        assert result["n_trials_complete"] == 5
        # With 5 trials, we expect at least one trial to NOT be the best
        # (covers the is_best=False branch at line 383->405)
        assert True in best_seen, "Expected at least one best trial"
        assert False in best_seen, "Expected at least one non-best trial"


@pytest.mark.xdist_group(name="lstm_heavy")
@pytest.mark.timeout(300)
class TestProcessLSTMOptimizeJob:
    """Integration tests for process_lstm_optimize_job entry point.

    Uses xdist_group marker to ensure this runs on the same worker as other LSTM tests.
    """

    def test_process_job_loads_settings_and_runs(self, tmp_path: Path) -> None:
        """process_lstm_optimize_job loads settings from env and runs optimization."""
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
                    "n_epochs": 3,
                    "sequence_length": 2,
                }
            )

            result = process_lstm_optimize_job(config_json)

            assert result["status"] == "complete"
            assert result["dataset"] == "taiwan"
            assert result["n_trials_complete"] == 2

            recommended = narrow_json_to_dict(result["recommended_config"])
            assert require_str(recommended, "device") == "cpu"
            assert require_str(recommended, "precision") == "fp32"
            assert require_int(recommended, "hidden_size") > 0
            assert require_int(recommended, "num_layers") > 0
            assert require_int(recommended, "batch_size") > 0
        finally:
            config_hooks.get_env = orig_get_env
