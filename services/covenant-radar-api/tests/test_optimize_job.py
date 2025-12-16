"""Tests for hyperparameter optimization job."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    narrow_json_to_dict,
    require_int,
    require_str,
)

from covenant_radar_api.worker.optimize_job import (
    _cuda_available,
    _get_search_space,
    _get_xgb_classifier,
    _load_dataset,
    _optional_int,
    _parse_device,
    _parse_feature_preset,
    _parse_optimize_config,
    _parse_space_profile,
    process_optimize_job,
    run_optimization,
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


class TestParseDevice:
    """Tests for _parse_device function."""

    def test_parse_device_defaults_to_auto(self) -> None:
        """None input returns 'auto'."""
        assert _parse_device(None) == "auto"

    def test_parse_device_accepts_cpu(self) -> None:
        """'cpu' is accepted."""
        assert _parse_device("cpu") == "cpu"

    def test_parse_device_accepts_cuda(self) -> None:
        """'cuda' is accepted."""
        assert _parse_device("cuda") == "cuda"

    def test_parse_device_accepts_auto(self) -> None:
        """'auto' is accepted."""
        assert _parse_device("auto") == "auto"

    def test_parse_device_rejects_invalid_string(self) -> None:
        """Invalid device string raises ValueError."""
        with pytest.raises(ValueError, match="device must be one of"):
            _parse_device("tpu")

    def test_parse_device_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="device must be a string"):
            _parse_device(123)


class TestParseSpaceProfile:
    """Tests for _parse_space_profile function."""

    def test_parse_space_profile_defaults_to_default(self) -> None:
        """None input returns 'default'."""
        assert _parse_space_profile(None) == "default"

    def test_parse_space_profile_accepts_default(self) -> None:
        """'default' is accepted."""
        assert _parse_space_profile("default") == "default"

    def test_parse_space_profile_accepts_categorical(self) -> None:
        """'categorical' is accepted."""
        assert _parse_space_profile("categorical") == "categorical"

    def test_parse_space_profile_rejects_invalid_string(self) -> None:
        """Invalid space_profile string raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="space_profile must be one of"):
            _parse_space_profile("invalid")

    def test_parse_space_profile_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="space_profile must be a string"):
            _parse_space_profile(123)


class TestParseFeaturePreset:
    """Tests for _parse_feature_preset function."""

    def test_parse_feature_preset_defaults_to_none(self) -> None:
        """None input returns 'none'."""
        assert _parse_feature_preset(None) == "none"

    def test_parse_feature_preset_accepts_none(self) -> None:
        """'none' is accepted."""
        assert _parse_feature_preset("none") == "none"

    def test_parse_feature_preset_accepts_log_only(self) -> None:
        """'log_only' is accepted."""
        assert _parse_feature_preset("log_only") == "log_only"

    def test_parse_feature_preset_accepts_ratios_only(self) -> None:
        """'ratios_only' is accepted."""
        assert _parse_feature_preset("ratios_only") == "ratios_only"

    def test_parse_feature_preset_accepts_full(self) -> None:
        """'full' is accepted."""
        assert _parse_feature_preset("full") == "full"

    def test_parse_feature_preset_rejects_invalid_string(self) -> None:
        """Invalid feature_preset string raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="feature_preset must be one of"):
            _parse_feature_preset("invalid")

    def test_parse_feature_preset_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="feature_preset must be a string"):
            _parse_feature_preset(123)


class TestOptionalInt:
    """Tests for _optional_int function."""

    def test_optional_int_returns_default_on_missing(self) -> None:
        """_optional_int returns default when key is missing."""
        assert _optional_int({}, "missing", 10) == 10

    def test_optional_int_returns_value_when_present(self) -> None:
        """_optional_int returns value when present."""
        assert _optional_int({"val": 20}, "val", 10) == 20

    def test_optional_int_converts_float_to_int(self) -> None:
        """_optional_int converts float to int."""
        assert _optional_int({"val": 15.5}, "val", 0) == 15

    def test_optional_int_raises_on_invalid_type(self) -> None:
        """_optional_int raises JSONTypeError on invalid type."""
        with pytest.raises(JSONTypeError, match="must be a number"):
            _optional_int({"val": "string"}, "val", 0)


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

    def test_get_search_space_categorical(self) -> None:
        """Categorical profile returns a valid search space."""
        space = _get_search_space("categorical")
        lr_spec = space["learning_rate"]
        assert lr_spec["param_type"] == "categorical_float"


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

    def test_parse_config_valid_us(self) -> None:
        """Parse valid config for US dataset."""
        config_json = dump_json_str(
            {
                "dataset": "us",
                "n_trials": 100,
                "timeout_seconds": 3600,
                "device": "cuda",
                "space_profile": "categorical",
                "random_state": 123,
            }
        )
        result = _parse_optimize_config(config_json)

        assert result["dataset"] == "us"
        assert result["n_trials"] == 100
        assert result["timeout_seconds"] == 3600
        assert result["device"] == "cuda"
        assert result["space_profile"] == "categorical"
        assert result["random_state"] == 123

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


class TestLoadDataset:
    """Tests for _load_dataset function."""

    def test_load_taiwan_dataset(self, tmp_path: Path) -> None:
        """_load_dataset loads Taiwan data successfully."""
        _, n_rows, feature_names = _copy_real_taiwan(tmp_path)
        dataset = _load_dataset("taiwan", tmp_path)

        assert dataset["n_samples"] == n_rows
        assert dataset["n_features"] == len(feature_names)

    def test_load_us_dataset(self, tmp_path: Path) -> None:
        """_load_dataset loads US data successfully."""
        _, n_rows_us, feature_names_us = _copy_real_us(tmp_path)
        dataset = _load_dataset("us", tmp_path)

        assert dataset["n_samples"] == n_rows_us
        assert dataset["n_features"] == len(feature_names_us)

    def test_load_polish_dataset(self, tmp_path: Path) -> None:
        """_load_dataset loads Polish data successfully."""
        _, n_rows_pl, feature_names_pl = _copy_real_polish(tmp_path)
        dataset = _load_dataset("polish", tmp_path)

        assert dataset["n_samples"] == n_rows_pl
        assert dataset["n_features"] == len(feature_names_pl)

    def test_load_dataset_missing_taiwan(self, tmp_path: Path) -> None:
        """_load_dataset raises FileNotFoundError for missing Taiwan data."""
        with pytest.raises(FileNotFoundError, match="Taiwan dataset not found"):
            _load_dataset("taiwan", tmp_path)

    def test_load_dataset_missing_us(self, tmp_path: Path) -> None:
        """_load_dataset raises FileNotFoundError for missing US data."""
        with pytest.raises(FileNotFoundError, match="US dataset not found"):
            _load_dataset("us", tmp_path)

    def test_load_dataset_missing_polish(self, tmp_path: Path) -> None:
        """_load_dataset raises FileNotFoundError for missing Polish data."""
        with pytest.raises(FileNotFoundError, match="Polish dataset not found"):
            _load_dataset("polish", tmp_path)


class TestGetXgbClassifier:
    """Tests for _get_xgb_classifier function."""

    def test_get_xgb_classifier_returns_class(self) -> None:
        """_get_xgb_classifier returns a usable class."""
        cls = _get_xgb_classifier()
        # Verify it's callable and has the expected name
        assert callable(cls)
        assert cls.__name__ == "XGBClassifier"


class TestCudaAvailable:
    """Tests for _cuda_available function."""

    def test_cuda_available_returns_true_or_false(self) -> None:
        """_cuda_available returns True or False."""
        result = _cuda_available()
        # Use value comparison instead of isinstance
        assert result is True or result is False


class TestXGBoostObjective:
    """Tests for _XGBoostObjective class."""

    def test_objective_initialization(self, tmp_path: Path) -> None:
        """_XGBoostObjective can be initialized with data."""
        from covenant_radar_api.worker.optimize_job import _XGBoostObjective

        # Create minimal test data
        rng = np.random.default_rng(42)
        x_features: NDArray[np.float64] = rng.random((100, 10)).astype(np.float64)
        y_labels: NDArray[np.int64] = rng.integers(0, 2, 100).astype(np.int64)
        feature_names = [f"f{i}" for i in range(10)]

        objective = _XGBoostObjective(x_features, y_labels, feature_names, "cpu", "none")
        assert objective._device == "cpu"

    def test_objective_with_feature_engineering(self, tmp_path: Path) -> None:
        """_XGBoostObjective applies feature engineering when preset is not 'none'."""
        from covenant_radar_api.worker.optimize_job import _XGBoostObjective

        # Create minimal test data
        rng = np.random.default_rng(42)
        x_features: NDArray[np.float64] = rng.random((100, 10)).astype(np.float64)
        y_labels: NDArray[np.int64] = rng.integers(0, 2, 100).astype(np.int64)
        feature_names = [f"f{i}" for i in range(10)]

        # Use log_only preset to trigger feature engineering
        objective = _XGBoostObjective(x_features, y_labels, feature_names, "cpu", "log_only")
        assert objective._device == "cpu"
        # With log_only, features should be doubled (original + log transforms)
        n_engineered: int = int(objective._splits.x_train.shape[1])
        assert n_engineered == 20  # 10 original + 10 log

    def test_objective_call_returns_float(self, tmp_path: Path) -> None:
        """_XGBoostObjective __call__ returns a float AUC value."""
        from covenant_radar_api.worker.optimize_job import _XGBoostObjective

        # Create minimal test data with some class balance
        rng = np.random.default_rng(42)
        x_features: NDArray[np.float64] = rng.random((100, 10)).astype(np.float64)
        y_labels: NDArray[np.int64] = np.zeros(100, dtype=np.int64)
        y_labels[:30] = 1  # 30% positive
        feature_names = [f"f{i}" for i in range(10)]

        objective = _XGBoostObjective(x_features, y_labels, feature_names, "cpu", "none")

        # Call with hyperparameters
        result = objective(
            x_features=x_features,
            y_labels=y_labels,
            feature_names=["f" + str(i) for i in range(10)],
            max_depth=3,
            n_estimators=10,
            learning_rate=0.1,
            reg_alpha=0.0,
            reg_lambda=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        # AUC should be between 0 and 1 (value assertion implies float)
        assert 0.0 <= result <= 1.0


class TestCreateXgboostObjective:
    """Tests for _create_xgboost_objective function."""

    def test_create_objective_returns_protocol(self) -> None:
        """_create_xgboost_objective returns an XGBoostObjectiveProtocol instance."""
        from covenant_radar_api.worker.optimize_job import _create_xgboost_objective

        # Create minimal test data
        rng = np.random.default_rng(42)
        x_features: NDArray[np.float64] = rng.random((100, 10)).astype(np.float64)
        y_labels: NDArray[np.int64] = np.zeros(100, dtype=np.int64)
        y_labels[:30] = 1
        feature_names = [f"f{i}" for i in range(10)]

        objective = _create_xgboost_objective(x_features, y_labels, feature_names, "cpu", "none")

        # Verify it's callable
        assert callable(objective)


class TestBuildOptimizationConfig:
    """Tests for _build_optimization_config function."""

    def test_build_config_with_timeout(self) -> None:
        """_build_optimization_config creates config with timeout."""
        from covenant_radar_api.worker.optimize_job import _build_optimization_config

        config = _build_optimization_config(
            n_trials=50,
            timeout_seconds=3600,
            random_state=42,
        )

        assert config["n_trials"] == 50
        assert config["timeout_seconds"] == 3600
        assert config["random_state"] == 42

    def test_build_config_without_timeout(self) -> None:
        """_build_optimization_config creates config without timeout."""
        from covenant_radar_api.worker.optimize_job import _build_optimization_config

        config = _build_optimization_config(
            n_trials=25,
            timeout_seconds=None,
            random_state=123,
        )

        assert config["n_trials"] == 25
        assert config["timeout_seconds"] is None
        assert config["random_state"] == 123


class TestGenerateTrainConfig:
    """Tests for _generate_train_config function."""

    def test_generate_train_config_from_summary(self) -> None:
        """_generate_train_config creates correct TrainConfig from summary."""
        from covenant_ml.optimizer import OptimizationSummary

        from covenant_radar_api.worker.optimize_job import _generate_train_config

        summary: OptimizationSummary = {
            "n_trials_total": 50,
            "n_trials_complete": 50,
            "n_trials_pruned": 0,
            "n_trials_failed": 0,
            "best_trial_number": 25,
            "best_value": 0.85,
            "best_max_depth": 5,
            "best_n_estimators": 100,
            "best_learning_rate": 0.1,
            "best_reg_alpha": 0.01,
            "best_reg_lambda": 1.0,
            "best_subsample": 0.8,
            "best_colsample_bytree": 0.8,
            "total_duration_seconds": 120.5,
        }

        config = _generate_train_config(summary, "cuda")

        assert config["device"] == "cuda"
        assert config["learning_rate"] == 0.1
        assert config["max_depth"] == 5
        assert config["n_estimators"] == 100
        assert config["reg_alpha"] == 0.01
        assert config["reg_lambda"] == 1.0
        assert config["subsample"] == 0.8
        assert config["colsample_bytree"] == 0.8
        assert config["random_state"] == 42
        assert config["train_ratio"] == 0.7
        assert config["val_ratio"] == 0.15
        assert config["test_ratio"] == 0.15
        assert config["early_stopping_rounds"] == 20


class TestRunOptimization:
    """Integration tests for run_optimization with real data."""

    def test_run_optimization_taiwan(self, tmp_path: Path) -> None:
        """run_optimization completes successfully on Taiwan dataset."""
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
                "space_profile": "categorical",  # faster due to fewer search options
                "random_state": 42,
            }
        )

        result = run_optimization(config_json, external_dir, output_dir)

        # Verify result structure
        assert result["status"] == "complete"
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
        assert config["device"] == "cpu"
        assert config["max_depth"] == result["best_max_depth"]
        assert config["n_estimators"] == result["best_n_estimators"]
        assert config["learning_rate"] == result["best_learning_rate"]

        # Verify output files created
        assert (output_dir / "taiwan_optuna_result.json").exists()
        assert (output_dir / "taiwan_optimal_config.json").exists()

    def test_run_optimization_us(self, tmp_path: Path) -> None:
        """run_optimization completes successfully on US dataset."""
        external_dir = tmp_path / "external"
        _copy_real_us(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "us",
                "n_trials": 2,
                "device": "cpu",
                "space_profile": "categorical",
                "random_state": 42,
            }
        )

        result = run_optimization(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "us"
        assert result["n_samples"] > 0
        assert result["n_trials_complete"] == 2
        assert 0.0 <= result["best_val_auc"] <= 1.0

        # Verify output files created
        assert (output_dir / "us_optuna_result.json").exists()
        assert (output_dir / "us_optimal_config.json").exists()

    def test_run_optimization_polish(self, tmp_path: Path) -> None:
        """run_optimization completes successfully on Polish dataset."""
        external_dir = tmp_path / "external"
        _copy_real_polish(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "polish",
                "n_trials": 2,
                "device": "cpu",
                "space_profile": "categorical",
                "random_state": 42,
            }
        )

        result = run_optimization(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "polish"
        assert result["n_samples"] > 0
        assert result["n_trials_complete"] == 2
        assert 0.0 <= result["best_val_auc"] <= 1.0

        # Verify output files created
        assert (output_dir / "polish_optuna_result.json").exists()
        assert (output_dir / "polish_optimal_config.json").exists()

    def test_run_optimization_with_timeout(self, tmp_path: Path) -> None:
        """run_optimization respects timeout_seconds parameter."""
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
                "space_profile": "categorical",
                "random_state": 42,
            }
        )

        result = run_optimization(config_json, external_dir, output_dir)

        # Should complete but with fewer trials than requested
        assert result["status"] == "complete"
        # May complete some trials before timeout
        assert result["n_trials_complete"] >= 1
        assert result["n_trials_complete"] <= 100

    def test_run_optimization_default_space(self, tmp_path: Path) -> None:
        """run_optimization works with default space profile."""
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
            }
        )

        result = run_optimization(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["n_trials_complete"] == 2
        assert 0.0 <= result["best_val_auc"] <= 1.0

    def test_run_optimization_with_progress_callback(self, tmp_path: Path) -> None:
        """run_optimization calls progress callback with trial info."""
        from covenant_radar_api.worker.optimize_job import TrialProgressInfo

        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)
        output_dir = tmp_path / "optuna_output"

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "n_trials": 2,
                "device": "cpu",
                "space_profile": "categorical",
                "random_state": 42,
            }
        )

        callback_calls: list[TrialProgressInfo] = []

        def progress_callback(info: TrialProgressInfo) -> None:
            callback_calls.append(info)

        result = run_optimization(config_json, external_dir, output_dir, progress_callback)

        # Verify callback was called for each trial
        assert len(callback_calls) == 2
        assert result["status"] == "complete"

        # Verify callback info structure - use direct value checks instead of key existence
        for info in callback_calls:
            assert info["trial_number"] >= 0
            assert info["n_trials_total"] == 2
            assert 0.0 <= info["current_auc"] <= 1.0
            assert 0.0 <= info["best_auc"] <= 1.0
            assert info["best_trial"] >= 0
            assert info["is_best"] in (True, False)


class TestProcessOptimizeJob:
    """Integration tests for process_optimize_job entry point."""

    def test_process_job_loads_settings_and_runs(self, tmp_path: Path) -> None:
        """process_optimize_job loads settings from env and runs optimization."""
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
                    "space_profile": "categorical",
                    "random_state": 42,
                }
            )

            result = process_optimize_job(config_json)

            assert result["status"] == "complete"
            assert result["dataset"] == "taiwan"
            assert result["n_trials_complete"] == 2

            # Verify recommended config is included
            recommended = narrow_json_to_dict(result["recommended_config"])
            assert require_str(recommended, "device") == "cpu"
            assert require_int(recommended, "max_depth") > 0
            assert require_int(recommended, "n_estimators") > 0
        finally:
            config_hooks.get_env = orig_get_env
