"""Tests for external CSV training job with automatic feature selection."""

from __future__ import annotations

from pathlib import Path

import pytest
from covenant_ml.types import MLPConfig, TrainConfig
from platform_core.json_utils import (
    InvalidJsonError,
    JSONTypeError,
    dump_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)

from covenant_radar_api.worker.train_external_job import (
    _load_dataset,
    _metrics_to_json,
    _optional_float,
    _optional_int,
    _parse_device,
    _parse_external_train_config,
    process_external_train_job,
    run_external_training,
)


def _write_taiwan_dataset(base_dir: Path) -> Path:
    """Write a minimal Taiwan-style CSV dataset for testing."""
    taiwan_dir = base_dir / "taiwan_data"
    taiwan_dir.mkdir(parents=True, exist_ok=True)
    path = taiwan_dir / "data.csv"
    # Minimal Taiwan format with 3 features
    # Need at least 10 samples for train/val/test split
    rows = [" Bankrupt?, Feat1, Feat2, Feat3"]
    for i in range(15):
        label = 1 if i < 5 else 0  # 5 bankrupt, 10 healthy
        rows.append(f"{label},{i * 0.1:.1f},{i * 0.2:.1f},{i * 0.3:.1f}")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _write_us_dataset(base_dir: Path) -> Path:
    """Write a minimal US-style CSV dataset for testing."""
    us_dir = base_dir / "us_data"
    us_dir.mkdir(parents=True, exist_ok=True)
    path = us_dir / "american_bankruptcy.csv"
    # US format: company_name, status_label, year, X1-X18
    headers = ["company_name", "status_label", "year"] + [f"X{i}" for i in range(1, 19)]
    rows = [",".join(headers)]
    for i in range(15):
        status = "failed" if i < 5 else "alive"
        values = [f"company_{i}", status, "2020"] + [f"{i * 0.1:.1f}" for _ in range(18)]
        rows.append(",".join(values))
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _write_polish_dataset(base_dir: Path) -> Path:
    """Write a minimal Polish-style ARFF dataset for testing."""
    polish_dir = base_dir / "polish_data"
    polish_dir.mkdir(parents=True, exist_ok=True)
    path = polish_dir / "1year.arff"

    attrs = "\n".join([f"@attribute Attr{i} numeric" for i in range(1, 65)])
    rows = ["@relation test", attrs, "@attribute class {0,1}", "", "@data"]

    for i in range(15):
        label = 1 if i < 5 else 0
        features = ",".join([f"{i * 0.01:.2f}" for _ in range(64)])
        rows.append(f"{features},{label}")

    path.write_text("\n".join(rows), encoding="utf-8")
    return path


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


class TestOptionalHelpers:
    """Tests for _optional_float and _optional_int helpers."""

    def test_optional_float_returns_default_on_missing(self) -> None:
        """_optional_float returns default when key is missing."""
        assert _optional_float({}, "missing", 0.5) == 0.5

    def test_optional_float_returns_value_when_present(self) -> None:
        """_optional_float returns value when present."""
        assert _optional_float({"val": 0.8}, "val", 0.5) == 0.8

    def test_optional_float_converts_int_to_float(self) -> None:
        """_optional_float converts int to float."""
        assert _optional_float({"val": 5}, "val", 0.0) == 5.0

    def test_optional_float_raises_on_invalid_type(self) -> None:
        """_optional_float raises JSONTypeError on invalid type."""
        with pytest.raises(JSONTypeError, match="must be a number"):
            _optional_float({"val": "string"}, "val", 0.0)

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


class TestParseExternalTrainConfig:
    """Tests for _parse_external_train_config function."""

    def test_parse_config_valid_taiwan(self) -> None:
        """Parse valid config for Taiwan dataset."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        result = _parse_external_train_config(config_json)

        assert result["dataset"] == "taiwan"
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["device"] == "auto"  # default

    def test_parse_config_valid_us(self) -> None:
        """Parse valid config for US dataset."""
        config_json = dump_json_str(
            {
                "dataset": "us",
                "learning_rate": 0.2,
                "max_depth": 4,
                "n_estimators": 50,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 99,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["dataset"] == "us"

    def test_parse_config_valid_polish(self) -> None:
        """Parse valid config for Polish dataset."""
        config_json = dump_json_str(
            {
                "dataset": "polish",
                "learning_rate": 0.15,
                "max_depth": 5,
                "n_estimators": 100,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 7,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["dataset"] == "polish"

    def test_parse_config_invalid_dataset(self) -> None:
        """Invalid dataset name raises ValueError."""
        config_json = dump_json_str(
            {
                "dataset": "invalid",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        with pytest.raises(ValueError, match="dataset must be one of"):
            _parse_external_train_config(config_json)

    def test_parse_config_invalid_ratios(self) -> None:
        """Ratios that don't sum to 1.0 raise ValueError."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "train_ratio": 0.5,
                "val_ratio": 0.5,
                "test_ratio": 0.5,  # Sum = 1.5
            }
        )
        with pytest.raises(ValueError, match=r"Split ratios must sum to 1\.0"):
            _parse_external_train_config(config_json)

    def test_parse_config_not_object(self) -> None:
        """Non-object JSON raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            _parse_external_train_config("[1, 2, 3]")

    def test_parse_config_invalid_json(self) -> None:
        """Invalid JSON raises InvalidJsonError."""
        with pytest.raises(InvalidJsonError):
            _parse_external_train_config("not json")

    def test_parse_config_with_custom_ratios(self) -> None:
        """Config with custom split ratios is accepted."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["config"]["train_ratio"] == 0.8
        assert result["config"]["val_ratio"] == 0.1
        assert result["config"]["test_ratio"] == 0.1

    def test_parse_config_with_scale_pos_weight(self) -> None:
        """Config with scale_pos_weight is parsed correctly."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "scale_pos_weight": 2.5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["backend"] == "xgboost"
        xgb_config: TrainConfig = result["config"]  # type narrowed by backend check
        assert xgb_config.get("scale_pos_weight") == 2.5

    def test_parse_config_mlp_backend(self) -> None:
        """Parse valid config for MLP backend."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "taiwan",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64, 32],
                "precision": "fp32",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["backend"] == "mlp"
        assert result["dataset"] == "taiwan"
        assert result["config"]["learning_rate"] == 0.001
        assert result["config"]["hidden_sizes"] == (64, 32)

    def test_parse_config_mlp_precision_fp16(self) -> None:
        """Parse MLP config with fp16 precision."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64],
                "precision": "fp16",
                "optimizer": "adam",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["backend"] == "mlp"
        mlp_config: MLPConfig = result["config"]
        assert mlp_config["precision"] == "fp16"
        assert mlp_config["optimizer"] == "adam"

    def test_parse_config_mlp_precision_bf16(self) -> None:
        """Parse MLP config with bf16 precision."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64],
                "precision": "bf16",
                "optimizer": "sgd",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["backend"] == "mlp"
        mlp_config: MLPConfig = result["config"]
        assert mlp_config["precision"] == "bf16"
        assert mlp_config["optimizer"] == "sgd"

    def test_parse_config_mlp_precision_auto(self) -> None:
        """Parse MLP config with auto precision."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64],
                "precision": "auto",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["backend"] == "mlp"
        mlp_config: MLPConfig = result["config"]
        assert mlp_config["precision"] == "auto"

    def test_parse_config_mlp_invalid_precision(self) -> None:
        """Invalid precision raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64],
                "precision": "invalid",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        with pytest.raises(JSONTypeError, match="precision must be"):
            _parse_external_train_config(config_json)

    def test_parse_config_mlp_invalid_optimizer(self) -> None:
        """Invalid optimizer raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64],
                "precision": "fp32",
                "optimizer": "invalid",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        with pytest.raises(JSONTypeError, match="optimizer must be"):
            _parse_external_train_config(config_json)

    def test_parse_config_mlp_invalid_hidden_sizes_not_list(self) -> None:
        """hidden_sizes not a list raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": "not_a_list",
                "precision": "fp32",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        with pytest.raises(JSONTypeError, match="hidden_sizes must be list"):
            _parse_external_train_config(config_json)

    def test_parse_config_mlp_invalid_hidden_sizes_not_ints(self) -> None:
        """hidden_sizes with non-int elements raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64, "not_int"],
                "precision": "fp32",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        with pytest.raises(JSONTypeError, match="hidden_sizes must be list"):
            _parse_external_train_config(config_json)

    def test_parse_config_invalid_scale_pos_weight_type(self) -> None:
        """scale_pos_weight with non-number value raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "scale_pos_weight": "not_a_number",
            }
        )
        with pytest.raises(JSONTypeError, match="scale_pos_weight must be a number"):
            _parse_external_train_config(config_json)


class TestLoadDataset:
    """Tests for _load_dataset function."""

    def test_load_taiwan_dataset(self, tmp_path: Path) -> None:
        """_load_dataset loads Taiwan data successfully."""
        _write_taiwan_dataset(tmp_path)
        dataset = _load_dataset("taiwan", tmp_path)

        assert dataset["n_samples"] == 15
        assert dataset["n_features"] == 3

    def test_load_us_dataset(self, tmp_path: Path) -> None:
        """_load_dataset loads US data successfully."""
        _write_us_dataset(tmp_path)
        dataset = _load_dataset("us", tmp_path)

        assert dataset["n_samples"] == 15
        assert dataset["n_features"] == 18

    def test_load_polish_dataset(self, tmp_path: Path) -> None:
        """_load_dataset loads Polish data successfully."""
        _write_polish_dataset(tmp_path)
        dataset = _load_dataset("polish", tmp_path)

        assert dataset["n_samples"] == 15
        assert dataset["n_features"] == 64

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


class TestMetricsToJson:
    """Tests for _metrics_to_json function."""

    def test_metrics_to_json_conversion(self) -> None:
        """_metrics_to_json correctly converts EvalMetrics."""
        from covenant_ml.types import EvalMetrics

        metrics: EvalMetrics = {
            "loss": 0.25,
            "ppl": 1.284,
            "auc": 0.85,
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 0.70,
            "f1_score": 0.72,
        }
        result = _metrics_to_json(metrics)

        assert result["loss"] == 0.25
        assert result["auc"] == 0.85
        assert result["accuracy"] == 0.80
        assert result["precision"] == 0.75
        assert result["recall"] == 0.70
        assert result["f1_score"] == 0.72


class TestRunExternalTraining:
    """Integration tests for run_external_training."""

    def test_train_taiwan_produces_model_with_importances(self, tmp_path: Path) -> None:
        """run_external_training trains model and returns feature importances."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        _write_taiwan_dataset(external_dir)

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.3,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
            }
        )

        result = run_external_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        assert result["samples_total"] == 15
        assert result["n_features"] == 3

        # Verify model files were created
        model_path = Path(str(result["model_path"]))
        assert model_path.exists()

        active_path = Path(str(result["active_model_path"]))
        assert active_path.exists()

        # Verify feature importances
        importances = result["feature_importances"]
        assert type(importances) is list
        assert len(importances) == 3  # 3 features

        # First importance should have rank 1
        first_imp = narrow_json_to_dict(importances[0])
        assert require_int(first_imp, "rank") == 1
        assert require_str(first_imp, "name") in ["Feat1", "Feat2", "Feat3"]
        assert require_float(first_imp, "importance") >= 0.0

    def test_train_us_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains model on US data."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        _write_us_dataset(external_dir)

        config_json = dump_json_str(
            {
                "dataset": "us",
                "learning_rate": 0.3,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
            }
        )

        result = run_external_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "us"
        assert result["n_features"] == 18

    def test_train_polish_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains model on Polish data."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        _write_polish_dataset(external_dir)

        config_json = dump_json_str(
            {
                "dataset": "polish",
                "learning_rate": 0.3,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
            }
        )

        result = run_external_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "polish"
        assert result["n_features"] == 64

    def test_train_mlp_backend_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains MLP model on Taiwan data."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        _write_taiwan_dataset(external_dir)

        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "taiwan",
                "learning_rate": 0.01,
                "batch_size": 4,
                "n_epochs": 5,  # Enough epochs to learn from small dataset
                "dropout": 0.0,
                "hidden_sizes": [8, 4],
                "precision": "fp32",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )

        result = run_external_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        # MLP produces a .pt file
        model_path = Path(str(result["model_path"]))
        assert model_path.exists()
        assert model_path.suffix == ".pt"


class TestProcessExternalTrainJob:
    """Tests for process_external_train_job RQ entry point."""

    def test_process_job_loads_settings_and_runs(self, tmp_path: Path) -> None:
        """process_external_train_job loads settings from env and runs training."""
        from platform_core.config import _test_hooks as config_hooks
        from platform_core.testing import FakeEnv

        # Create fake data directories
        data_root = tmp_path / "data"
        external_dir = data_root / "external"
        models_dir = tmp_path / "models"

        _write_taiwan_dataset(external_dir)

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
                    "learning_rate": 0.3,
                    "max_depth": 3,
                    "n_estimators": 10,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "random_state": 42,
                }
            )

            result = process_external_train_job(config_json)

            assert result["status"] == "complete"
            assert result["dataset"] == "taiwan"

            # Verify model was created
            model_path = Path(str(result["model_path"]))
            assert model_path.exists()
        finally:
            config_hooks.get_env = orig_get_env
