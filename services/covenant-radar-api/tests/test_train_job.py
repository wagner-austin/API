"""Integration tests for training job with real XGBoost training."""

from __future__ import annotations

from pathlib import Path

import pytest
from covenant_persistence.testing import InMemoryStore
from platform_core.json_utils import (
    InvalidJsonError,
    JSONTypeError,
    dump_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)

from covenant_radar_api.worker.train_job import _parse_train_config, run_training
from tests._train_job_fixtures import (
    _add_covenant_results_for_deal,
    _add_deal,
    _add_measurements_for_deal,
    _TrainingProvider,
)


class TestRunTraining:
    """Tests for run_training job function with real XGBoost training."""

    def test_train_with_valid_data(self, tmp_path: Path) -> None:
        """Test training with valid training data produces a model."""
        store = InMemoryStore()

        # Add multiple deals with varying outcomes (minimum 10 samples required)
        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["North America", "Europe", "Asia"]

        for i in range(12):
            deal_id = f"d{i + 1}"
            sector = sectors[i % 3]
            region = regions[i % 3]
            _add_deal(store, deal_id, sector, region)
            _add_measurements_for_deal(store, deal_id)
            # Alternate breach outcomes for class balance
            has_breach = i % 2 == 0
            _add_covenant_results_for_deal(store, deal_id, f"c{i + 1}", has_breach=has_breach)

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "scale_pos_weight": 1.2,
            }
        )

        result = run_training(config_json, provider)

        assert result["status"] == "complete"
        assert result["samples_total"] == 12

        # Verify model file was created and has valid path
        model_path = Path(str(result["model_path"]))
        assert model_path.exists()
        assert model_path.suffix == ".ubj"

        # Verify model_id is a valid UUID
        model_id = require_str(result, "model_id")
        import uuid

        uuid.UUID(model_id)  # Raises ValueError if invalid

        # Verify config is returned with correct values
        config = narrow_json_to_dict(result["config"])
        assert require_float(config, "learning_rate") == 0.1
        assert require_int(config, "max_depth") == 3
        assert require_float(config, "reg_alpha") == 0.0
        assert require_float(config, "reg_lambda") == 1.0
        assert require_float(config, "scale_pos_weight") == 1.2

    def test_train_with_no_data_raises(self, tmp_path: Path) -> None:
        """Test training with no data raises ValueError."""
        store = InMemoryStore()
        # Empty store - no deals

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
            }
        )

        with pytest.raises(ValueError, match="No training data"):
            run_training(config_json, provider)

    def test_train_with_insufficient_data_raises(self, tmp_path: Path) -> None:
        """Test training with insufficient data (1-9 samples) raises ValueError."""
        store = InMemoryStore()

        # Add only 5 deals (less than minimum 10)
        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["North America", "Europe", "Asia"]

        for i in range(5):
            deal_id = f"d{i + 1}"
            sector = sectors[i % 3]
            region = regions[i % 3]
            _add_deal(store, deal_id, sector, region)
            _add_measurements_for_deal(store, deal_id)
            has_breach = i % 2 == 0
            _add_covenant_results_for_deal(store, deal_id, f"c{i + 1}", has_breach=has_breach)

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
            }
        )

        with pytest.raises(ValueError, match=r"Insufficient training data: 5 samples"):
            run_training(config_json, provider)

    def test_train_with_invalid_config_raises(self, tmp_path: Path) -> None:
        """Test training with invalid config JSON raises."""
        store = InMemoryStore()
        _add_deal(store, "d1", "Technology", "North America")
        _add_measurements_for_deal(store, "d1")
        _add_covenant_results_for_deal(store, "d1", "c1", has_breach=False)

        provider = _TrainingProvider(store, tmp_path)
        config_json = "not valid json"

        with pytest.raises(InvalidJsonError):
            run_training(config_json, provider)

    def test_train_with_missing_config_field_raises(self, tmp_path: Path) -> None:
        """Test training with missing config field raises."""
        store = InMemoryStore()
        _add_deal(store, "d1", "Technology", "North America")
        _add_measurements_for_deal(store, "d1")
        _add_covenant_results_for_deal(store, "d1", "c1", has_breach=False)

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str({"learning_rate": 0.1})  # Missing other fields

        with pytest.raises(JSONTypeError, match="Missing required field"):
            run_training(config_json, provider)

    def test_train_with_invalid_scale_pos_weight_raises(self, tmp_path: Path) -> None:
        """Test training raises when scale_pos_weight has invalid type."""
        store = InMemoryStore()
        provider = _TrainingProvider(store, tmp_path)

        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "scale_pos_weight": "invalid",
            }
        )

        with pytest.raises(JSONTypeError, match="scale_pos_weight must be a number"):
            run_training(config_json, provider)

    def test_train_with_invalid_early_stopping_type_raises(self, tmp_path: Path) -> None:
        """Test training raises when early_stopping_rounds is not numeric."""
        store = InMemoryStore()
        provider = _TrainingProvider(store, tmp_path)

        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "early_stopping_rounds": "ten",
            }
        )

        with pytest.raises(JSONTypeError, match="Field 'early_stopping_rounds' must be a number"):
            run_training(config_json, provider)

    def test_train_with_config_not_object_raises(self, tmp_path: Path) -> None:
        """Test training with config that is not a JSON object raises."""
        store = InMemoryStore()
        _add_deal(store, "d1", "Technology", "North America")
        _add_measurements_for_deal(store, "d1")
        _add_covenant_results_for_deal(store, "d1", "c1", has_breach=False)

        provider = _TrainingProvider(store, tmp_path)
        config_json = "[1, 2, 3]"  # JSON array, not object

        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            run_training(config_json, provider)

    def test_train_with_invalid_ratio_type_raises(self, tmp_path: Path) -> None:
        """Test training raises when optional ratio is not numeric."""
        store = InMemoryStore()
        provider = _TrainingProvider(store, tmp_path)

        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "train_ratio": "heavy",
            }
        )

        with pytest.raises(JSONTypeError, match="Field 'train_ratio' must be a number"):
            run_training(config_json, provider)

    def test_train_with_invalid_device_raises(self, tmp_path: Path) -> None:
        """Test training raises when device is unsupported."""
        store = InMemoryStore()
        provider = _TrainingProvider(store, tmp_path)

        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "device": "tpu",
            }
        )

        with pytest.raises(ValueError, match="device must be one of: cpu, cuda, auto"):
            run_training(config_json, provider)

    def test_parse_train_config_accepts_cpu_cuda_and_auto(self) -> None:
        """Test _parse_train_config handles supported device values."""
        base = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 10,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 42,
        }
        config_cpu = _parse_train_config(dump_json_str({**base, "device": "cpu"}))
        assert config_cpu["device"] == "cpu"
        config_cuda = _parse_train_config(dump_json_str({**base, "device": "cuda"}))
        assert config_cuda["device"] == "cuda"
        config_auto = _parse_train_config(dump_json_str({**base, "device": "auto"}))
        assert config_auto["device"] == "auto"

    def test_parse_train_config_rejects_non_string_device(self) -> None:
        """Test _parse_train_config raises for non-string device types."""
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "device": 99,
            }
        )

        with pytest.raises(JSONTypeError, match="device must be a string"):
            _parse_train_config(config_json)

    def test_train_with_explicit_split_ratios(self, tmp_path: Path) -> None:
        """Test training with explicitly provided split ratios."""
        store = InMemoryStore()

        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["North America", "Europe", "Asia"]

        for i in range(12):
            deal_id = f"d{i + 1}"
            sector = sectors[i % 3]
            region = regions[i % 3]
            _add_deal(store, deal_id, sector, region)
            _add_measurements_for_deal(store, deal_id)
            has_breach = i % 2 == 0
            _add_covenant_results_for_deal(store, deal_id, f"c{i + 1}", has_breach=has_breach)

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "early_stopping_rounds": 5,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            }
        )

        result = run_training(config_json, provider)

        assert result["status"] == "complete"
        assert result["samples_total"] == 12
        # Config should reflect provided values
        config = narrow_json_to_dict(result["config"])
        assert require_float(config, "train_ratio") == 0.8
        assert require_float(config, "val_ratio") == 0.1
        assert require_float(config, "test_ratio") == 0.1
        assert require_int(config, "early_stopping_rounds") == 5
        assert require_float(config, "reg_alpha") == 1.0
        assert require_float(config, "reg_lambda") == 5.0

    def test_train_with_float_early_stopping(self, tmp_path: Path) -> None:
        """Test training with early_stopping_rounds as float (converts to int)."""
        store = InMemoryStore()

        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["North America", "Europe", "Asia"]

        for i in range(12):
            deal_id = f"d{i + 1}"
            sector = sectors[i % 3]
            region = regions[i % 3]
            _add_deal(store, deal_id, sector, region)
            _add_measurements_for_deal(store, deal_id)
            has_breach = i % 2 == 0
            _add_covenant_results_for_deal(store, deal_id, f"c{i + 1}", has_breach=has_breach)

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "early_stopping_rounds": 8.0,  # Float value
            }
        )

        result = run_training(config_json, provider)

        assert result["status"] == "complete"
        config = narrow_json_to_dict(result["config"])
        assert require_int(config, "early_stopping_rounds") == 8

    def test_train_with_invalid_ratios_raises(self, tmp_path: Path) -> None:
        """Test training with ratios that don't sum to 1.0 raises ValueError."""
        store = InMemoryStore()
        _add_deal(store, "d1", "Technology", "North America")
        _add_measurements_for_deal(store, "d1")
        _add_covenant_results_for_deal(store, "d1", "c1", has_breach=False)

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "train_ratio": 0.5,
                "val_ratio": 0.3,
                "test_ratio": 0.3,  # Sum = 1.1
            }
        )

        with pytest.raises(ValueError, match=r"Split ratios must sum to 1\.0"):
            run_training(config_json, provider)

    def test_train_skips_deals_without_measurements(self, tmp_path: Path) -> None:
        """Test training skips deals that have no measurements."""
        store = InMemoryStore()

        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["North America", "Europe", "Asia"]

        # Add 12 deals with measurements
        for i in range(12):
            deal_id = f"d{i + 1}"
            sector = sectors[i % 3]
            region = regions[i % 3]
            _add_deal(store, deal_id, sector, region)
            _add_measurements_for_deal(store, deal_id)
            has_breach = i % 2 == 0
            _add_covenant_results_for_deal(store, deal_id, f"c{i + 1}", has_breach=has_breach)

        # Add deal without measurements - should be skipped
        _add_deal(store, "d-no-data", "Finance", "Europe")
        # No measurements for d-no-data

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
            }
        )

        result = run_training(config_json, provider)

        # Only 12 deals should be trained (d-no-data skipped)
        assert result["samples_total"] == 12

    def test_train_model_file_has_unique_name(self, tmp_path: Path) -> None:
        """Test that each training run produces a uniquely named model file."""
        store = InMemoryStore()

        # Add minimum 10 deals for training
        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["North America", "Europe", "Asia"]

        for i in range(12):
            deal_id = f"d{i + 1}"
            sector = sectors[i % 3]
            region = regions[i % 3]
            _add_deal(store, deal_id, sector, region)
            _add_measurements_for_deal(store, deal_id)
            has_breach = i % 2 == 0
            _add_covenant_results_for_deal(store, deal_id, f"c{i + 1}", has_breach=has_breach)

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
            }
        )

        result1 = run_training(config_json, provider)
        result2 = run_training(config_json, provider)

        # Model IDs should be different
        assert result1["model_id"] != result2["model_id"]

        # Model paths should be different
        assert result1["model_path"] != result2["model_path"]

        # Both model files should exist
        assert Path(str(result1["model_path"])).exists()
        assert Path(str(result2["model_path"])).exists()
