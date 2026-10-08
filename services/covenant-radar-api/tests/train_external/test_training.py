"""Integration tests for run_external_training."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    dump_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)

from covenant_radar_api.worker.train_external_job import run_external_training
from tests._real_datasets import copy_real_polish, copy_real_taiwan, copy_real_us

# Trains torch models: shares one xdist worker with the other torch modules
# (tests/conftest.py, pytest_itemcollected).
pytestmark = pytest.mark.xdist_group("torch")


class TestXGBoostTraining:
    """Tests for XGBoost model training."""

    def test_taiwan_produces_model_with_importances(self, tmp_path: Path) -> None:
        """run_external_training trains model and returns feature importances."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        _, n_rows, feature_names = copy_real_taiwan(external_dir)

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
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
        assert result["samples_total"] == n_rows
        assert result["n_features"] == len(feature_names)

        model_path = Path(str(result["model_path"]))
        assert model_path.exists()

        active_path = Path(str(result["active_model_path"]))
        assert active_path.exists()

        importances = result["feature_importances"]
        assert type(importances) is list
        assert len(importances) == len(feature_names)

        first_imp = narrow_json_to_dict(importances[0])
        assert require_int(first_imp, "rank") == 1
        assert require_str(first_imp, "name") in feature_names
        assert require_float(first_imp, "importance") >= 0.0

    def test_us_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains model on US data."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        _, _n_rows, feature_names = copy_real_us(external_dir)

        config_json = dump_json_str(
            {
                "dataset": "us",
                "device": "cpu",
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
        assert result["n_features"] == len(feature_names)

    def test_polish_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains model on Polish data."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        _, _n_rows, feature_names = copy_real_polish(external_dir)

        config_json = dump_json_str(
            {
                "dataset": "polish",
                "device": "cpu",
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
        assert result["n_features"] == len(feature_names)


class TestMLPTraining:
    """Tests for MLP model training."""

    def test_taiwan_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains MLP model on a sample of the real Taiwan dataset."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        copy_real_taiwan(external_dir)

        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "taiwan",
                "device": "cpu",
                "learning_rate": 0.01,
                "batch_size": 1024,
                "n_epochs": 3,
                "dropout": 0.1,
                "hidden_sizes": [64, 32],
                "precision": "fp32",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 2,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            }
        )

        result = run_external_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        model_path = Path(str(result["model_path"]))
        assert model_path.exists()
        assert model_path.suffix == ".pt"

    def test_us_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains MLP on a sample of the real US dataset."""
        external_dir = tmp_path / "external"
        copy_real_us(external_dir)

        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "device": "cpu",
                "learning_rate": 0.05,
                "batch_size": 1024,
                "n_epochs": 3,
                "dropout": 0.1,
                "hidden_sizes": [64, 32],
                "precision": "fp32",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 2,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            }
        )

        result = run_external_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "us"
        model_path = Path(str(result["model_path"]))
        assert model_path.exists() and model_path.suffix == ".pt"
        assert require_float(result, "best_val_auc") >= 0.5


class TestLSTMTraining:
    """Tests for LSTM model training."""

    def test_taiwan_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains LSTM model on a sample of the real Taiwan dataset."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        copy_real_taiwan(external_dir)

        config_json = dump_json_str(
            {
                "backend": "lstm",
                "dataset": "taiwan",
                "device": "cpu",
                "learning_rate": 0.01,
                "batch_size": 1024,
                "n_epochs": 2,
                "dropout": 0.0,
                "hidden_size": 32,
                "num_layers": 1,
                "bidirectional": False,
                "sequence_length": 4,
                "precision": "fp32",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 2,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            }
        )

        result = run_external_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        model_path = Path(str(result["model_path"]))
        assert model_path.exists()
        assert model_path.suffix == ".pt"


class TestClearGBMTraining:
    """Tests for ClearGBM model training."""

    def test_taiwan_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains ClearGBM model on Taiwan dataset."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        _, _n_rows, _feature_names = copy_real_taiwan(external_dir)

        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "device": "cpu",
                "n_estimators": 5,
                "max_depth": 3,
                "learning_rate": 0.3,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": None,
                "subsample": 1.0,
                "random_state": 42,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            }
        )

        result = run_external_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        model_path = Path(str(result["model_path"]))
        assert model_path.exists()
        assert model_path.suffix == ".json"


class TestLogRegTraining:
    """Tests for LogReg model training."""

    def test_taiwan_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains LogReg model on Taiwan dataset."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        _, _n_rows, _feature_names = copy_real_taiwan(external_dir)

        config_json = dump_json_str(
            {
                "backend": "logreg",
                "dataset": "taiwan",
                "device": "cpu",
                "solver": "saga",
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 100,
                "tol": 0.001,
                "class_weight_balanced": True,
                "random_state": 42,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            }
        )

        result = run_external_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        model_path = Path(str(result["model_path"]))
        assert model_path.exists()
        assert model_path.suffix == ".joblib"

        # LogReg should produce metadata
        active_path = Path(str(result["active_model_path"]))
        meta_path = active_path.parent / "active_logreg_meta.json"
        assert meta_path.exists()


class TestRandomForestTraining:
    """Tests for RandomForest model training."""

    def test_taiwan_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains RF model on Taiwan dataset."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        _, _n_rows, _feature_names = copy_real_taiwan(external_dir)

        config_json = dump_json_str(
            {
                "backend": "random_forest",
                "dataset": "taiwan",
                "device": "cpu",
                "n_estimators": 10,
                "max_depth": 5,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight_balanced": True,
                "random_state": 42,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            }
        )

        result = run_external_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        model_path = Path(str(result["model_path"]))
        assert model_path.exists()
        assert model_path.suffix == ".joblib"

        # RF should produce metadata
        active_path = Path(str(result["active_model_path"]))
        meta_path = active_path.parent / "active_rf_meta.json"
        assert meta_path.exists()


class TestLightGBMTraining:
    """Tests for LightGBM model training."""

    def test_taiwan_produces_model(self, tmp_path: Path) -> None:
        """run_external_training trains LightGBM model on a sample of the real Taiwan dataset."""
        external_dir = tmp_path / "external"
        output_dir = tmp_path / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        copy_real_taiwan(external_dir)

        config_json = dump_json_str(
            {
                "backend": "lightgbm",
                "dataset": "taiwan",
                "device": "cpu",
                "learning_rate": 0.1,
                "n_estimators": 10,
                "max_depth": 3,
                "num_leaves": 8,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "random_state": 42,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            }
        )

        result = run_external_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        model_path = Path(str(result["model_path"]))
        assert model_path.exists()
        assert model_path.suffix == ".txt"
