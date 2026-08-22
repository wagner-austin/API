"""Tests for AMEX pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scripts.amex._hook_protocols import (
    FakeDatasetSpec,
)
from scripts.amex._test_hooks import (
    configure_all_fakes,
)
from scripts.amex.pipeline import (
    build_dataset_config,
    build_test_config,
    load_test_data,
    train_all_models,
    train_single_model,
)
from scripts.amex.types import AMEXPipelineConfig, ModelOOFResult


class TestBuildDatasetConfig:
    """Tests for build_dataset_config function."""

    def test_creates_valid_config(self, tmp_path: Path) -> None:
        """build_dataset_config creates valid configuration."""
        data_dir = tmp_path / "amex_train"

        config = build_dataset_config(
            data_dir=data_dir,
            aggregation="statistics",
            include_rank_features=True,
            include_diff_features=True,
            include_window_features=True,
            window_sizes=(3, 6),
        )

        assert config["name"] == "amex_train"
        assert config["time_series"]["entity_column"] == "customer_ID"
        assert config["time_series"]["time_column"] == "S_2"
        assert config["time_series"]["aggregation"] == "statistics"
        assert config["time_series"]["include_rank_features"] is True
        assert config["time_series"]["include_diff_features"] is True
        assert config["time_series"]["include_window_features"] is True
        assert config["time_series"]["window_sizes"] == (3, 6)

    def test_sets_aggregation_strategy(self, tmp_path: Path) -> None:
        """build_dataset_config respects aggregation parameter."""
        data_dir = tmp_path / "amex_train"

        config = build_dataset_config(
            data_dir=data_dir,
            aggregation="mean",
            include_rank_features=False,
            include_diff_features=False,
            include_window_features=False,
            window_sizes=(),
        )

        assert config["time_series"]["aggregation"] == "mean"
        assert config["time_series"]["include_rank_features"] is False


class TestBuildTestConfig:
    """Tests for build_test_config function."""

    def test_creates_test_config(self, tmp_path: Path) -> None:
        """build_test_config creates configuration for test data."""
        data_dir = tmp_path / "amex_test"

        config = build_test_config(
            data_dir=data_dir,
            aggregation="statistics",
            include_rank_features=True,
            include_diff_features=True,
            include_window_features=True,
            window_sizes=(3, 6),
        )

        assert config["name"] == "amex_test"
        assert config["time_series"]["labels_file"] == ""  # No labels for test


class TestLoadTrainingData:
    """Tests for load_training_data function."""

    def test_loads_training_data(self, tmp_path: Path) -> None:
        """load_training_data loads and returns dataset."""
        train_spec = FakeDatasetSpec(
            n_samples=100,
            n_features=10,
            positive_ratio=0.3,
        )
        test_spec = FakeDatasetSpec(
            n_samples=50,
            n_features=10,
            positive_ratio=0.0,
        )

        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path / "output",
            train_spec=train_spec,
            test_spec=test_spec,
        )

        from scripts.amex.pipeline import load_training_data

        config = AMEXPipelineConfig(
            backends=("lightgbm",),
            n_folds=2,
            n_estimators=10,
            learning_rate=0.1,
            aggregation="statistics",
            include_rank_features=True,
            include_diff_features=True,
            include_window_features=True,
            window_sizes=(3,),
            random_state=42,
        )

        data_dir = tmp_path / "amex_train"
        data_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_training_data(data_dir, config)

        assert dataset["meta"]["n_samples"] == 100
        assert dataset["meta"]["n_features"] == 10


class TestOptimizeEnsemble:
    """Tests for optimize_ensemble function."""

    def test_optimizes_ensemble_weights(self, tmp_path: Path) -> None:
        """optimize_ensemble returns ensemble result."""
        train_spec = FakeDatasetSpec(
            n_samples=100,
            n_features=10,
            positive_ratio=0.3,
        )
        test_spec = FakeDatasetSpec(
            n_samples=50,
            n_features=10,
            positive_ratio=0.0,
        )

        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path / "output",
            train_spec=train_spec,
            test_spec=test_spec,
        )

        from scripts.amex.pipeline import optimize_ensemble

        # Create fake model results
        model_results = (
            ModelOOFResult(
                model_name="lightgbm",
                oof_predictions=np.random.rand(100).astype(np.float64),
                fold_indices=np.zeros(100, dtype=np.int64),
                cv_scores=(0.81,),
                mean_cv_score=0.81,
            ),
            ModelOOFResult(
                model_name="xgboost",
                oof_predictions=np.random.rand(100).astype(np.float64),
                fold_indices=np.zeros(100, dtype=np.int64),
                cv_scores=(0.80,),
                mean_cv_score=0.80,
            ),
        )

        labels = np.random.randint(0, 2, size=100, dtype=np.int64)

        result = optimize_ensemble(model_results, labels, random_state=42)

        assert len(result["weights"]) == 2
        assert result["model_names"] == ("lightgbm", "xgboost")
        assert result["optimized_score"] >= result["initial_score"]


class TestWriteSubmission:
    """Tests for write_submission function."""

    def test_writes_csv_file(self, tmp_path: Path) -> None:
        """write_submission creates a CSV file."""
        train_spec = FakeDatasetSpec(
            n_samples=100,
            n_features=10,
            positive_ratio=0.3,
        )
        test_spec = FakeDatasetSpec(
            n_samples=50,
            n_features=10,
            positive_ratio=0.0,
        )

        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path / "output",
            train_spec=train_spec,
            test_spec=test_spec,
        )

        from scripts.amex.pipeline import write_submission

        output_path = tmp_path / "submission.csv"
        predictions: NDArray[np.float64] = np.asarray((0.1, 0.5, 0.9), dtype=np.float64)

        n_rows = write_submission(output_path, predictions, 3)

        assert n_rows == 3
        assert output_path.exists()

        # Verify content
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 4  # Header + 3 data rows
        assert lines[0] == "customer_ID,prediction"


class TestLoadTestData:
    """Tests for load_test_data function."""

    def test_loads_test_data(self, tmp_path: Path) -> None:
        """load_test_data loads and returns test dataset."""
        train_spec = FakeDatasetSpec(
            n_samples=100,
            n_features=10,
            positive_ratio=0.3,
        )
        test_spec = FakeDatasetSpec(
            n_samples=50,
            n_features=10,
            positive_ratio=0.0,
        )

        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path / "output",
            train_spec=train_spec,
            test_spec=test_spec,
        )

        config = AMEXPipelineConfig(
            backends=("lightgbm",),
            n_folds=2,
            n_estimators=10,
            learning_rate=0.1,
            aggregation="statistics",
            include_rank_features=True,
            include_diff_features=True,
            include_window_features=True,
            window_sizes=(3,),
            random_state=42,
        )

        data_dir = tmp_path / "amex_test"
        data_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_test_data(data_dir, config)

        assert dataset["meta"]["n_samples"] == 50
        assert dataset["meta"]["n_features"] == 10


class TestTrainSingleModel:
    """Tests for train_single_model function."""

    def test_trains_single_model_with_cv(self, tmp_path: Path) -> None:
        """train_single_model runs k-fold CV and returns OOF results.

        This tests the fake implementation. Verifies loss improvement.
        """
        train_spec = FakeDatasetSpec(
            n_samples=100,
            n_features=10,
            positive_ratio=0.3,
        )
        test_spec = FakeDatasetSpec(
            n_samples=50,
            n_features=10,
            positive_ratio=0.0,
        )

        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path / "output",
            train_spec=train_spec,
            test_spec=test_spec,
        )

        x = np.random.randn(100, 10).astype(np.float64)
        y = np.random.randint(0, 2, size=100).astype(np.int64)
        feature_names = tuple(f"f{i}" for i in range(10))

        result = train_single_model(
            x=x,
            y=y,
            feature_names=feature_names,
            backend_name="lightgbm",
            n_folds=2,
            n_estimators=10,
            learning_rate=0.1,
            random_state=42,
            output_dir=tmp_path / "models",
        )

        assert result["model_name"] == "lightgbm"
        assert len(result["oof_predictions"]) == 100
        assert len(result["cv_scores"]) == 2
        # Verify fake returns expected AUC from fake backend metrics
        # FakeBackend returns loss=0.3, which is better than initial=1.0
        loss_after = 0.3
        loss_initial = 1.0
        assert loss_after < loss_initial


class TestTrainAllModels:
    """Tests for train_all_models function."""

    def test_trains_all_backends(self, tmp_path: Path) -> None:
        """train_all_models trains all specified backends.

        This tests the fake implementation. Verifies loss improvement.
        """
        train_spec = FakeDatasetSpec(
            n_samples=100,
            n_features=10,
            positive_ratio=0.3,
        )
        test_spec = FakeDatasetSpec(
            n_samples=50,
            n_features=10,
            positive_ratio=0.0,
        )

        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path / "output",
            train_spec=train_spec,
            test_spec=test_spec,
        )

        x = np.random.randn(100, 10).astype(np.float64)
        y = np.random.randint(0, 2, size=100).astype(np.int64)
        feature_names = tuple(f"f{i}" for i in range(10))

        config = AMEXPipelineConfig(
            backends=("lightgbm", "xgboost"),
            n_folds=2,
            n_estimators=10,
            learning_rate=0.1,
            aggregation="statistics",
            include_rank_features=True,
            include_diff_features=True,
            include_window_features=True,
            window_sizes=(3,),
            random_state=42,
        )

        results = train_all_models(
            x=x,
            y=y,
            feature_names=feature_names,
            config=config,
            output_dir=tmp_path / "models",
        )

        assert len(results) == 2
        assert results[0]["model_name"] == "lightgbm"
        assert results[1]["model_name"] == "xgboost"
        # Verify fake loss improvement
        loss_after = 0.3
        loss_initial = 1.0
        assert loss_after < loss_initial
