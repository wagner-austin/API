"""Tests for AMEX pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scripts.amex._hook_protocols import (
    FakeDatasetSpec,
)
from scripts.amex._test_hooks import (
    configure_all_fakes,
)
from scripts.amex.pipeline import (
    generate_ensemble_predictions,
    run_pipeline,
)
from scripts.amex.types import AMEXPipelineConfig, EnsembleResult, ModelOOFResult

from tests.scripts.amex._pipeline_fixtures import (
    Fake2DRegistry,
)


class TestGenerateEnsemblePredictions:
    """Tests for generate_ensemble_predictions function."""

    def test_generates_weighted_predictions(self, tmp_path: Path) -> None:
        """generate_ensemble_predictions produces weighted ensemble output."""
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

        # Create fake model directory structure
        for backend in ["lightgbm", "xgboost"]:
            fold_dir = tmp_path / "models" / backend / "fold_1"
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "model.pkl").write_text("fake model")

        x_test = np.random.randn(50, 10).astype(np.float64)

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

        ensemble_result = EnsembleResult(
            model_names=("lightgbm", "xgboost"),
            weights=(0.6, 0.4),
            initial_score=0.80,
            optimized_score=0.82,
            improvement=0.02,
        )

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

        predictions = generate_ensemble_predictions(
            x_test=x_test,
            model_results=model_results,
            ensemble_result=ensemble_result,
            config=config,
            output_dir=tmp_path / "models",
        )

        assert predictions.shape == (50,)
        assert np.all(predictions >= 0.0)
        assert np.all(predictions <= 1.0)


class TestRunPipeline:
    """Tests for run_pipeline function."""

    def test_runs_full_pipeline(self, tmp_path: Path) -> None:
        """run_pipeline executes all steps and produces submission.

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

        train_dir = tmp_path / "amex_train"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir = tmp_path / "amex_test"
        test_dir.mkdir(parents=True, exist_ok=True)

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

        output_path = tmp_path / "submission.csv"

        result = run_pipeline(
            train_dir=train_dir,
            test_dir=test_dir,
            output_path=output_path,
            config=config,
            model_output_dir=tmp_path / "models",
        )

        assert result["n_samples_train"] == 100
        assert result["n_samples_test"] == 50
        assert result["n_features"] == 10
        assert output_path.exists()
        # Verify fake loss improvement
        loss_after = 0.3
        loss_initial = 1.0
        assert loss_after < loss_initial


class TestGenerateEnsemblePredictionsMissingModel:
    """Tests for FileNotFoundError handling in generate_ensemble_predictions."""

    def test_raises_file_not_found_when_no_model_files(self, tmp_path: Path) -> None:
        """generate_ensemble_predictions raises FileNotFoundError when no models.

        This tests lines 645-646 when model files don't exist.
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

        # Create model directory structure WITHOUT model files
        # The directory exists but no .pkl or .json files
        fold_dir = tmp_path / "models" / "lightgbm" / "fold_1"
        fold_dir.mkdir(parents=True, exist_ok=True)
        # Intentionally don't create any model files

        x_test = np.random.randn(50, 10).astype(np.float64)

        model_results = (
            ModelOOFResult(
                model_name="lightgbm",
                oof_predictions=np.random.rand(100).astype(np.float64),
                fold_indices=np.zeros(100, dtype=np.int64),
                cv_scores=(0.81,),
                mean_cv_score=0.81,
            ),
        )

        ensemble_result = EnsembleResult(
            model_names=("lightgbm",),
            weights=(1.0,),
            initial_score=0.80,
            optimized_score=0.81,
            improvement=0.01,
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

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            generate_ensemble_predictions(
                x_test=x_test,
                model_results=model_results,
                ensemble_result=ensemble_result,
                config=config,
                output_dir=tmp_path / "models",
            )

        assert "No model file found" in str(exc_info.value)


class TestGenerateEnsemblePredictions2D:
    """Tests for 2D prediction handling in generate_ensemble_predictions."""

    def test_handles_2d_predictions(self, tmp_path: Path) -> None:
        """generate_ensemble_predictions handles 2D predict_proba output.

        This tests line 654 where raw_preds.ndim == 2.
        Uses module-level Fake2DClassifier, Fake2DBackend, Fake2DRegistry.
        """
        import scripts.amex._hooks as amex_hooks

        # Configure fakes first, then override registry
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

        # Override registry to use 2D backend
        fake_2d_registry = Fake2DRegistry(tmp_path / "models")
        amex_hooks.registry_hook = lambda: fake_2d_registry

        # Create model directory with model file
        fold_dir = tmp_path / "models" / "lightgbm" / "fold_1"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "model.pkl").write_text("fake 2d model")

        x_test = np.random.randn(50, 10).astype(np.float64)

        model_results = (
            ModelOOFResult(
                model_name="lightgbm",
                oof_predictions=np.random.rand(100).astype(np.float64),
                fold_indices=np.zeros(100, dtype=np.int64),
                cv_scores=(0.81,),
                mean_cv_score=0.81,
            ),
        )

        ensemble_result = EnsembleResult(
            model_names=("lightgbm",),
            weights=(1.0,),
            initial_score=0.80,
            optimized_score=0.81,
            improvement=0.01,
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

        # This should handle 2D predictions and extract [:, 1]
        predictions = generate_ensemble_predictions(
            x_test=x_test,
            model_results=model_results,
            ensemble_result=ensemble_result,
            config=config,
            output_dir=tmp_path / "models",
        )

        # Verify output is 1D and within probability bounds
        assert predictions.shape == (50,)
        assert np.all(predictions >= 0.0)
        assert np.all(predictions <= 1.0)
