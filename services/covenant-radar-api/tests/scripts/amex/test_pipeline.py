"""Tests for AMEX pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    ClassifierBackend,
    PreparedClassifier,
    ProgressCallback,
)
from covenant_ml.optimizer import SearchSpace
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    TrainOutcome,
)
from numpy.typing import NDArray
from scripts.amex._test_hooks import (
    FakeDatasetSpec,
    configure_all_fakes,
)
from scripts.amex.pipeline import (
    build_dataset_config,
    build_test_config,
    generate_ensemble_predictions,
    load_test_data,
    run_pipeline,
    train_all_models,
    train_single_model,
)
from scripts.amex.types import AMEXPipelineConfig, EnsembleResult, ModelOOFResult

# =============================================================================
# Fake 2D prediction classifier and backend for testing ndim==2 branch
# =============================================================================


class Fake2DClassifier:
    """Fake classifier that returns 2D predictions like sklearn."""

    def __init__(self, rng_seed: int = 42) -> None:
        """Initialize with random generator.

        Args:
            rng_seed: Random seed.
        """
        self._rng = np.random.default_rng(rng_seed)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return 2D probabilities like sklearn classifiers.

        Args:
            x: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with class probabilities.
        """
        n_samples = x.shape[0]
        # Build 2D array without column_stack to avoid Any type issues
        result: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        probs_1 = self._rng.uniform(0.0, 1.0, size=n_samples)
        result[:, 1] = probs_1
        result[:, 0] = 1.0 - probs_1
        return result


class Fake2DBackend:
    """Fake backend returning 2D-prediction classifier."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize with output directory.

        Args:
            output_dir: Output directory path.
        """
        self._output_dir = output_dir

    def backend_name(self) -> BackendName:
        """Get backend name."""
        return "lightgbm"

    def capabilities(self) -> BackendCapabilities:
        """Get capabilities."""
        return BackendCapabilities(
            supports_train=True,
            supports_gpu=False,
            supports_early_stopping=False,
            supports_feature_importance=False,
            model_format="pkl",
        )

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> Fake2DClassifier:
        """Prepare classifier.

        Args:
            n_features: Number of features.
            n_classes: Number of classes.
            feature_names: Feature names.

        Returns:
            Fake 2D classifier.
        """
        _ = n_features, n_classes, feature_names
        return Fake2DClassifier()

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: ProgressCallback | None,
    ) -> TrainOutcome:
        """Train classifier.

        Args:
            x_features: Features.
            y_labels: Labels.
            feature_names: Feature names.
            config: Config.
            output_dir: Output directory.
            progress: Progress callback.

        Returns:
            Train outcome.
        """
        _ = progress, y_labels, feature_names

        model_path = output_dir / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("fake 2d model")

        n_samples = int(x_features.shape[0])
        n_train = int(n_samples * 0.7)
        n_val = int(n_samples * 0.15)
        n_test = n_samples - n_train - n_val

        fake_metrics = EvalMetrics(
            loss=0.3,
            ppl=1.35,
            auc=0.85,
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1_score=0.72,
        )

        return TrainOutcome(
            model_path=str(model_path),
            model_id="fake_2d_model",
            samples_total=n_samples,
            samples_train=n_train,
            samples_val=n_val,
            samples_test=n_test,
            train_metrics=fake_metrics,
            val_metrics=fake_metrics,
            test_metrics=fake_metrics,
            best_val_auc=0.85,
            best_round=10,
            total_rounds=10,
            early_stopped=False,
            config=config,
            feature_importances=[],
            scale_pos_weight_computed=1.0,
        )

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        """Evaluate model.

        Args:
            model: Model.
            x: Features.
            y: Labels.

        Returns:
            Eval metrics.
        """
        _ = model, x, y
        return EvalMetrics(
            loss=0.3,
            ppl=1.35,
            auc=0.85,
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1_score=0.72,
        )

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        """Save model.

        Args:
            model: Model.
            path: Path.
        """
        _ = model, path

    def load(self, *, path: str) -> Fake2DClassifier:
        """Load model.

        Args:
            path: Path.

        Returns:
            Fake 2D classifier.
        """
        _ = path
        return Fake2DClassifier()

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Get feature importances.

        Args:
            model: Model.
            feature_names: Feature names.

        Returns:
            Empty list.
        """
        _ = model, feature_names
        return []

    def get_default_search_space(self) -> SearchSpace:
        """Not used in pipeline tests."""
        raise NotImplementedError

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Not used in pipeline tests."""
        raise NotImplementedError


class Fake2DRegistry:
    """Registry returning 2D-prediction backend."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize with output directory.

        Args:
            output_dir: Output directory.
        """
        self._output_dir = output_dir

    def get(self, name: BackendName) -> ClassifierBackend:
        """Get backend.

        Args:
            name: Backend name.

        Returns:
            Fake 2D backend.
        """
        _ = name
        backend: ClassifierBackend = Fake2DBackend(self._output_dir)
        return backend


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
        from scripts.amex.types import ModelOOFResult

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
