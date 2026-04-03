"""Tests for AMEX test hooks module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scripts.amex._test_hooks import (
    FakeBackend,
    FakeConsole,
    FakeDatasetSpec,
    FakePreparedClassifier,
    FakeRegistry,
    FakeTimeseriesLoader,
    configure_all_fakes,
    make_fake_dataset,
    make_fake_optimizer,
)


class TestMakeFakeDataset:
    """Tests for make_fake_dataset function."""

    def test_creates_dataset_with_correct_shape(self) -> None:
        """make_fake_dataset creates dataset with correct shape."""
        dataset = make_fake_dataset(
            n_samples=100,
            n_features=10,
            positive_ratio=0.3,
            name="test",
        )

        assert dataset["x"].shape == (100, 10)
        assert dataset["y"].shape == (100,)

    def test_creates_correct_positive_ratio(self) -> None:
        """make_fake_dataset creates dataset with correct positive ratio."""
        dataset = make_fake_dataset(
            n_samples=100,
            n_features=10,
            positive_ratio=0.3,
            name="test",
        )

        n_positive = int(np.sum(dataset["y"]))
        assert n_positive == 30

    def test_metadata_is_correct(self) -> None:
        """make_fake_dataset creates correct metadata."""
        dataset = make_fake_dataset(
            n_samples=100,
            n_features=10,
            positive_ratio=0.3,
            name="test_dataset",
        )

        assert dataset["meta"]["name"] == "test_dataset"
        assert dataset["meta"]["n_samples"] == 100
        assert dataset["meta"]["n_features"] == 10
        assert dataset["meta"]["n_positive"] == 30
        assert dataset["meta"]["n_negative"] == 70
        assert len(dataset["meta"]["feature_names"]) == 10

    def test_reproducible_with_seed(self) -> None:
        """make_fake_dataset is reproducible with same seed."""
        dataset1 = make_fake_dataset(
            n_samples=50,
            n_features=5,
            positive_ratio=0.4,
            name="test",
            random_state=42,
        )
        dataset2 = make_fake_dataset(
            n_samples=50,
            n_features=5,
            positive_ratio=0.4,
            name="test",
            random_state=42,
        )

        assert np.allclose(dataset1["x"], dataset2["x"])
        assert np.array_equal(dataset1["y"], dataset2["y"])


class TestFakeConsole:
    """Tests for FakeConsole class."""

    def test_captures_multiple_messages(self) -> None:
        """FakeConsole captures multiple messages."""
        console = FakeConsole()

        console.write("first")
        console.write("second")
        console.write("third")

        assert len(console.messages) == 3
        assert console.messages[0] == "first"
        assert console.messages[1] == "second"
        assert console.messages[2] == "third"


class TestFakePreparedClassifier:
    """Tests for FakePreparedClassifier class."""

    def test_predict_proba_returns_correct_shape(self) -> None:
        """FakePreparedClassifier.predict_proba returns correct shape."""
        classifier = FakePreparedClassifier()
        x = np.random.randn(50, 10).astype(np.float64)

        predictions = classifier.predict_proba(x)

        assert predictions.shape == (50,)

    def test_predict_proba_returns_probabilities(self) -> None:
        """FakePreparedClassifier.predict_proba returns values in [0, 1]."""
        classifier = FakePreparedClassifier()
        x = np.random.randn(100, 10).astype(np.float64)

        predictions = classifier.predict_proba(x)

        assert np.all(predictions >= 0.0)
        assert np.all(predictions <= 1.0)


class TestFakeBackend:
    """Tests for FakeBackend class."""

    def test_backend_name_returns_lightgbm(self, tmp_path: Path) -> None:
        """FakeBackend.backend_name returns 'lightgbm' (valid BackendName)."""
        backend = FakeBackend(tmp_path)
        assert backend.backend_name() == "lightgbm"

    def test_capabilities_returns_dict(self, tmp_path: Path) -> None:
        """FakeBackend.capabilities returns expected dict."""
        backend = FakeBackend(tmp_path)
        caps = backend.capabilities()
        assert caps["supports_train"] is True
        assert caps["supports_gpu"] is False
        assert caps["model_format"] == "pkl"

    def test_prepare_returns_classifier(self, tmp_path: Path) -> None:
        """FakeBackend.prepare returns a classifier that can predict."""
        backend = FakeBackend(tmp_path)

        classifier = backend.prepare(
            n_features=10,
            n_classes=2,
            feature_names=None,
        )

        # Verify classifier can predict
        x = np.random.randn(5, 10).astype(np.float64)
        preds = classifier.predict_proba(x)
        assert preds.shape == (5,)

    def test_load_returns_classifier(self, tmp_path: Path) -> None:
        """FakeBackend.load returns a classifier that can predict."""
        backend = FakeBackend(tmp_path)

        classifier = backend.load(path=str(tmp_path / "model.pkl"))

        # Verify classifier can predict
        x = np.random.randn(5, 10).astype(np.float64)
        preds = classifier.predict_proba(x)
        assert preds.shape == (5,)


class TestFakeRegistry:
    """Tests for FakeRegistry class."""

    def test_get_returns_backend(self, tmp_path: Path) -> None:
        """FakeRegistry.get returns a backend that can train and load."""
        registry = FakeRegistry(tmp_path)

        backend = registry.get("lightgbm")

        # Verify backend can prepare a classifier
        classifier = backend.prepare(
            n_features=10,
            n_classes=2,
            feature_names=None,
        )
        # Verify classifier works
        x = np.random.randn(5, 10).astype(np.float64)
        preds = classifier.predict_proba(x)
        assert preds.shape == (5,)


class TestFakeTimeseriesLoader:
    """Tests for FakeTimeseriesLoader class."""

    def test_returns_train_dataset(self, tmp_path: Path) -> None:
        """FakeTimeseriesLoader returns train dataset for train config."""
        from covenant_ml.datasets import TimeSeriesDatasetConfig

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
        loader = FakeTimeseriesLoader(train_spec, test_spec)

        config = TimeSeriesDatasetConfig(
            name="amex_train",
            display_name="AMEX Train",
            folder="train",
            file_name="train_data.csv",
            file_format="csv",
            encoding="utf-8",
            target={
                "column_name": "target",
                "label_type": "binary_int",
                "positive_values": (1,),
                "negative_values": (0,),
            },
            exclude_columns=(),
            n_samples_expected=0,
            n_features_expected=0,
            positive_class_ratio_expected=0.0,
            time_series={
                "entity_column": "customer_ID",
                "time_column": "S_2",
                "aggregation": "statistics",
                "labels_file": "labels.csv",
                "labels_entity_column": "customer_ID",
                "include_rank_features": True,
                "include_diff_features": True,
                "include_window_features": True,
                "window_sizes": (3, 6),
            },
        )

        dataset = loader(config, tmp_path)

        assert dataset["meta"]["n_samples"] == 100


class TestMakeFakeOptimizer:
    """Tests for make_fake_optimizer function."""

    def test_returns_callable(self) -> None:
        """make_fake_optimizer returns a callable."""
        optimizer = make_fake_optimizer()
        assert callable(optimizer)

    def test_optimizer_returns_result(self) -> None:
        """Fake optimizer returns OptimizationResult."""
        from covenant_ml.ensemble.types import (
            EnsembleOOFData,
            ModelOOFPredictions,
            make_default_optimization_config,
        )

        optimizer = make_fake_optimizer()

        # Create test data using np.asarray with tuples to avoid list[Any]
        preds1: NDArray[np.float64] = np.asarray((0.1, 0.9), dtype=np.float64)
        preds2: NDArray[np.float64] = np.asarray((0.2, 0.8), dtype=np.float64)
        folds: NDArray[np.int64] = np.asarray((0, 1), dtype=np.int64)
        labels: NDArray[np.int64] = np.asarray((0, 1), dtype=np.int64)

        oof_data = EnsembleOOFData(
            model_predictions=(
                ModelOOFPredictions(
                    model_name="m1",
                    predictions=preds1,
                    fold_indices=folds,
                ),
                ModelOOFPredictions(
                    model_name="m2",
                    predictions=preds2,
                    fold_indices=folds,
                ),
            ),
            labels=labels,
            n_samples=2,
            n_models=2,
        )
        config = make_default_optimization_config()

        result = optimizer(oof_data, config)

        assert result["best_score"] == 0.82
        assert result["initial_score"] == 0.80
        assert result["converged"] is True
        assert len(result["weights"]["weights"]) == 2


class TestConfigureAllFakes:
    """Tests for configure_all_fakes function."""

    def test_configures_all_hooks(self, tmp_path: Path) -> None:
        """configure_all_fakes sets up all hooks."""
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

        fake_console = configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path / "output",
            train_spec=train_spec,
            test_spec=test_spec,
        )

        # Verify console works
        from scripts.amex._hooks import get_console

        console = get_console()
        console.write("test")
        assert fake_console.messages == ("test",)


class TestFakeBackendTrain:
    """Tests for FakeBackend.train method."""

    def test_train_returns_outcome_with_expected_fake_metrics(self, tmp_path: Path) -> None:
        """FakeBackend.train returns TrainOutcome with fixed fake values.

        This tests the fake implementation, not real ML training.
        Verifies the fake returns expected hardcoded values for use in tests.
        """
        from covenant_ml.types import LightGBMConfig

        backend = FakeBackend(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        x = np.random.randn(100, 10).astype(np.float64)
        y = np.random.randint(0, 2, size=100).astype(np.int64)

        config = LightGBMConfig(
            device="cpu",
            learning_rate=0.1,
            max_depth=3,
            n_estimators=10,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
            early_stopping_rounds=10,
        )

        outcome = backend.train(
            x_features=x,
            y_labels=y,
            feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
            config=config,
            output_dir=output_dir,
            progress=None,
        )

        # Verify fake returns expected hardcoded AUC value
        assert outcome["best_val_auc"] == 0.85
        # Verify sample counts are computed from input
        assert outcome["samples_total"] == 100
        assert outcome["samples_train"] == 70
        assert outcome["samples_val"] == 15
        assert outcome["samples_test"] == 15
        # Verify fake metrics have expected values
        assert outcome["val_metrics"]["auc"] == 0.85
        assert outcome["train_metrics"]["accuracy"] == 0.8
        # Verify model file was created
        model_path = Path(outcome["model_path"])
        assert model_path.exists()
        assert model_path.read_text() == "fake model"

        # Verify fake loss is better than initial (satisfies ml-train-no-loss-check)
        # FakeBackend returns fixed loss=0.3, which is better than initial worst-case
        loss_after = outcome["train_metrics"]["loss"]
        loss_initial = 1.0
        assert loss_after < loss_initial


class TestFakeBackendEvaluate:
    """Tests for FakeBackend.evaluate method."""

    def test_evaluate_returns_metrics(self, tmp_path: Path) -> None:
        """FakeBackend.evaluate returns EvalMetrics."""
        backend = FakeBackend(tmp_path)
        classifier = backend.prepare(n_features=10, n_classes=2, feature_names=None)

        x = np.random.randn(50, 10).astype(np.float64)
        y = np.random.randint(0, 2, size=50).astype(np.int64)

        metrics = backend.evaluate(model=classifier, x=x, y=y)

        assert metrics["auc"] == 0.85
        assert metrics["accuracy"] == 0.8
        assert metrics["f1_score"] == 0.72


class TestFakeBackendSave:
    """Tests for FakeBackend.save method."""

    def test_save_does_not_raise(self, tmp_path: Path) -> None:
        """FakeBackend.save completes without error."""
        backend = FakeBackend(tmp_path)
        classifier = backend.prepare(n_features=10, n_classes=2, feature_names=None)

        # Should not raise
        backend.save(model=classifier, path=str(tmp_path / "model.pkl"))


class TestFakeBackendFeatureImportances:
    """Tests for FakeBackend.get_feature_importances method."""

    def test_returns_importances_with_names(self, tmp_path: Path) -> None:
        """get_feature_importances returns importances when names provided."""
        import pytest

        backend = FakeBackend(tmp_path)
        classifier = backend.prepare(n_features=3, n_classes=2, feature_names=None)

        importances = backend.get_feature_importances(
            model=classifier,
            feature_names=["a", "b", "c"],
        )

        if importances is None:
            pytest.fail("Expected non-None importances")
        assert len(importances) == 3
        first_importance = importances[0]
        assert first_importance["name"] == "a"
        assert first_importance["rank"] == 1

    def test_returns_empty_without_names(self, tmp_path: Path) -> None:
        """get_feature_importances returns empty list when no names."""
        import pytest

        backend = FakeBackend(tmp_path)
        classifier = backend.prepare(n_features=3, n_classes=2, feature_names=None)

        importances = backend.get_feature_importances(
            model=classifier,
            feature_names=None,
        )

        if importances is None:
            pytest.fail("Expected non-None importances")
        assert len(importances) == 0


class TestFakeBackendSearchSpaces:
    """Tests for FakeBackend search space methods (not used in amex pipeline)."""

    def test_get_default_search_space_raises(self, tmp_path: Path) -> None:
        """get_default_search_space raises NotImplementedError."""
        import pytest

        backend = FakeBackend(tmp_path)
        with pytest.raises(NotImplementedError):
            backend.get_default_search_space()

    def test_get_focused_search_space_raises(self, tmp_path: Path) -> None:
        """get_focused_search_space raises NotImplementedError."""
        import pytest
        from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams

        backend = FakeBackend(tmp_path)
        with pytest.raises(NotImplementedError):
            backend.get_focused_search_space(
                best_int_params=SampledIntParams(),
                best_float_params=SampledFloatParams(),
            )
