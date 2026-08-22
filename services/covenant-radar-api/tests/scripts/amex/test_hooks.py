"""Tests for AMEX pipeline hooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scripts.amex._hook_protocols import (
    FakeDatasetSpec,
)
from scripts.amex._hooks import (
    ConsoleProtocol,
    _default_console_factory,
    _default_project_root,
    _RichConsoleAdapter,
    get_console,
    get_ensemble_optimizer,
    get_project_root,
    get_registry,
    get_timeseries_loader,
)
from scripts.amex._test_fakes import (
    FakeConsole,
)
from scripts.amex._test_hooks import (
    configure_fake_console,
    configure_fake_ensemble_optimizer,
    configure_fake_project_root,
    configure_fake_registry,
    configure_fake_timeseries_loader,
)


class TestConsoleHook:
    """Tests for console hook."""

    def test_get_console_returns_protocol(self) -> None:
        """get_console returns a ConsoleProtocol implementation."""
        configure_fake_console()
        console: ConsoleProtocol = get_console()
        # Verify write method works by calling it
        console.write("test message")

    def test_fake_console_captures_messages(self) -> None:
        """Fake console captures written messages."""
        fake_console = configure_fake_console()
        console = get_console()

        console.write("test message 1")
        console.write("test message 2")

        assert fake_console.messages == ("test message 1", "test message 2")


class TestProjectRootHook:
    """Tests for project root hook."""

    def test_get_project_root_returns_path(self) -> None:
        """get_project_root returns a Path."""
        fake_path = Path("/fake/project/root")
        configure_fake_project_root(fake_path)

        result = get_project_root()
        assert result == fake_path


class TestRegistryHook:
    """Tests for registry hook."""

    def test_get_registry_returns_registry(self, tmp_path: Path) -> None:
        """get_registry returns a registry with get method."""
        configure_fake_registry(tmp_path)
        registry = get_registry()

        # Should be able to get a backend
        backend = registry.get("lightgbm")

        # Verify backend methods work
        n_features = 10
        n_classes = 2
        classifier = backend.prepare(
            n_features=n_features,
            n_classes=n_classes,
            feature_names=None,
        )
        # Verify classifier can predict

        x = np.random.randn(5, 10).astype(np.float64)
        preds = classifier.predict_proba(x)
        assert preds.shape == (5,)


class TestTimeseriesLoaderHook:
    """Tests for time-series loader hook."""

    def test_get_timeseries_loader_returns_callable(self, tmp_path: Path) -> None:
        """get_timeseries_loader returns a callable."""
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
        configure_fake_timeseries_loader(train_spec, test_spec)

        loader = get_timeseries_loader()
        assert callable(loader)


class TestEnsembleOptimizerHook:
    """Tests for ensemble optimizer hook."""

    def test_get_ensemble_optimizer_returns_callable(self) -> None:
        """get_ensemble_optimizer returns a callable."""
        configure_fake_ensemble_optimizer()
        optimizer = get_ensemble_optimizer()
        assert callable(optimizer)


class TestFakeConsole:
    """Tests for FakeConsole."""

    def test_write_captures_messages(self) -> None:
        """FakeConsole.write captures messages."""
        console = FakeConsole()

        console.write("message 1")
        console.write("message 2")

        assert console.messages == ("message 1", "message 2")

    def test_empty_messages_initially(self) -> None:
        """FakeConsole starts with empty messages."""
        console = FakeConsole()
        assert console.messages == ()


class TestFakeTimeseriesLoader:
    """Tests for FakeTimeseriesLoader."""

    def test_returns_train_dataset(self, tmp_path: Path) -> None:
        """FakeTimeseriesLoader returns training dataset for train config."""
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
        configure_fake_timeseries_loader(train_spec, test_spec)

        loader = get_timeseries_loader()

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
        assert dataset["meta"]["n_features"] == 10

    def test_returns_test_dataset(self, tmp_path: Path) -> None:
        """FakeTimeseriesLoader returns test dataset for test config."""
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
        configure_fake_timeseries_loader(train_spec, test_spec)

        loader = get_timeseries_loader()

        config = TimeSeriesDatasetConfig(
            name="amex_test",
            display_name="AMEX Test",
            folder="test",
            file_name="test_data.csv",
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
                "labels_file": "",
                "labels_entity_column": "customer_ID",
                "include_rank_features": True,
                "include_diff_features": True,
                "include_window_features": True,
                "window_sizes": (3, 6),
            },
        )

        dataset = loader(config, tmp_path)

        assert dataset["meta"]["n_samples"] == 50
        assert dataset["meta"]["n_features"] == 10


class TestRealImplementations:
    """Tests for real hook implementations.

    These test the default/production implementations that are normally
    replaced by fakes in tests.
    """

    def test_default_project_root_returns_path(self) -> None:
        """_default_project_root returns a valid path."""
        result = _default_project_root()

        # Verify it's the covenant-radar-api root by checking name
        assert result.name == "covenant-radar-api"
        # Verify path structure is correct (has parent)
        assert result.parent.exists()

    def test_default_console_factory_returns_adapter(self) -> None:
        """_default_console_factory returns a RichConsoleAdapter."""
        result = _default_console_factory()

        # Verify it has the write method expected by ConsoleProtocol
        assert callable(result.write)

    def test_rich_console_adapter_write_is_callable(self) -> None:
        """_RichConsoleAdapter.write is callable."""
        adapter = _RichConsoleAdapter()

        # Verify write method exists and is callable
        assert callable(adapter.write)

    def test_rich_console_adapter_write_executes(self) -> None:
        """_RichConsoleAdapter.write executes without error.

        This exercises the real Rich console output path (lines 55-59).
        """
        from platform_core.rich_logging import setup_rich_logging

        # Setup rich logging before calling write (required by platform_core)
        setup_rich_logging()

        adapter = _RichConsoleAdapter()

        # Actually call write to cover the implementation
        adapter.write("test message from unit test")

    def test_default_registry_factory_returns_registry(self) -> None:
        """_default_registry_factory returns a ClassifierRegistry."""
        from scripts.amex._hooks import _default_registry_factory

        registry = _default_registry_factory()

        # Verify get method works by calling it
        backend = registry.get("lightgbm")
        # Should return a backend with prepare method
        assert callable(backend.prepare)

    def test_real_timeseries_loader_is_callable(self) -> None:
        """_real_timeseries_loader is callable."""
        from scripts.amex._hooks import _real_timeseries_loader

        assert callable(_real_timeseries_loader)

    def test_real_timeseries_loader_loads_data(self, tmp_path: Path) -> None:
        """_real_timeseries_loader loads actual time-series CSV data.

        This exercises lines 237-238 by calling the real loader with fixture data.
        """
        from covenant_ml.datasets import TimeSeriesDatasetConfig
        from scripts.amex._hooks import _real_timeseries_loader

        # Create test data directory structure
        dataset_folder = tmp_path / "test_ts_dataset"
        dataset_folder.mkdir(parents=True, exist_ok=True)

        # Create a time-series CSV file with multiple observations per entity
        train_csv = dataset_folder / "train.csv"
        train_csv.write_text(
            "customer_ID,S_2,feature_1,feature_2\n"
            "C_001,2023-01-01,1.0,2.0\n"
            "C_001,2023-01-02,1.5,2.5\n"
            "C_001,2023-01-03,2.0,3.0\n"
            "C_002,2023-01-01,0.5,1.0\n"
            "C_002,2023-01-02,0.6,1.1\n"
        )

        # Create labels file
        labels_csv = dataset_folder / "labels.csv"
        labels_csv.write_text("customer_ID,target\nC_001,1\nC_002,0\n")

        # Create config for loading
        config = TimeSeriesDatasetConfig(
            name="test_ts_dataset_train",
            display_name="Test Time Series",
            folder="test_ts_dataset",
            file_name="train.csv",
            file_format="csv",
            encoding="utf-8",
            target={
                "column_name": "target",
                "label_type": "binary_int",
                "positive_values": (1,),
                "negative_values": (0,),
            },
            exclude_columns=(),
            n_samples_expected=0,  # Skip validation
            n_features_expected=0,  # Skip validation
            positive_class_ratio_expected=0.0,  # Skip validation
            time_series={
                "entity_column": "customer_ID",
                "time_column": "S_2",
                "aggregation": "last",
                "labels_file": "labels.csv",
                "labels_entity_column": "customer_ID",
                "include_rank_features": False,
                "include_diff_features": False,
                "include_window_features": False,
                "window_sizes": (),
            },
        )

        # Call real loader - this exercises lines 237-238
        dataset = _real_timeseries_loader(config, tmp_path)

        # Verify dataset was loaded
        assert dataset["meta"]["n_samples"] == 2  # 2 unique customers
        assert dataset["meta"]["n_features"] == 2  # feature_1, feature_2
        assert dataset["x"].shape == (2, 2)
        assert dataset["y"].shape == (2,)

    def test_real_ensemble_optimizer_is_callable(self) -> None:
        """_real_ensemble_optimizer is callable."""
        from scripts.amex._hooks import _real_ensemble_optimizer

        assert callable(_real_ensemble_optimizer)

    def test_real_ensemble_optimizer_returns_result(self) -> None:
        """_real_ensemble_optimizer returns OptimizationResult."""
        from covenant_ml.ensemble.types import (
            EnsembleOOFData,
            ModelOOFPredictions,
            make_default_optimization_config,
        )
        from scripts.amex._hooks import _real_ensemble_optimizer, restore_real_minimize

        # Ensure the real solver is in place after any earlier fake
        restore_real_minimize()

        # Create test data using tuples to avoid list[Any]
        preds1 = np.asarray((0.1, 0.9, 0.2, 0.8), dtype=np.float64)
        preds2 = np.asarray((0.2, 0.8, 0.3, 0.7), dtype=np.float64)
        folds = np.asarray((0, 0, 1, 1), dtype=np.int64)
        labels = np.asarray((0, 1, 0, 1), dtype=np.int64)

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
            n_samples=4,
            n_models=2,
        )
        config = make_default_optimization_config()

        result = _real_ensemble_optimizer(oof_data, config)

        assert len(result["weights"]["weights"]) == 2
        assert result["converged"] is True
