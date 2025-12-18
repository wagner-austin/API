"""Tests for DatasetRegistry and TimeSeriesDatasetRegistry."""

from __future__ import annotations

import pytest

from covenant_ml.datasets.registry import (
    DatasetRegistry,
    TimeSeriesDatasetRegistry,
    make_default_registry,
    make_default_timeseries_registry,
)
from covenant_ml.datasets.types import (
    DatasetConfig,
    TargetColumnSpec,
    TimeSeriesDatasetConfig,
    TimeSeriesSpec,
)


def _make_test_config(name: str) -> DatasetConfig:
    """Create a test dataset config."""
    return DatasetConfig(
        name=name,
        display_name=f"Test {name}",
        folder=f"{name}_folder",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=100,
        n_features_expected=10,
        positive_class_ratio_expected=0.1,
    )


class TestDatasetRegistry:
    """Tests for DatasetRegistry class."""

    def test_registry_init_empty(self) -> None:
        """Empty registry has no configs."""
        registry = DatasetRegistry(())
        assert len(registry) == 0
        assert registry.list_names() == ()

    def test_registry_init_single_config(self) -> None:
        """Registry with single config works correctly."""
        config = _make_test_config("test_dataset")
        registry = DatasetRegistry((config,))

        assert len(registry) == 1
        assert "test_dataset" in registry
        assert registry.list_names() == ("test_dataset",)

    def test_registry_init_multiple_configs(self) -> None:
        """Registry with multiple configs works correctly."""
        config1 = _make_test_config("alpha")
        config2 = _make_test_config("beta")
        config3 = _make_test_config("gamma")

        registry = DatasetRegistry((config1, config2, config3))

        assert len(registry) == 3
        assert "alpha" in registry
        assert "beta" in registry
        assert "gamma" in registry
        assert registry.list_names() == ("alpha", "beta", "gamma")

    def test_registry_init_duplicate_raises(self) -> None:
        """Duplicate dataset names raise ValueError."""
        config1 = _make_test_config("duplicate")
        config2 = _make_test_config("duplicate")

        with pytest.raises(ValueError, match="Duplicate dataset name: duplicate"):
            DatasetRegistry((config1, config2))

    def test_registry_get_existing(self) -> None:
        """Get returns config for existing dataset."""
        config = _make_test_config("existing")
        registry = DatasetRegistry((config,))

        result = registry.get("existing")

        assert result["name"] == "existing"
        assert result["display_name"] == "Test existing"
        assert result["folder"] == "existing_folder"

    def test_registry_get_missing_raises(self) -> None:
        """Get raises KeyError for missing dataset."""
        config = _make_test_config("only_one")
        registry = DatasetRegistry((config,))

        with pytest.raises(KeyError, match="Dataset 'missing' not found"):
            registry.get("missing")

    def test_registry_get_missing_shows_available(self) -> None:
        """KeyError message shows available datasets."""
        config1 = _make_test_config("alpha")
        config2 = _make_test_config("beta")
        registry = DatasetRegistry((config1, config2))

        with pytest.raises(KeyError, match="Available: alpha, beta"):
            registry.get("gamma")

    def test_registry_contains_true(self) -> None:
        """Contains returns True for existing dataset."""
        config = _make_test_config("exists")
        registry = DatasetRegistry((config,))

        assert "exists" in registry

    def test_registry_contains_false(self) -> None:
        """Contains returns False for missing dataset."""
        config = _make_test_config("exists")
        registry = DatasetRegistry((config,))

        assert "missing" not in registry

    def test_registry_list_names_sorted(self) -> None:
        """List names returns sorted tuple."""
        config1 = _make_test_config("zebra")
        config2 = _make_test_config("alpha")
        config3 = _make_test_config("middle")

        registry = DatasetRegistry((config1, config2, config3))

        assert registry.list_names() == ("alpha", "middle", "zebra")


class TestMakeDefaultRegistry:
    """Tests for make_default_registry factory."""

    def test_default_registry_not_empty(self) -> None:
        """Default registry has verified configs."""
        registry = make_default_registry()

        # Registry must have at least taiwan, us, polish datasets
        assert len(registry) >= 3

    def test_default_registry_has_taiwan(self) -> None:
        """Default registry includes Taiwan dataset."""
        registry = make_default_registry()

        assert "taiwan" in registry
        config = registry.get("taiwan")
        assert config["display_name"] == "Taiwan Bankruptcy (Original)"
        assert config["file_format"] == "csv"

    def test_default_registry_has_us(self) -> None:
        """Default registry includes US dataset."""
        registry = make_default_registry()

        assert "us" in registry
        config = registry.get("us")
        assert config["display_name"] == "US Bankruptcy (Original)"
        assert config["encoding"] == "utf-8-sig"

    def test_default_registry_has_polish(self) -> None:
        """Default registry includes Polish dataset."""
        registry = make_default_registry()

        assert "polish" in registry
        config = registry.get("polish")
        assert config["display_name"] == "Polish Bankruptcy (Original)"
        assert config["file_format"] == "arff"

    def test_default_registry_configs_have_required_fields(self) -> None:
        """All configs in default registry have required fields."""
        registry = make_default_registry()

        for name in registry.list_names():
            config = registry.get(name)
            assert config["name"] == name
            # Check string fields are non-empty with minimum lengths
            assert len(config["display_name"]) >= 3, f"{name}: display_name too short"
            assert len(config["folder"]) >= 2, f"{name}: folder too short"
            assert len(config["file_name"]) >= 4, f"{name}: file_name too short"
            assert config["file_format"] in ("csv", "arff", "excel")
            assert config["encoding"] in ("utf-8", "utf-8-sig", "latin-1", "cp1252")
            assert config["n_samples_expected"] > 0
            assert config["n_features_expected"] > 0
            assert 0.0 <= config["positive_class_ratio_expected"] <= 1.0


def _make_test_timeseries_config(name: str) -> TimeSeriesDatasetConfig:
    """Create a test time-series dataset config."""
    return TimeSeriesDatasetConfig(
        name=name,
        display_name=f"Test {name}",
        folder=f"{name}_folder",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="target",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=100,
        n_features_expected=10,
        positive_class_ratio_expected=0.1,
        time_series=TimeSeriesSpec(
            entity_column="entity_id",
            time_column="timestamp",
            aggregation="last",
            labels_file="labels.csv",
            labels_entity_column="entity_id",
        ),
    )


class TestTimeSeriesDatasetRegistry:
    """Tests for TimeSeriesDatasetRegistry class."""

    def test_registry_init_empty(self) -> None:
        """Empty registry has no configs."""
        registry = TimeSeriesDatasetRegistry(())
        assert len(registry) == 0
        assert registry.list_names() == ()

    def test_registry_init_single_config(self) -> None:
        """Registry with single config works correctly."""
        config = _make_test_timeseries_config("test_ts")
        registry = TimeSeriesDatasetRegistry((config,))

        assert len(registry) == 1
        assert "test_ts" in registry
        assert registry.list_names() == ("test_ts",)

    def test_registry_init_multiple_configs(self) -> None:
        """Registry with multiple configs works correctly."""
        config1 = _make_test_timeseries_config("alpha_ts")
        config2 = _make_test_timeseries_config("beta_ts")

        registry = TimeSeriesDatasetRegistry((config1, config2))

        assert len(registry) == 2
        assert "alpha_ts" in registry
        assert "beta_ts" in registry
        assert registry.list_names() == ("alpha_ts", "beta_ts")

    def test_registry_init_duplicate_raises(self) -> None:
        """Duplicate dataset names raise ValueError."""
        config1 = _make_test_timeseries_config("duplicate")
        config2 = _make_test_timeseries_config("duplicate")

        with pytest.raises(ValueError, match="Duplicate dataset name: duplicate"):
            TimeSeriesDatasetRegistry((config1, config2))

    def test_registry_get_existing(self) -> None:
        """Get returns config for existing dataset."""
        config = _make_test_timeseries_config("existing_ts")
        registry = TimeSeriesDatasetRegistry((config,))

        result = registry.get("existing_ts")

        assert result["name"] == "existing_ts"
        assert result["display_name"] == "Test existing_ts"
        assert result["time_series"]["entity_column"] == "entity_id"
        assert result["time_series"]["aggregation"] == "last"

    def test_registry_get_missing_raises(self) -> None:
        """Get raises KeyError for missing dataset."""
        config = _make_test_timeseries_config("only_one")
        registry = TimeSeriesDatasetRegistry((config,))

        with pytest.raises(KeyError, match="Time-series dataset 'missing' not found"):
            registry.get("missing")

    def test_registry_get_missing_shows_available(self) -> None:
        """KeyError message shows available datasets."""
        config1 = _make_test_timeseries_config("alpha_ts")
        config2 = _make_test_timeseries_config("beta_ts")
        registry = TimeSeriesDatasetRegistry((config1, config2))

        with pytest.raises(KeyError, match="Available: alpha_ts, beta_ts"):
            registry.get("gamma_ts")

    def test_registry_contains_true(self) -> None:
        """Contains returns True for existing dataset."""
        config = _make_test_timeseries_config("exists_ts")
        registry = TimeSeriesDatasetRegistry((config,))

        assert "exists_ts" in registry

    def test_registry_contains_false(self) -> None:
        """Contains returns False for missing dataset."""
        config = _make_test_timeseries_config("exists_ts")
        registry = TimeSeriesDatasetRegistry((config,))

        assert "missing" not in registry

    def test_registry_list_names_sorted(self) -> None:
        """List names returns sorted tuple."""
        config1 = _make_test_timeseries_config("zebra_ts")
        config2 = _make_test_timeseries_config("alpha_ts")

        registry = TimeSeriesDatasetRegistry((config1, config2))

        assert registry.list_names() == ("alpha_ts", "zebra_ts")


class TestMakeDefaultTimeseriesRegistry:
    """Tests for make_default_timeseries_registry factory."""

    def test_default_timeseries_registry_empty(self) -> None:
        """Default time-series registry starts empty (configs added as verified)."""
        registry = make_default_timeseries_registry()

        # Initially empty until we verify time-series datasets
        assert len(registry) == 0
        assert registry.list_names() == ()
