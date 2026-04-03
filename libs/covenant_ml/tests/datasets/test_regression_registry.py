"""Tests for RegressionDatasetRegistry.

Covers construction, registration, lookup, and the default registry
with the Financial Distress dataset.
"""

from __future__ import annotations

import pytest

from covenant_ml.datasets.registry import (
    RegressionDatasetRegistry,
    make_default_regression_registry,
)
from covenant_ml.datasets.types import (
    RegressionDatasetConfig,
    RegressionTargetSpec,
)


def _make_test_config(name: str) -> RegressionDatasetConfig:
    """Create a test regression dataset config.

    Args:
        name: Dataset name.

    Returns:
        RegressionDatasetConfig for testing.
    """
    return RegressionDatasetConfig(
        name=name,
        display_name=f"Test {name}",
        folder=f"{name}_folder",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=RegressionTargetSpec(column_name="target"),
        exclude_columns=(),
        n_samples_expected=100,
        n_features_expected=10,
        target_mean_expected=0.0,
    )


class TestRegressionDatasetRegistry:
    """Tests for RegressionDatasetRegistry class."""

    def test_registry_init_empty(self) -> None:
        """Empty registry has no configs."""
        registry = RegressionDatasetRegistry(())
        assert len(registry) == 0
        assert registry.list_names() == ()

    def test_registry_init_single_config(self) -> None:
        """Registry with single config works correctly."""
        config = _make_test_config("test_regression")
        registry = RegressionDatasetRegistry((config,))

        assert len(registry) == 1
        assert "test_regression" in registry
        assert registry.list_names() == ("test_regression",)

    def test_registry_get_returns_config(self) -> None:
        """get() returns the registered config."""
        config = _make_test_config("my_dataset")
        registry = RegressionDatasetRegistry((config,))

        result = registry.get("my_dataset")
        assert result["name"] == "my_dataset"
        assert result["target"]["column_name"] == "target"

    def test_registry_get_raises_on_missing(self) -> None:
        """get() raises KeyError for unregistered dataset."""
        registry = RegressionDatasetRegistry(())
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_registry_contains(self) -> None:
        """__contains__ returns True for registered, False for missing."""
        config = _make_test_config("present")
        registry = RegressionDatasetRegistry((config,))

        assert "present" in registry
        assert "absent" not in registry

    def test_registry_len(self) -> None:
        """__len__ returns correct count."""
        configs = (_make_test_config("a"), _make_test_config("b"), _make_test_config("c"))
        registry = RegressionDatasetRegistry(configs)
        assert len(registry) == 3

    def test_registry_list_names_sorted(self) -> None:
        """list_names() returns alphabetically sorted tuple."""
        configs = (_make_test_config("zebra"), _make_test_config("alpha"), _make_test_config("mid"))
        registry = RegressionDatasetRegistry(configs)
        assert registry.list_names() == ("alpha", "mid", "zebra")

    def test_registry_duplicate_name_raises(self) -> None:
        """Duplicate dataset names raise ValueError."""
        config_a = _make_test_config("dup")
        config_b = _make_test_config("dup")
        with pytest.raises(ValueError, match="Duplicate"):
            RegressionDatasetRegistry((config_a, config_b))


class TestDefaultRegressionRegistry:
    """Tests for make_default_regression_registry()."""

    def test_has_financial_distress(self) -> None:
        """Default registry includes financial_distress."""
        registry = make_default_regression_registry()
        assert "financial_distress" in registry

    def test_financial_distress_config(self) -> None:
        """Financial distress config has correct metadata."""
        registry = make_default_regression_registry()
        config = registry.get("financial_distress")

        assert config["display_name"] == "Financial Distress"
        assert config["folder"] == "kaggle_financial_distress"
        assert config["file_name"] == "Financial Distress.csv"
        assert config["file_format"] == "csv"
        assert config["encoding"] == "utf-8"
        assert config["target"]["column_name"] == "Financial Distress"
        assert config["exclude_columns"] == ("Company", "Time")
        assert config["n_samples_expected"] == 3672
        assert config["n_features_expected"] == 83
