"""Tests for regression dataset types.

Verifies construction and immutability of RegressionTargetSpec,
RegressionDatasetConfig, and RegressionDatasetMeta TypedDicts.
"""

from __future__ import annotations

from covenant_ml.datasets.types import (
    RegressionDatasetConfig,
    RegressionDatasetMeta,
    RegressionTargetSpec,
)


class TestRegressionTargetSpec:
    """Tests for RegressionTargetSpec TypedDict."""

    def test_construction(self) -> None:
        """RegressionTargetSpec stores column_name."""
        spec = RegressionTargetSpec(column_name="Financial Distress")
        assert spec["column_name"] == "Financial Distress"

    def test_different_column_name(self) -> None:
        """RegressionTargetSpec works with arbitrary column names."""
        spec = RegressionTargetSpec(column_name="price")
        assert spec["column_name"] == "price"


class TestRegressionDatasetConfig:
    """Tests for RegressionDatasetConfig TypedDict."""

    def test_construction(self) -> None:
        """RegressionDatasetConfig stores all required fields."""
        config = RegressionDatasetConfig(
            name="test_regression",
            display_name="Test Regression Dataset",
            folder="test_folder",
            file_name="data.csv",
            file_format="csv",
            encoding="utf-8",
            target=RegressionTargetSpec(column_name="target"),
            exclude_columns=("id",),
            n_samples_expected=100,
            n_features_expected=10,
            target_mean_expected=5.0,
        )

        assert config["name"] == "test_regression"
        assert config["display_name"] == "Test Regression Dataset"
        assert config["folder"] == "test_folder"
        assert config["file_name"] == "data.csv"
        assert config["file_format"] == "csv"
        assert config["encoding"] == "utf-8"
        assert config["target"]["column_name"] == "target"
        assert config["exclude_columns"] == ("id",)
        assert config["n_samples_expected"] == 100
        assert config["n_features_expected"] == 10
        assert config["target_mean_expected"] == 5.0

    def test_empty_exclude_columns(self) -> None:
        """RegressionDatasetConfig works with no excluded columns."""
        config = RegressionDatasetConfig(
            name="minimal",
            display_name="Minimal",
            folder="min",
            file_name="data.csv",
            file_format="csv",
            encoding="utf-8",
            target=RegressionTargetSpec(column_name="y"),
            exclude_columns=(),
            n_samples_expected=10,
            n_features_expected=3,
            target_mean_expected=0.0,
        )
        assert config["exclude_columns"] == ()


class TestRegressionDatasetMeta:
    """Tests for RegressionDatasetMeta TypedDict."""

    def test_construction(self) -> None:
        """RegressionDatasetMeta stores all target distribution fields."""
        meta = RegressionDatasetMeta(
            name="test",
            n_samples=100,
            n_features=10,
            target_mean=5.5,
            target_std=2.1,
            target_min=-1.0,
            target_max=12.3,
            feature_names=("f0", "f1"),
            categorical_encodings=(),
        )

        assert meta["name"] == "test"
        assert meta["n_samples"] == 100
        assert meta["n_features"] == 10
        assert meta["target_mean"] == 5.5
        assert meta["target_std"] == 2.1
        assert meta["target_min"] == -1.0
        assert meta["target_max"] == 12.3
        assert meta["feature_names"] == ("f0", "f1")
        assert meta["categorical_encodings"] == ()
