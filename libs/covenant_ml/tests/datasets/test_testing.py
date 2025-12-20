"""Tests for FakeDatasetLoader in testing module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from covenant_ml.datasets.testing import (
    FakeDatasetLoader,
    create_fake_dataset_loader,
)
from covenant_ml.datasets.types import DatasetConfig, TargetColumnSpec


def _make_test_config(name: str = "test") -> DatasetConfig:
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


class TestFakeDatasetLoader:
    """Tests for FakeDatasetLoader class."""

    def test_init_with_defaults(self) -> None:
        """FakeDatasetLoader initializes with default parameters."""
        loader = FakeDatasetLoader()
        config = _make_test_config()
        result = loader.load(config, Path("/fake"))

        assert result["meta"]["n_samples"] == 100
        assert result["meta"]["n_features"] == 10
        assert result["meta"]["positive_ratio"] == 0.3

    def test_init_with_custom_params(self) -> None:
        """FakeDatasetLoader respects custom parameters."""
        loader = FakeDatasetLoader(
            n_samples=50,
            n_features=5,
            positive_ratio=0.2,
            random_state=123,
        )
        config = _make_test_config()
        result = loader.load(config, Path("/fake"))

        assert result["meta"]["n_samples"] == 50
        assert result["meta"]["n_features"] == 5
        assert result["meta"]["positive_ratio"] == 0.2

    def test_load_returns_correct_shapes(self) -> None:
        """Load returns arrays with correct shapes."""
        loader = FakeDatasetLoader(n_samples=100, n_features=10)
        config = _make_test_config()
        result = loader.load(config, Path("/fake"))

        assert result["x"].shape == (100, 10)
        assert result["y"].shape == (100,)

    def test_load_returns_correct_dtypes(self) -> None:
        """Load returns arrays with correct dtypes."""
        loader = FakeDatasetLoader()
        config = _make_test_config()
        result = loader.load(config, Path("/fake"))

        assert result["x"].dtype == np.float64
        assert result["y"].dtype == np.int64

    def test_load_uses_config_name(self) -> None:
        """Load uses name from config in metadata."""
        loader = FakeDatasetLoader()
        config = _make_test_config("my_dataset")
        result = loader.load(config, Path("/fake"))

        assert result["meta"]["name"] == "my_dataset"

    def test_load_generates_feature_names(self) -> None:
        """Load generates feature names based on n_features."""
        loader = FakeDatasetLoader(n_features=5)
        config = _make_test_config()
        result = loader.load(config, Path("/fake"))

        assert result["meta"]["feature_names"] == (
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
        )

    def test_load_respects_positive_ratio(self) -> None:
        """Load generates labels with approximately correct positive ratio."""
        loader = FakeDatasetLoader(
            n_samples=100,
            positive_ratio=0.3,
            random_state=42,
        )
        config = _make_test_config()
        result = loader.load(config, Path("/fake"))

        n_positive = int(np.sum(result["y"]))
        assert n_positive == 30  # Exactly 30% of 100

    def test_load_computes_n_positive_n_negative(self) -> None:
        """Load correctly computes n_positive and n_negative in metadata."""
        loader = FakeDatasetLoader(
            n_samples=100,
            positive_ratio=0.4,
        )
        config = _make_test_config()
        result = loader.load(config, Path("/fake"))

        assert result["meta"]["n_positive"] == 40
        assert result["meta"]["n_negative"] == 60

    def test_load_deterministic_with_same_seed(self) -> None:
        """Load produces identical results with same random_state."""
        loader1 = FakeDatasetLoader(random_state=42)
        loader2 = FakeDatasetLoader(random_state=42)
        config = _make_test_config()

        result1 = loader1.load(config, Path("/fake"))
        result2 = loader2.load(config, Path("/fake"))

        np.testing.assert_array_equal(result1["x"], result2["x"])
        np.testing.assert_array_equal(result1["y"], result2["y"])

    def test_load_different_with_different_seed(self) -> None:
        """Load produces different results with different random_state."""
        loader1 = FakeDatasetLoader(random_state=42)
        loader2 = FakeDatasetLoader(random_state=99)
        config = _make_test_config()

        result1 = loader1.load(config, Path("/fake"))
        result2 = loader2.load(config, Path("/fake"))

        # Arrays should be different
        assert not np.allclose(result1["x"], result2["x"])


class TestCreateFakeDatasetLoader:
    """Tests for create_fake_dataset_loader factory."""

    def test_creates_loader_with_defaults(self) -> None:
        """Factory creates loader with default parameters."""
        loader = create_fake_dataset_loader()
        config = _make_test_config()
        result = loader.load(config, Path("/fake"))

        assert result["meta"]["n_samples"] == 100
        assert result["meta"]["n_features"] == 10

    def test_creates_loader_with_custom_params(self) -> None:
        """Factory creates loader with custom parameters."""
        loader = create_fake_dataset_loader(
            n_samples=200,
            n_features=20,
            positive_ratio=0.5,
            random_state=99,
        )
        config = _make_test_config()
        result = loader.load(config, Path("/fake"))

        assert result["meta"]["n_samples"] == 200
        assert result["meta"]["n_features"] == 20
        assert result["meta"]["positive_ratio"] == 0.5
