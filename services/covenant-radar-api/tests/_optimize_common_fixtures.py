"""Shared fixtures and helpers for test_optimize_common splits."""

from __future__ import annotations

import numpy as np
from covenant_ml.datasets import (
    DatasetConfig,
    DatasetMeta,
    DatasetRegistry,
    LoadedDataset,
    TimeSeriesDatasetConfig,
    TimeSeriesDatasetRegistry,
)
from numpy.typing import NDArray


def _make_fake_standard_dataset(name: str = "taiwan") -> LoadedDataset:
    """Create fake standard dataset for testing.

    Args:
        name: Dataset name.

    Returns:
        LoadedDataset with synthetic data.
    """
    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((100, 10)).astype(np.float64)
    y: NDArray[np.int64] = rng.integers(0, 2, size=100).astype(np.int64)
    n_positive = int(np.sum(y))
    meta: DatasetMeta = {
        "name": name,
        "n_samples": 100,
        "n_features": 10,
        "n_positive": n_positive,
        "n_negative": 100 - n_positive,
        "positive_ratio": n_positive / 100,
        "feature_names": tuple(f"feature_{i}" for i in range(10)),
        "categorical_encodings": (),
    }
    return {"meta": meta, "x": x, "y": y, "groups": None}


def _make_fake_timeseries_dataset(name: str = "kaggle_amex_default") -> LoadedDataset:
    """Create fake time-series dataset for testing.

    Args:
        name: Dataset name.

    Returns:
        LoadedDataset with synthetic aggregated time-series data.
    """
    rng = np.random.default_rng(123)
    # Time-series datasets typically have more features after aggregation
    x: NDArray[np.float64] = rng.random((500, 188)).astype(np.float64)
    y: NDArray[np.int64] = rng.integers(0, 2, size=500).astype(np.int64)
    n_positive = int(np.sum(y))
    meta: DatasetMeta = {
        "name": name,
        "n_samples": 500,
        "n_features": 188,
        "n_positive": n_positive,
        "n_negative": 500 - n_positive,
        "positive_ratio": n_positive / 500,
        "feature_names": tuple(f"ts_feature_{i}" for i in range(188)),
        "categorical_encodings": (),
    }
    return {"meta": meta, "x": x, "y": y, "groups": None}


def _make_fake_standard_config(name: str) -> DatasetConfig:
    """Create fake standard dataset config.

    Args:
        name: Dataset name.

    Returns:
        DatasetConfig for standard dataset.
    """
    return {
        "name": name,
        "display_name": f"Fake {name}",
        "folder": f"{name}_data",
        "file_name": "data.csv",
        "file_format": "csv",
        "encoding": "utf-8",
        "target": {
            "column_name": "target",
            "label_type": "binary_int",
            "positive_values": (1,),
            "negative_values": (0,),
        },
        "exclude_columns": (),
        "n_samples_expected": 100,
        "n_features_expected": 10,
        "positive_class_ratio_expected": 0.3,
    }


def _make_fake_timeseries_config(name: str) -> TimeSeriesDatasetConfig:
    """Create fake time-series dataset config.

    Args:
        name: Dataset name.

    Returns:
        TimeSeriesDatasetConfig for time-series dataset.
    """
    return TimeSeriesDatasetConfig(
        name=name,
        display_name=f"Fake {name} Time Series",
        folder=f"{name}_data",
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
        n_samples_expected=500,
        n_features_expected=188,
        positive_class_ratio_expected=0.26,
        time_series={
            "entity_column": "customer_ID",
            "time_column": "S_2",
            "aggregation": "last",
            "labels_file": "train_labels.csv",
            "labels_entity_column": "customer_ID",
            "include_rank_features": False,
            "include_diff_features": False,
            "include_window_features": False,
            "window_sizes": (),
        },
    )


def _make_fake_standard_registry() -> DatasetRegistry:
    """Create fake standard dataset registry.

    Returns:
        DatasetRegistry with taiwan, us, polish datasets.
    """
    configs = (
        _make_fake_standard_config("taiwan"),
        _make_fake_standard_config("us"),
        _make_fake_standard_config("polish"),
    )
    return DatasetRegistry(configs)


def _make_fake_timeseries_registry() -> TimeSeriesDatasetRegistry:
    """Create fake time-series dataset registry.

    Returns:
        TimeSeriesDatasetRegistry with kaggle_amex_default dataset.
    """
    configs = (_make_fake_timeseries_config("kaggle_amex_default"),)
    return TimeSeriesDatasetRegistry(configs)
