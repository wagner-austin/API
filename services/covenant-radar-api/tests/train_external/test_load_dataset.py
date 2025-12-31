"""Tests for _load_dataset function."""

from __future__ import annotations

from pathlib import Path

import pytest

from covenant_radar_api.worker.train_external_job import _load_dataset

from .conftest import copy_real_polish, copy_real_taiwan, copy_real_us


class TestLoadDataset:
    """Tests for _load_dataset function."""

    def test_taiwan(self, tmp_path: Path) -> None:
        """_load_dataset loads Taiwan data successfully."""
        _, n_rows, feature_names = copy_real_taiwan(tmp_path)
        dataset = _load_dataset("taiwan", tmp_path)
        meta = dataset["meta"]

        assert meta["n_samples"] == n_rows
        assert meta["n_features"] == len(feature_names)

    def test_us(self, tmp_path: Path) -> None:
        """_load_dataset loads US data successfully."""
        _, n_rows, feature_names = copy_real_us(tmp_path)
        dataset = _load_dataset("us", tmp_path)
        meta = dataset["meta"]

        assert meta["n_samples"] == n_rows
        assert meta["n_features"] == len(feature_names)

    def test_polish(self, tmp_path: Path) -> None:
        """_load_dataset loads Polish data successfully."""
        _, n_rows, feature_names = copy_real_polish(tmp_path)
        dataset = _load_dataset("polish", tmp_path)
        meta = dataset["meta"]

        assert meta["n_samples"] == n_rows
        assert meta["n_features"] == len(feature_names)

    def test_missing_taiwan(self, tmp_path: Path) -> None:
        """_load_dataset raises FileNotFoundError for missing Taiwan data."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            _load_dataset("taiwan", tmp_path)

    def test_missing_us(self, tmp_path: Path) -> None:
        """_load_dataset raises FileNotFoundError for missing US data."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            _load_dataset("us", tmp_path)

    def test_missing_polish(self, tmp_path: Path) -> None:
        """_load_dataset raises FileNotFoundError for missing Polish data."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            _load_dataset("polish", tmp_path)
