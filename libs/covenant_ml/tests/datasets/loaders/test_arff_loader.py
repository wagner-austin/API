"""Tests for ARFFLoader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from covenant_ml.datasets.loaders.arff_loader import ARFFLoader, create_arff_loader
from covenant_ml.datasets.types import DatasetConfig, LoadedDataset, TargetColumnSpec


def _get_fixtures_dir() -> Path:
    """Get path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


def _make_config(
    name: str = "test",
    folder: str = "small_arff",
    file_name: str = "data.arff",
    target_column: str = "class",
    positive_values: tuple[str | int, ...] = (1,),
    negative_values: tuple[str | int, ...] = (0,),
    exclude_columns: tuple[str, ...] = (),
    n_samples_expected: int = 5,
    n_features_expected: int = 3,
) -> DatasetConfig:
    """Create a test dataset config for ARFF."""
    return DatasetConfig(
        name=name,
        display_name=f"Test {name}",
        folder=folder,
        file_name=file_name,
        file_format="arff",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name=target_column,
            label_type="binary_int",
            positive_values=positive_values,
            negative_values=negative_values,
        ),
        exclude_columns=exclude_columns,
        n_samples_expected=n_samples_expected,
        n_features_expected=n_features_expected,
        positive_class_ratio_expected=0.4,
    )


class TestARFFLoader:
    """Tests for ARFFLoader class."""

    def test_load_simple_arff(self) -> None:
        """Load simple ARFF with numeric labels."""
        loader = ARFFLoader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["name"] == "test"
        assert result["meta"]["n_samples"] == 5
        assert result["meta"]["n_features"] == 3
        assert result["meta"]["n_positive"] == 2
        assert result["meta"]["n_negative"] == 3
        assert result["meta"]["feature_names"] == ("Attr1", "Attr2", "Attr3")

    def test_load_returns_correct_arrays(self) -> None:
        """Load returns correctly shaped arrays."""
        loader = ARFFLoader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["x"].shape == (5, 3)
        assert result["y"].shape == (5,)
        assert result["x"].dtype == np.float64
        assert result["y"].dtype == np.int64

    def test_load_correct_feature_values(self) -> None:
        """Load parses feature values correctly."""
        loader = ARFFLoader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        x_list: list[list[float]] = result["x"].tolist()

        # First row: 1.0, 2.0, 3.0
        assert x_list[0] == [1.0, 2.0, 3.0]

        # Second row: 4.0, 5.0, 6.0
        assert x_list[1] == [4.0, 5.0, 6.0]

    def test_load_correct_labels(self) -> None:
        """Load parses labels correctly."""
        loader = ARFFLoader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        y_list: list[int] = result["y"].tolist()

        # Labels: 0, 1, 0, 1, 0
        assert y_list == [0, 1, 0, 1, 0]

    def test_load_missing_values_replaced_with_zero(self) -> None:
        """Load replaces missing values (?) with 0.0."""
        loader = ARFFLoader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        x_list: list[list[float]] = result["x"].tolist()

        # Row 3: ?, 8.0, 9.0 -> 0.0, 8.0, 9.0
        assert x_list[2] == [0.0, 8.0, 9.0]

    def test_load_file_not_found_raises(self) -> None:
        """Load raises FileNotFoundError for missing file."""
        loader = ARFFLoader()
        config = _make_config(file_name="nonexistent.arff")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            loader.load(config, fixtures_dir)

    def test_load_missing_attribute_raises(self) -> None:
        """Load raises ValueError for missing target attribute."""
        loader = ARFFLoader()
        config = _make_config(target_column="nonexistent")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            loader.load(config, fixtures_dir)

    def test_load_unknown_label_raises(self) -> None:
        """Load raises ValueError for unknown label value."""
        loader = ARFFLoader()
        config = _make_config(
            positive_values=(99,),
            negative_values=(98,),
        )
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="Unknown label value"):
            loader.load(config, fixtures_dir)

    def test_load_positive_ratio_calculated(self) -> None:
        """Load calculates positive class ratio correctly."""
        loader = ARFFLoader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # 2 positive out of 5 = 0.4
        assert result["meta"]["positive_ratio"] == pytest.approx(0.4, abs=0.001)

    def test_load_case_insensitive_attribute_lookup(self) -> None:
        """Load finds attributes case-insensitively."""
        loader = ARFFLoader()
        config = _make_config(target_column="CLASS")  # Uppercase
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)
        assert result["meta"]["n_samples"] == 5

    def test_load_empty_data_raises(self) -> None:
        """Load raises ValueError for ARFF file with no data rows."""
        loader = ARFFLoader()
        config = _make_config(folder="empty_arff", target_column="class")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="No data rows found"):
            loader.load(config, fixtures_dir)

    def test_load_infinity_replaced_with_zero(self) -> None:
        """Load replaces infinity values with 0.0."""
        loader = ARFFLoader()
        config = _make_config(
            folder="inf_values_arff",
            n_samples_expected=3,
            n_features_expected=2,
        )
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        x_list: list[list[float]] = result["x"].tolist()

        # Row 1: 1.0, inf -> 1.0, 0.0
        assert x_list[0] == [1.0, 0.0]

        # Row 2: -inf, 2.0 -> 0.0, 2.0
        assert x_list[1] == [0.0, 2.0]

        # Row 3: 3.0, 4.0 (no changes)
        assert x_list[2] == [3.0, 4.0]

    def test_load_string_labels(self) -> None:
        """Load handles string labels in ARFF files."""
        loader = ARFFLoader()
        config = _make_config(
            folder="string_labels_arff",
            target_column="status",
            positive_values=("bankrupt",),
            negative_values=("healthy",),
            n_samples_expected=5,
            n_features_expected=2,
        )
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        y_list: list[int] = result["y"].tolist()

        # Labels: healthy, bankrupt, healthy, bankrupt, healthy -> 0,1,0,1,0
        assert y_list == [0, 1, 0, 1, 0]

    def test_load_malformed_attribute_skipped(self) -> None:
        """Load skips malformed @attribute lines with fewer than 2 parts."""
        loader = ARFFLoader()
        config = _make_config(
            folder="malformed_arff",
            target_column="class",
            n_samples_expected=3,
            n_features_expected=2,
        )
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # Malformed @attribute line was skipped, only Attr1 and Attr2 are features
        assert result["meta"]["n_features"] == 2
        assert result["meta"]["feature_names"] == ("Attr1", "Attr2")

    def test_load_mixed_int_string_labels(self) -> None:
        """Load handles mixed int and string labels.

        Covers branch where positive_values contains an int but
        the actual label is a string that doesn't parse as digit.
        """
        loader = ARFFLoader()
        config = _make_config(
            folder="mixed_labels_arff",
            target_column="class",
            positive_values=(1, "YES"),
            negative_values=(0, "NO"),
            n_samples_expected=5,
            n_features_expected=2,
        )
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        y_list: list[int] = result["y"].tolist()

        # Labels: 0, YES, NO, 1, NO -> 0, 1, 0, 1, 0
        assert y_list == [0, 1, 0, 1, 0]

    def test_load_multiple_string_negative_values(self) -> None:
        """Load handles multiple string negative values.

        Covers branch where first string negative value doesn't match
        but loop continues to check next value.
        """
        loader = ARFFLoader()
        # Use multiple string values for negative - "NOPE" checked first, "NO" second
        config = _make_config(
            folder="mixed_labels_arff",
            target_column="class",
            positive_values=(1, "YES"),
            negative_values=("NOPE", "NO", 0),  # "NOPE" first, then "NO", then 0
            n_samples_expected=5,
            n_features_expected=2,
        )
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        y_list: list[int] = result["y"].tolist()

        # Labels: 0, YES, NO, 1, NO -> 0, 1, 0, 1, 0
        # When checking "NO": first checks "NOPE" (string, no match), continues,
        # then checks "NO" (string, match) -> return 0
        assert y_list == [0, 1, 0, 1, 0]


class TestCreateARFFLoader:
    """Tests for create_arff_loader factory."""

    def test_create_arff_loader_can_load_data(self) -> None:
        """Factory creates loader that can successfully load ARFF data."""
        loader = create_arff_loader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # Verify actual behavior - loader loaded correct number of samples
        assert result["meta"]["n_samples"] == 5
        assert result["meta"]["n_features"] == 3
