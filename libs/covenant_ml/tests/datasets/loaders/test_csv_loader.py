"""Tests for CSVLoader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from covenant_ml.datasets.loaders.csv_loader import CSVLoader, create_csv_loader
from covenant_ml.datasets.types import (
    CategoricalEncoding,
    DatasetConfig,
    FileEncoding,
    LabelType,
    LoadedDataset,
    TargetColumnSpec,
)


def _get_fixtures_dir() -> Path:
    """Get path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


def _make_config(
    name: str = "test",
    folder: str = "small_csv",
    file_name: str = "data.csv",
    target_column: str = "target",
    label_type: LabelType = "binary_int",
    positive_values: tuple[str | int, ...] = (1,),
    negative_values: tuple[str | int, ...] = (0,),
    exclude_columns: tuple[str, ...] = (),
    encoding: FileEncoding = "utf-8",
    n_samples_expected: int = 5,
    n_features_expected: int = 3,
) -> DatasetConfig:
    """Create a test dataset config."""
    return DatasetConfig(
        name=name,
        display_name=f"Test {name}",
        folder=folder,
        file_name=file_name,
        file_format="csv",
        encoding=encoding,
        target=TargetColumnSpec(
            column_name=target_column,
            label_type=label_type,
            positive_values=positive_values,
            negative_values=negative_values,
        ),
        exclude_columns=exclude_columns,
        n_samples_expected=n_samples_expected,
        n_features_expected=n_features_expected,
        positive_class_ratio_expected=0.4,
    )


class TestCSVLoader:
    """Tests for CSVLoader class."""

    def test_load_simple_csv(self) -> None:
        """Load simple CSV with numeric labels."""
        loader = CSVLoader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["meta"]["name"] == "test"
        assert result["meta"]["n_samples"] == 5
        assert result["meta"]["n_features"] == 3
        assert result["meta"]["n_positive"] == 2
        assert result["meta"]["n_negative"] == 3
        assert result["meta"]["feature_names"] == ("feature_1", "feature_2", "feature_3")

    def test_load_returns_correct_arrays(self) -> None:
        """Load returns correctly shaped arrays."""
        loader = CSVLoader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        assert result["x"].shape == (5, 3)
        assert result["y"].shape == (5,)
        assert result["x"].dtype == np.float64
        assert result["y"].dtype == np.int64

    def test_load_correct_feature_values(self) -> None:
        """Load parses feature values correctly."""
        loader = CSVLoader()
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
        loader = CSVLoader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        y_list: list[int] = result["y"].tolist()

        # Labels: 0, 1, 0, 1, 0
        assert y_list == [0, 1, 0, 1, 0]

    def test_load_string_labels(self) -> None:
        """Load handles string labels correctly."""
        loader = CSVLoader()
        config = _make_config(
            folder="string_labels",
            target_column="status",
            label_type="binary_str",
            positive_values=("bankrupt",),
            negative_values=("healthy",),
            exclude_columns=("id",),
        )
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)

        assert result["meta"]["n_samples"] == 5
        assert result["meta"]["n_features"] == 2  # id excluded
        # Labels: healthy, bankrupt, healthy, bankrupt, healthy -> 0,1,0,1,0
        y_list: list[int] = result["y"].tolist()
        assert y_list == [0, 1, 0, 1, 0]

    def test_load_missing_values_replaced_with_zero(self) -> None:
        """Load replaces missing values with 0.0."""
        loader = CSVLoader()
        config = _make_config(
            folder="string_labels",
            target_column="status",
            label_type="binary_str",
            positive_values=("bankrupt",),
            negative_values=("healthy",),
            exclude_columns=("id",),
        )
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        x_list: list[list[float]] = result["x"].tolist()

        # Row 2: 3.5, ? -> 3.5, 0.0
        assert x_list[1] == [3.5, 0.0]

        # Row 3: NA, 6.5 -> 0.0, 6.5
        assert x_list[2] == [0.0, 6.5]

    def test_load_excludes_columns(self) -> None:
        """Load excludes specified columns."""
        loader = CSVLoader()
        config = _make_config(
            folder="string_labels",
            target_column="status",
            label_type="binary_str",
            positive_values=("bankrupt",),
            negative_values=("healthy",),
            exclude_columns=("id",),
        )
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # id column excluded, only feature_1 and feature_2 remain
        assert result["meta"]["n_features"] == 2
        assert "id" not in result["meta"]["feature_names"]
        assert result["meta"]["feature_names"] == ("feature_1", "feature_2")

    def test_load_file_not_found_raises(self) -> None:
        """Load raises FileNotFoundError for missing file."""
        loader = CSVLoader()
        config = _make_config(file_name="nonexistent.csv")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            loader.load(config, fixtures_dir)

    def test_load_missing_column_raises(self) -> None:
        """Load raises ValueError for missing target column."""
        loader = CSVLoader()
        config = _make_config(target_column="nonexistent_column")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="Column 'nonexistent_column' not found"):
            loader.load(config, fixtures_dir)

    def test_load_unknown_label_raises(self) -> None:
        """Load raises ValueError for unknown label value."""
        loader = CSVLoader()
        # Configure to expect only 0/1 but data has 0 and 1
        config = _make_config(
            positive_values=(99,),  # Will not match any value
            negative_values=(98,),  # Will not match any value
        )
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="Unknown label value"):
            loader.load(config, fixtures_dir)

    def test_load_positive_ratio_calculated(self) -> None:
        """Load calculates positive class ratio correctly."""
        loader = CSVLoader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # 2 positive out of 5 = 0.4
        assert result["meta"]["positive_ratio"] == pytest.approx(0.4, abs=0.001)

    def test_load_case_insensitive_column_lookup(self) -> None:
        """Load finds columns case-insensitively."""
        loader = CSVLoader()
        config = _make_config(target_column="TARGET")  # Uppercase
        fixtures_dir = _get_fixtures_dir()

        # Should find "target" column even though we specified "TARGET"
        result = loader.load(config, fixtures_dir)
        assert result["meta"]["n_samples"] == 5

    def test_load_empty_data_raises(self) -> None:
        """Load raises ValueError for CSV file with no data rows."""
        loader = CSVLoader()
        config = _make_config(folder="empty_csv")
        fixtures_dir = _get_fixtures_dir()

        with pytest.raises(ValueError, match="No data rows found"):
            loader.load(config, fixtures_dir)

    def test_load_infinity_replaced_with_zero(self) -> None:
        """Load replaces infinity values with 0.0."""
        loader = CSVLoader()
        config = _make_config(
            folder="inf_values_csv",
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

    def test_load_mixed_int_string_labels(self) -> None:
        """Load handles mixed int and string labels.

        Covers branch where positive_values contains an int but
        the actual label is a string that doesn't parse as digit.
        """
        loader = CSVLoader()
        config = _make_config(
            folder="mixed_labels_csv",
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
        loader = CSVLoader()
        # Use multiple string values for negative - "NOPE" checked first, "NO" second
        config = _make_config(
            folder="mixed_labels_csv",
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

    def test_load_categorical_columns_detected(self) -> None:
        """Load detects and encodes categorical columns."""
        loader = CSVLoader()
        config = _make_config(
            folder="categorical_csv",
            n_samples_expected=5,
            n_features_expected=3,
        )
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)

        # Check metadata has categorical encodings
        encodings = result["meta"]["categorical_encodings"]
        assert len(encodings) == 2  # feature_cat and feature_cat2

        # Check feature_cat encoding (alphabetically: HIGH, LOW, MEDIUM)
        cat_encoding = encodings[0]
        assert cat_encoding["column_name"] == "feature_cat"
        assert cat_encoding["n_categories"] == 3
        # Mapping: HIGH=0, LOW=1, MEDIUM=2
        mapping_dict = dict(cat_encoding["mapping"])
        assert mapping_dict["HIGH"] == 0
        assert mapping_dict["LOW"] == 1
        assert mapping_dict["MEDIUM"] == 2

    def test_load_categorical_values_encoded_correctly(self) -> None:
        """Load encodes categorical values to integers."""
        loader = CSVLoader()
        config = _make_config(
            folder="categorical_csv",
            n_samples_expected=5,
            n_features_expected=3,
        )
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)
        x_list: list[list[float]] = result["x"].tolist()

        # feature_cat: LOW=1, MEDIUM=2, HIGH=0, LOW=1, MEDIUM=2
        assert x_list[0][1] == 1.0  # LOW
        assert x_list[1][1] == 2.0  # MEDIUM
        assert x_list[2][1] == 0.0  # HIGH
        assert x_list[3][1] == 1.0  # LOW
        assert x_list[4][1] == 2.0  # MEDIUM

    def test_load_categorical_with_missing_values(self) -> None:
        """Load handles missing values in categorical columns."""
        loader = CSVLoader()
        config = _make_config(
            folder="categorical_missing_csv",
            n_samples_expected=5,
            n_features_expected=2,
        )
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)

        # Check encoding includes _MISSING_ at code 0
        encodings = result["meta"]["categorical_encodings"]
        assert len(encodings) == 1
        cat_encoding = encodings[0]
        assert cat_encoding["column_name"] == "feature_cat"

        mapping_dict = dict(cat_encoding["mapping"])
        assert "_MISSING_" in mapping_dict
        assert mapping_dict["_MISSING_"] == 0

        # Check values: LOW, missing, HIGH, LOW, missing
        x_list: list[list[float]] = result["x"].tolist()
        # _MISSING_=0, HIGH=1, LOW=2 (alphabetically after _MISSING_)
        assert x_list[1][1] == 0.0  # missing -> 0
        assert x_list[4][1] == 0.0  # ? -> 0 (missing)

    def test_load_numeric_data_no_categorical_encodings(self) -> None:
        """Load returns empty categorical_encodings for all-numeric data."""
        loader = CSVLoader()
        config = _make_config()  # Uses small_csv which is all numeric
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)

        assert result["meta"]["categorical_encodings"] == ()

    def test_load_scientific_notation_numeric(self) -> None:
        """Load handles scientific notation as numeric (not categorical)."""
        loader = CSVLoader()
        config = _make_config(
            folder="scientific_csv",
            n_samples_expected=3,
            n_features_expected=2,
        )
        fixtures_dir = _get_fixtures_dir()

        result: LoadedDataset = loader.load(config, fixtures_dir)

        # No categorical encodings - all numeric
        assert result["meta"]["categorical_encodings"] == ()

        # Check values parsed correctly
        x_list: list[list[float]] = result["x"].tolist()
        # Row 1: 1e-5, 2.5E10
        assert x_list[0][0] == pytest.approx(1e-5, rel=1e-6)
        assert x_list[0][1] == pytest.approx(2.5e10, rel=1e-6)
        # Row 2: -1.5e3, 4e-2
        assert x_list[1][0] == pytest.approx(-1500.0, rel=1e-6)
        assert x_list[1][1] == pytest.approx(0.04, rel=1e-6)


class TestCSVLoaderNumericDetection:
    """Tests for numeric value detection edge cases."""

    def test_is_numeric_value_empty_string(self) -> None:
        """Empty string after sign stripping returns False."""
        loader = CSVLoader()
        # "-" alone after stripping sign becomes empty
        assert loader._is_numeric_value("-") is False
        assert loader._is_numeric_value("+") is False

    def test_is_numeric_value_multiple_decimals(self) -> None:
        """Multiple decimal points returns False."""
        loader = CSVLoader()
        assert loader._is_numeric_value("1.2.3") is False

    def test_is_numeric_value_non_digit_parts(self) -> None:
        """Non-digit characters in parts returns False."""
        loader = CSVLoader()
        assert loader._is_numeric_value("1.2a") is False
        assert loader._is_numeric_value("abc") is False

    def test_is_numeric_value_scientific_invalid_multiple_e(self) -> None:
        """Multiple 'e' in scientific notation returns False."""
        loader = CSVLoader()
        assert loader._is_numeric_value("1e2e3") is False

    def test_is_numeric_value_scientific_invalid_mantissa(self) -> None:
        """Invalid mantissa in scientific notation returns False."""
        loader = CSVLoader()
        assert loader._is_numeric_value("abce5") is False

    def test_is_numeric_value_scientific_invalid_exponent(self) -> None:
        """Invalid exponent in scientific notation returns False."""
        loader = CSVLoader()
        assert loader._is_numeric_value("1eabc") is False
        assert loader._is_numeric_value("1e") is False

    def test_is_simple_numeric_empty_value(self) -> None:
        """Empty value returns False."""
        loader = CSVLoader()
        assert loader._is_simple_numeric("") is False

    def test_is_simple_numeric_only_decimal(self) -> None:
        """Single decimal point with no digits returns False."""
        loader = CSVLoader()
        assert loader._is_simple_numeric(".") is False

    def test_is_simple_numeric_valid_decimal(self) -> None:
        """Valid decimal numbers return True."""
        loader = CSVLoader()
        assert loader._is_simple_numeric("1.5") is True
        assert loader._is_simple_numeric(".5") is True
        assert loader._is_simple_numeric("1.") is True


class TestCreateCSVLoader:
    """Tests for create_csv_loader factory."""

    def test_create_csv_loader_can_load_data(self) -> None:
        """Factory creates loader that can successfully load CSV data."""
        loader = create_csv_loader()
        config = _make_config()
        fixtures_dir = _get_fixtures_dir()

        result = loader.load(config, fixtures_dir)

        # Verify actual behavior - loader loaded correct number of samples
        assert result["meta"]["n_samples"] == 5
        assert result["meta"]["n_features"] == 3
