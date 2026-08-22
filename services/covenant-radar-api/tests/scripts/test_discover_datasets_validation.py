"""Tests for discover_datasets main module."""

from __future__ import annotations

import pytest
from scripts.discover_datasets.main import (
    _classify_dataset_for_validation,
    _format_value_tuple,
    _is_valid_numeric,
    _print_validation,
)
from scripts.discover_datasets.types import DiscoveredDataset

from tests.scripts._discover_datasets_fixtures import (
    _get_test_console,
    _reset_hooks_impl,
)

_reset_hooks = pytest.fixture(_reset_hooks_impl)


class TestIsValidNumeric:
    """Tests for _is_valid_numeric function."""

    def test_empty_string_returns_false(self) -> None:
        """Test that empty string returns False."""
        assert _is_valid_numeric("") is False

    def test_valid_integer(self) -> None:
        """Test valid integer string."""
        assert _is_valid_numeric("123") is True

    def test_valid_float(self) -> None:
        """Test valid float string."""
        assert _is_valid_numeric("123.45") is True

    def test_negative_number(self) -> None:
        """Test negative number."""
        assert _is_valid_numeric("-42") is True

    def test_non_numeric(self) -> None:
        """Test non-numeric string."""
        assert _is_valid_numeric("abc") is False


class TestFormatValueTuple:
    """Tests for _format_value_tuple function."""

    def test_binary_int_with_numeric_values(self) -> None:
        """Test binary_int with numeric positive/negative values."""
        pos_str, neg_str = _format_value_tuple("1", "0", "binary_int")
        assert pos_str == "(1,)"
        assert neg_str == "(0,)"

    def test_binary_int_with_non_numeric_values(self) -> None:
        """Test binary_int with non-numeric values uses string fallback."""
        pos_str, neg_str = _format_value_tuple("yes", "no", "binary_int")
        assert pos_str == '("yes",)'
        assert neg_str == '("no",)'

    def test_binary_int_with_empty_values(self) -> None:
        """Test binary_int with empty values uses defaults."""
        pos_str, neg_str = _format_value_tuple("", "", "binary_int")
        assert pos_str == "(1,)"
        assert neg_str == "(0,)"

    def test_binary_str_with_values(self) -> None:
        """Test binary_str format."""
        pos_str, neg_str = _format_value_tuple("positive", "negative", "binary_str")
        assert pos_str == '("positive",)'
        assert neg_str == '("negative",)'

    def test_binary_str_with_empty_values(self) -> None:
        """Test binary_str with empty values."""
        pos_str, neg_str = _format_value_tuple("", "", "binary_str")
        assert pos_str == '("",)'
        assert neg_str == '("",)'


class TestClassifyDatasetForValidation:
    """Tests for _classify_dataset_for_validation function."""

    def test_error_status_returns_skip(self) -> None:
        """Test that error status returns skip."""
        ds: DiscoveredDataset = {
            "folder_name": "test",
            "file_name": "",
            "file_format": "unknown",
            "encoding": "utf-8",
            "n_rows": 0,
            "n_columns": 0,
            "target_candidates": (),
            "recommended_target": "",
            "recommended_exclude": (),
            "target_positive_value": "",
            "target_negative_value": "",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.0,
            "status": "error",
            "message": "Error",
        }
        assert _classify_dataset_for_validation(ds) == "skip"

    def test_no_target_returns_skip(self) -> None:
        """Test that no recommended target returns skip."""
        ds: DiscoveredDataset = {
            "folder_name": "test",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 5,
            "target_candidates": (),
            "recommended_target": "",
            "recommended_exclude": (),
            "target_positive_value": "",
            "target_negative_value": "",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.0,
            "status": "warning",
            "message": "No target",
        }
        assert _classify_dataset_for_validation(ds) == "skip"

    def test_complete_config_returns_pass(self) -> None:
        """Test that complete config returns pass."""
        ds: DiscoveredDataset = {
            "folder_name": "test",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 5,
            "target_candidates": (),
            "recommended_target": "target",
            "recommended_exclude": (),
            "target_positive_value": "1",
            "target_negative_value": "0",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.3,
            "status": "success",
            "message": "OK",
        }
        assert _classify_dataset_for_validation(ds) == "pass"

    def test_missing_values_returns_warn(self) -> None:
        """Test that missing positive/negative values returns warn."""
        ds: DiscoveredDataset = {
            "folder_name": "test",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 5,
            "target_candidates": (),
            "recommended_target": "target",
            "recommended_exclude": (),
            "target_positive_value": "",
            "target_negative_value": "",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.0,
            "status": "success",
            "message": "OK",
        }
        assert _classify_dataset_for_validation(ds) == "warn"


class TestPrintValidation:
    """Tests for _print_validation function."""

    def test_error_status_prints_skip(self, _reset_hooks: None) -> None:
        """Test that error status prints SKIP."""
        ds: DiscoveredDataset = {
            "folder_name": "error_folder",
            "file_name": "",
            "file_format": "unknown",
            "encoding": "utf-8",
            "n_rows": 0,
            "n_columns": 0,
            "target_candidates": (),
            "recommended_target": "",
            "recommended_exclude": (),
            "target_positive_value": "",
            "target_negative_value": "",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.0,
            "status": "error",
            "message": "No data files",
        }
        _print_validation(ds)
        output = _get_test_console().get_output()
        assert "SKIP" in output
        assert "error_folder" in output

    def test_no_target_prints_skip(self, _reset_hooks: None) -> None:
        """Test that no target prints SKIP."""
        ds: DiscoveredDataset = {
            "folder_name": "no_target_folder",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 5,
            "target_candidates": (),
            "recommended_target": "",
            "recommended_exclude": (),
            "target_positive_value": "",
            "target_negative_value": "",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.0,
            "status": "warning",
            "message": "No target",
        }
        _print_validation(ds)
        output = _get_test_console().get_output()
        assert "SKIP" in output
        assert "no_target_folder" in output

    def test_missing_positive_value_prints_warn(self, _reset_hooks: None) -> None:
        """Test that missing positive value prints WARN."""
        ds: DiscoveredDataset = {
            "folder_name": "warn_folder",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 5,
            "target_candidates": (),
            "recommended_target": "target",
            "recommended_exclude": (),
            "target_positive_value": "",
            "target_negative_value": "0",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.3,
            "status": "success",
            "message": "OK",
        }
        _print_validation(ds)
        output = _get_test_console().get_output()
        assert "WARN" in output
        assert "Missing positive_value" in output

    def test_missing_negative_value_prints_warn(self, _reset_hooks: None) -> None:
        """Test that missing negative value prints WARN."""
        ds: DiscoveredDataset = {
            "folder_name": "warn_folder",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 5,
            "target_candidates": (),
            "recommended_target": "target",
            "recommended_exclude": (),
            "target_positive_value": "1",
            "target_negative_value": "",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.3,
            "status": "success",
            "message": "OK",
        }
        _print_validation(ds)
        output = _get_test_console().get_output()
        assert "WARN" in output
        assert "Missing negative_value" in output

    def test_zero_ratio_with_positive_value_prints_warn(self, _reset_hooks: None) -> None:
        """Test that 0.0 ratio with positive value prints WARN."""
        ds: DiscoveredDataset = {
            "folder_name": "warn_folder",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 5,
            "target_candidates": (),
            "recommended_target": "target",
            "recommended_exclude": (),
            "target_positive_value": "1",
            "target_negative_value": "0",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.0,
            "status": "success",
            "message": "OK",
        }
        _print_validation(ds)
        output = _get_test_console().get_output()
        assert "WARN" in output
        assert "Ratio is 0.0" in output

    def test_pass_with_exclude_columns(self, _reset_hooks: None) -> None:
        """Test that PASS prints exclude columns."""
        ds: DiscoveredDataset = {
            "folder_name": "pass_folder",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 5,
            "target_candidates": (),
            "recommended_target": "target",
            "recommended_exclude": ("id", "name"),
            "target_positive_value": "1",
            "target_negative_value": "0",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.3,
            "status": "success",
            "message": "OK",
        }
        _print_validation(ds)
        output = _get_test_console().get_output()
        assert "PASS" in output
        assert "Exclude:" in output
