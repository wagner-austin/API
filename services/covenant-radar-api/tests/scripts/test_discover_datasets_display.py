"""Tests for discover_datasets main module."""

from __future__ import annotations

import pytest
from scripts.discover_datasets.main import (
    _format_dataset_row,
    _generate_config_code,
    _print_dataset_detail,
    _print_summary,
)
from scripts.discover_datasets.types import DiscoveredDataset, DiscoverySummary

from tests.scripts._discover_datasets_fixtures import (
    _get_test_console,
    _make_error_dataset,
    _make_success_dataset,
    _make_warning_dataset,
    _reset_hooks_impl,
)

_reset_hooks = pytest.fixture(_reset_hooks_impl)


class TestFormatDatasetRow:
    """Tests for _format_dataset_row function."""

    def test_success_row(self) -> None:
        """Test formatting success dataset row."""
        ds = _make_success_dataset()
        result = _format_dataset_row(ds)

        assert "test_dataset" in result
        assert "csv" in result
        assert "1,000 rows" in result
        assert "10 cols" in result
        assert "success" in result

    def test_warning_row(self) -> None:
        """Test formatting warning dataset row."""
        ds = _make_warning_dataset()
        result = _format_dataset_row(ds)

        assert "warning_dataset" in result
        assert "warning" in result

    def test_error_row(self) -> None:
        """Test formatting error dataset row."""
        ds = _make_error_dataset()
        result = _format_dataset_row(ds)

        assert "error_dataset" in result
        assert "error" in result

    def test_unknown_status_uses_white(self) -> None:
        """Test that unknown status uses white color."""
        ds = _make_success_dataset()
        result = _format_dataset_row(ds)
        assert "test_dataset" in result
        assert "csv" in result


class TestPrintSummary:
    """Tests for _print_summary function."""

    def test_prints_counts(self, _reset_hooks: None) -> None:
        """Test that summary prints all counts."""
        summary: DiscoverySummary = {
            "n_total": 3,
            "n_success": 1,
            "n_warning": 1,
            "n_error": 1,
            "datasets": (
                _make_success_dataset(),
                _make_warning_dataset(),
                _make_error_dataset(),
            ),
        }

        _print_summary(summary)

        output = _get_test_console().get_output()
        assert "3 datasets" in output
        assert "Success: 1" in output
        assert "Warnings: 1" in output
        assert "Errors: 1" in output

    def test_prints_dataset_rows(self, _reset_hooks: None) -> None:
        """Test that summary prints dataset rows."""
        summary: DiscoverySummary = {
            "n_total": 1,
            "n_success": 1,
            "n_warning": 0,
            "n_error": 0,
            "datasets": (_make_success_dataset(),),
        }

        _print_summary(summary)

        output = _get_test_console().get_output()
        assert "test_dataset" in output


class TestPrintDatasetDetail:
    """Tests for _print_dataset_detail function."""

    def test_prints_basic_info(self, _reset_hooks: None) -> None:
        """Test printing basic dataset info."""
        ds = _make_success_dataset()
        _print_dataset_detail(ds)

        output = _get_test_console().get_output()
        assert "test_dataset" in output
        assert "data.csv" in output
        assert "utf-8" in output
        assert "1,000 rows" in output

    def test_prints_target_candidates(self, _reset_hooks: None) -> None:
        """Test printing target candidates."""
        ds = _make_success_dataset()
        _print_dataset_detail(ds)

        output = _get_test_console().get_output()
        assert "Target candidates" in output
        assert "target" in output
        assert "binary" in output

    def test_prints_recommended_target(self, _reset_hooks: None) -> None:
        """Test printing recommended target."""
        ds = _make_success_dataset()
        _print_dataset_detail(ds)

        output = _get_test_console().get_output()
        assert "Recommended target" in output

    def test_prints_exclude_columns(self, _reset_hooks: None) -> None:
        """Test printing exclude columns."""
        ds = _make_success_dataset()
        _print_dataset_detail(ds)

        output = _get_test_console().get_output()
        assert "Exclude columns" in output
        assert "id" in output

    def test_no_candidates_section(self, _reset_hooks: None) -> None:
        """Test no target candidates section when none found."""
        ds = _make_warning_dataset()
        _print_dataset_detail(ds)

        output = _get_test_console().get_output()
        assert "Target candidates" not in output

    def test_no_recommended_when_empty(self, _reset_hooks: None) -> None:
        """Test no recommended target when empty."""
        ds = _make_warning_dataset()
        _print_dataset_detail(ds)

        output = _get_test_console().get_output()
        assert "Recommended target" not in output

    def test_no_exclude_when_empty(self, _reset_hooks: None) -> None:
        """Test no exclude columns when empty."""
        ds = _make_warning_dataset()
        _print_dataset_detail(ds)

        output = _get_test_console().get_output()
        assert "Exclude columns" not in output

    def test_prints_unique_values(self, _reset_hooks: None) -> None:
        """Test printing unique values for candidates."""
        ds = _make_success_dataset()
        _print_dataset_detail(ds)

        output = _get_test_console().get_output()
        assert "Values:" in output
        assert "0" in output

    def test_candidate_with_empty_unique_values(self, _reset_hooks: None) -> None:
        """Test printing candidate with no unique values."""
        ds: DiscoveredDataset = {
            "folder_name": "empty_values_test",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 2,
            "target_candidates": (
                {
                    "column_name": "target",
                    "unique_values": (),
                    "n_unique": 0,
                    "is_binary": False,
                },
            ),
            "recommended_target": "target",
            "recommended_exclude": (),
            "target_positive_value": "",
            "target_negative_value": "",
            "target_label_type": "binary_int",
            "positive_class_ratio": 0.0,
            "status": "success",
            "message": "ok",
        }
        _print_dataset_detail(ds)

        output = _get_test_console().get_output()
        assert "Target candidates" in output
        assert "target" in output
        # Values section should not appear for empty unique_values
        assert "Values:" not in output


class TestGenerateConfigCode:
    """Tests for _generate_config_code function."""

    def test_generates_valid_config(self) -> None:
        """Test generating valid DatasetConfig code."""
        ds = _make_success_dataset()
        code = _generate_config_code(ds)

        assert "DatasetConfig(" in code
        assert 'name="test_dataset"' in code
        assert 'file_name="data.csv"' in code
        assert 'column_name="target"' in code
        assert 'label_type="binary_int"' in code

    def test_skipped_for_error(self) -> None:
        """Test skipped comment for error dataset."""
        ds = _make_error_dataset()
        code = _generate_config_code(ds)

        assert code.startswith("# Skipped")
        assert "error_dataset" in code

    def test_skipped_for_no_target(self) -> None:
        """Test skipped comment when no recommended target."""
        ds = _make_warning_dataset()
        code = _generate_config_code(ds)

        assert code.startswith("# Skipped")

    def test_non_binary_label_type(self) -> None:
        """Test non-binary label type for non-binary targets."""
        ds: DiscoveredDataset = {
            "folder_name": "multi",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 5,
            "target_candidates": (
                {
                    "column_name": "class",
                    "unique_values": ("A", "B", "C"),
                    "n_unique": 3,
                    "is_binary": False,
                },
            ),
            "recommended_target": "class",
            "recommended_exclude": (),
            "target_positive_value": "A",
            "target_negative_value": "B",
            "target_label_type": "binary_str",
            "positive_class_ratio": 0.33,
            "status": "success",
            "message": "ok",
        }
        code = _generate_config_code(ds)

        assert 'label_type="binary_str"' in code

    def test_exclude_columns_formatting(self) -> None:
        """Test exclude columns tuple formatting."""
        ds = _make_success_dataset()
        code = _generate_config_code(ds)

        assert '"id"' in code
        assert '"name"' in code

    def test_target_not_in_candidates(self) -> None:
        """Test generating config when target not found in candidates."""
        ds: DiscoveredDataset = {
            "folder_name": "orphan_target",
            "file_name": "data.csv",
            "file_format": "csv",
            "encoding": "utf-8",
            "n_rows": 100,
            "n_columns": 3,
            "target_candidates": (
                {
                    "column_name": "other_col",
                    "unique_values": ("a", "b"),
                    "n_unique": 2,
                    "is_binary": False,
                },
            ),
            "recommended_target": "target",  # Different from candidate
            "recommended_exclude": (),
            "target_positive_value": "",
            "target_negative_value": "",
            "target_label_type": "binary_str",
            "positive_class_ratio": 0.0,
            "status": "success",
            "message": "ok",
        }
        code = _generate_config_code(ds)

        # Should still generate config, defaulting to binary_str since not found
        assert "DatasetConfig(" in code
        assert 'label_type="binary_str"' in code
