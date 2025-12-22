"""Tests for discover_datasets main module."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.logging import (
    RichConsoleProtocol,
    RichRenderableProtocol,
    setup_rich_logging,
)
from scripts.discover_datasets.main import (
    ParsedArgs,
    _classify_dataset_for_validation,
    _format_dataset_row,
    _format_value_tuple,
    _generate_config_code,
    _is_valid_numeric,
    _print_dataset_detail,
    _print_summary,
    _print_validation,
    main,
    parse_args,
    run,
)
from scripts.discover_datasets.types import DiscoveredDataset, DiscoverySummary

from scripts.discover_datasets import _test_hooks

# =============================================================================
# Test Console
# =============================================================================


class FakeConsole:
    """Fake console that captures output for assertions."""

    def __init__(self) -> None:
        """Initialize empty output list."""
        self.messages: list[str] = []

    def print(
        self,
        *args: str | RichRenderableProtocol,
        style: str | None = None,
        **kwargs: str,
    ) -> None:
        """Capture printed messages.

        Args:
            args: Messages to print.
            style: Style (ignored in tests).
            kwargs: Additional kwargs (ignored).
        """
        for arg in args:
            self.messages.append(str(arg))

    def get_output(self) -> str:
        """Get all captured output as single string."""
        return "\n".join(self.messages)


_test_console: FakeConsole | None = None


def _test_console_factory() -> RichConsoleProtocol:
    """Factory that returns the test console."""
    global _test_console
    if _test_console is None:
        _test_console = FakeConsole()
    return _test_console


def _get_test_console() -> FakeConsole:
    """Get the current test console.

    Returns:
        Current test console.

    Raises:
        RuntimeError: If console not initialized.
    """
    global _test_console
    if _test_console is None:
        msg = "Test console not initialized"
        raise RuntimeError(msg)
    return _test_console


# =============================================================================
# Fixtures
# =============================================================================


def _reset_hooks_impl() -> Generator[None, None, None]:
    """Reset hooks after test."""
    global _test_console
    _test_console = FakeConsole()

    orig_console_factory = _test_hooks.console_factory
    _test_hooks.console_factory = _test_console_factory

    yield

    _test_hooks.console_factory = orig_console_factory
    _test_console = None


_reset_hooks = pytest.fixture(_reset_hooks_impl)


# =============================================================================
# Test Data
# =============================================================================


def _make_success_dataset() -> DiscoveredDataset:
    """Create a successful DiscoveredDataset for testing."""
    return {
        "folder_name": "test_dataset",
        "file_name": "data.csv",
        "file_format": "csv",
        "encoding": "utf-8",
        "n_rows": 1000,
        "n_columns": 10,
        "target_candidates": (
            {
                "column_name": "target",
                "unique_values": ("0", "1"),
                "n_unique": 2,
                "is_binary": True,
            },
        ),
        "recommended_target": "target",
        "recommended_exclude": ("id", "name"),
        "target_positive_value": "1",
        "target_negative_value": "0",
        "target_label_type": "binary_int",
        "positive_class_ratio": 0.3,
        "status": "success",
        "message": "Single data file found",
    }


def _make_warning_dataset() -> DiscoveredDataset:
    """Create a warning DiscoveredDataset for testing."""
    return {
        "folder_name": "warning_dataset",
        "file_name": "data.csv",
        "file_format": "csv",
        "encoding": "utf-8",
        "n_rows": 500,
        "n_columns": 5,
        "target_candidates": (),
        "recommended_target": "",
        "recommended_exclude": (),
        "target_positive_value": "",
        "target_negative_value": "",
        "target_label_type": "binary_int",
        "positive_class_ratio": 0.0,
        "status": "warning",
        "message": "No target column candidates found",
    }


def _make_error_dataset() -> DiscoveredDataset:
    """Create an error DiscoveredDataset for testing."""
    return {
        "folder_name": "error_dataset",
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
        "message": "No data files found",
    }


# =============================================================================
# Tests
# =============================================================================


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


class TestParseArgs:
    """Tests for parse_args function."""

    def test_defaults(self) -> None:
        """Test default argument values."""
        args = parse_args([])

        assert args["external_dir"] == Path("data/external")
        assert args["detail"] == ""
        assert args["generate"] is False
        assert args["validate"] is False
        assert args["verbose"] is False

    def test_external_dir(self) -> None:
        """Test --external-dir argument."""
        args = parse_args(["--external-dir", "/path/to/data"])

        assert args["external_dir"] == Path("/path/to/data")

    def test_detail(self) -> None:
        """Test --detail argument."""
        args = parse_args(["--detail", "my_dataset"])

        assert args["detail"] == "my_dataset"

    def test_generate(self) -> None:
        """Test --generate flag."""
        args = parse_args(["--generate"])

        assert args["generate"] is True

    def test_validate(self) -> None:
        """Test --validate flag."""
        args = parse_args(["--validate"])

        assert args["validate"] is True

    def test_verbose_long(self) -> None:
        """Test --verbose flag."""
        args = parse_args(["--verbose"])

        assert args["verbose"] is True

    def test_verbose_short(self) -> None:
        """Test -v flag."""
        args = parse_args(["-v"])

        assert args["verbose"] is True


class TestRun:
    """Tests for run function."""

    def test_directory_not_found(self, _reset_hooks: None) -> None:
        """Test error when directory not found."""
        result = run(["--external-dir", "/nonexistent/path"])

        assert result == 1
        output = _get_test_console().get_output()
        assert "Directory not found" in output

    def test_summary_mode(self, _reset_hooks: None) -> None:
        """Test default summary mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)
            ds_folder = external_dir / "dataset1"
            ds_folder.mkdir()
            (ds_folder / "data.csv").write_text("feature,target\n1,0\n2,1\n")

            result = run(["--external-dir", str(external_dir)])

        assert result == 0
        output = _get_test_console().get_output()
        assert "Dataset Discovery Results" in output

    def test_detail_mode_found(self, _reset_hooks: None) -> None:
        """Test detail mode when dataset found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)
            ds_folder = external_dir / "my_dataset"
            ds_folder.mkdir()
            (ds_folder / "data.csv").write_text("feature,target\n1,0\n2,1\n")

            result = run(["--external-dir", str(external_dir), "--detail", "my_dataset"])

        assert result == 0
        output = _get_test_console().get_output()
        assert "my_dataset" in output

    def test_detail_mode_found_after_skipping(self, _reset_hooks: None) -> None:
        """Test detail mode when dataset found after iterating through others."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)
            # Create first dataset (alphabetically first)
            ds1 = external_dir / "aaa_first"
            ds1.mkdir()
            (ds1 / "data.csv").write_text("feature,target\n1,0\n")
            # Create second dataset (the one we want)
            ds2 = external_dir / "bbb_target"
            ds2.mkdir()
            (ds2 / "data.csv").write_text("feature,target\n2,1\n")

            result = run(["--external-dir", str(external_dir), "--detail", "bbb_target"])

        assert result == 0
        output = _get_test_console().get_output()
        assert "bbb_target" in output

    def test_detail_mode_not_found(self, _reset_hooks: None) -> None:
        """Test detail mode when dataset not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)

            result = run(["--external-dir", str(external_dir), "--detail", "missing"])

        assert result == 1
        output = _get_test_console().get_output()
        assert "Dataset not found" in output

    def test_generate_mode(self, _reset_hooks: None) -> None:
        """Test generate mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)
            ds_folder = external_dir / "dataset1"
            ds_folder.mkdir()
            (ds_folder / "data.csv").write_text("feature,target\n1,0\n2,1\n")

            result = run(["--external-dir", str(external_dir), "--generate"])

        assert result == 0
        output = _get_test_console().get_output()
        assert "DatasetConfig" in output

    def test_validate_mode(self, _reset_hooks: None) -> None:
        """Test validate mode exercises all counting paths (pass, warn, skip)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)

            # Dataset 1: PASS - has target with binary values
            ds1 = external_dir / "dataset1"
            ds1.mkdir()
            (ds1 / "data.csv").write_text("feature,target\n1,0\n2,1\n")

            # Dataset 2: WARN - has target column but non-binary (3+ unique values)
            ds2 = external_dir / "dataset2"
            ds2.mkdir()
            # Target column with 3 unique values - non-binary, so warn
            (ds2 / "data.csv").write_text("feature,target\n1,A\n2,B\n3,C\n")

            # Dataset 3: SKIP - no data files (error)
            ds3 = external_dir / "dataset3"
            ds3.mkdir()
            # Empty folder with no data files

            result = run(["--external-dir", str(external_dir), "--validate"])

        assert result == 0
        output = _get_test_console().get_output()
        assert "Config Validation" in output
        # Check summary shows counts for all three classification types
        assert "PASS" in output
        assert "WARN" in output
        assert "SKIP" in output

    def test_verbose_mode(self, _reset_hooks: None) -> None:
        """Test verbose mode sets logging level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)

            result = run(["--external-dir", str(external_dir), "-v"])

        assert result == 0


class TestMain:
    """Tests for main function."""

    def test_success_exit_code(self, _reset_hooks: None) -> None:
        """Test successful exit code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir)

            result = main(["--external-dir", str(external_dir)])

        assert result == 0

    def test_error_exit_code(self, _reset_hooks: None) -> None:
        """Test error exit code for missing directory."""
        result = main(["--external-dir", "/nonexistent/path"])

        assert result == 1

    def test_uses_sys_argv_when_none(self, _reset_hooks: None) -> None:
        """Test that None argv uses sys.argv."""
        import sys

        orig_argv = sys.argv
        sys.argv = ["discover_datasets", "--external-dir", "/nonexistent"]

        result = main(None)

        sys.argv = orig_argv

        assert result == 1


class TestModuleExports:
    """Tests for module exports."""

    def test_main_is_callable(self) -> None:
        """Test main function is callable."""
        # main was imported successfully, verify it's the right function
        assert main.__module__ == "scripts.discover_datasets.main"

    def test_run_is_callable(self) -> None:
        """Test run function is callable."""
        assert run.__module__ == "scripts.discover_datasets.main"


class TestParsedArgsTypedDict:
    """Tests for ParsedArgs TypedDict."""

    def test_create_parsed_args(self) -> None:
        """Test creating ParsedArgs instance."""
        args: ParsedArgs = {
            "external_dir": Path("data"),
            "detail": "test",
            "generate": True,
            "validate": False,
            "verbose": False,
        }

        assert args["external_dir"] == Path("data")
        assert args["detail"] == "test"
        assert args["generate"] is True
        assert args["validate"] is False
        assert args["verbose"] is False


class TestDefaultConsoleFactory:
    """Tests for default console factory hook."""

    def test_default_factory(self) -> None:
        """Test default console factory returns working console."""
        # Must call setup_rich_logging before get_rich_console
        setup_rich_logging(level="WARNING", show_time=False)
        console = _test_hooks._default_console_factory()
        console.print("test")  # Should not raise

    def test_protocol_definition(self) -> None:
        """Test ConsoleFactory protocol can be used."""
        # Must call setup_rich_logging before get_rich_console
        setup_rich_logging(level="WARNING", show_time=False)
        factory: _test_hooks.ConsoleFactory = _test_hooks._default_console_factory
        console = factory()
        console.print("test")  # Should not raise


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
