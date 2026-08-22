"""Tests for discover_datasets main module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from platform_core.rich_logging import (
    setup_rich_logging,
)
from scripts.discover_datasets.main import (
    ParsedArgs,
    main,
    parse_args,
    run,
)

from scripts.discover_datasets import _test_hooks
from tests.scripts._discover_datasets_fixtures import (
    _get_test_console,
    _reset_hooks_impl,
)

_reset_hooks = pytest.fixture(_reset_hooks_impl)


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
