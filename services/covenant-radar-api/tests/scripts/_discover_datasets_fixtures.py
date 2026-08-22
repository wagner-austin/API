"""Shared fixtures and helpers for test_discover_datasets_main splits."""

from __future__ import annotations

from collections.abc import Generator

from platform_core.rich_logging import (
    RichConsoleProtocol,
    RichRenderableProtocol,
)
from scripts.discover_datasets.types import DiscoveredDataset

from scripts.discover_datasets import _test_hooks


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


def _reset_hooks_impl() -> Generator[None, None, None]:
    """Reset hooks after test."""
    global _test_console
    _test_console = FakeConsole()

    orig_console_factory = _test_hooks.console_factory
    _test_hooks.console_factory = _test_console_factory

    yield

    _test_hooks.console_factory = orig_console_factory
    _test_console = None


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
