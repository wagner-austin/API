"""Tests for logging: test_setup_rich_logging_returns_logger."""

from __future__ import annotations

import pytest

from platform_core.logging import (
    stdlib_logging,
)


def test_setup_rich_logging_returns_logger() -> None:
    """Test setup_rich_logging returns configured root logger."""
    from platform_core.rich_logging import setup_rich_logging

    logger = setup_rich_logging(level="INFO")

    assert logger.level == stdlib_logging.INFO
    assert len(logger.handlers) == 1


def test_setup_rich_logging_with_options() -> None:
    """Test setup_rich_logging accepts all options."""
    from platform_core.rich_logging import setup_rich_logging

    logger = setup_rich_logging(
        level="DEBUG",
        show_time=False,
        show_path=True,
    )

    assert logger.level == stdlib_logging.DEBUG


def test_setup_rich_logging_silences_third_party() -> None:
    """Test setup_rich_logging sets WARNING level for noisy loggers."""
    from platform_core.rich_logging import setup_rich_logging

    setup_rich_logging(level="DEBUG")

    assert stdlib_logging.getLogger("urllib3").level == stdlib_logging.WARNING
    assert stdlib_logging.getLogger("httpx").level == stdlib_logging.WARNING
    assert stdlib_logging.getLogger("httpcore").level == stdlib_logging.WARNING
    assert stdlib_logging.getLogger("optuna").level == stdlib_logging.WARNING


def test_get_rich_console_after_setup() -> None:
    """Test get_rich_console returns console after setup."""
    from platform_core.rich_logging import (
        RichConsoleProtocol,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()

    # Verify it implements the protocol by calling print
    typed_console: RichConsoleProtocol = console
    typed_console.print("test output")  # Should not raise


def test_get_rich_console_before_setup_raises() -> None:
    """Test get_rich_console raises RuntimeError before setup."""
    import platform_core.rich_logging as rich_mod
    from platform_core.rich_logging import get_rich_console

    # Reset module state to simulate no setup
    original = rich_mod._rich_console
    rich_mod._rich_console = None

    try:
        with pytest.raises(RuntimeError) as raised:
            get_rich_console()
        assert "setup_rich_logging()" in str(raised.value)
    finally:
        rich_mod._rich_console = original


def test_create_rich_table_basic() -> None:
    """Test create_rich_table creates a table with columns and rows."""
    from platform_core.rich_logging import (
        RichTableProtocol,
        create_rich_table,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()
    table = create_rich_table(title="Test Table")

    # Verify Protocol interface by using methods
    typed_table: RichTableProtocol = table
    typed_table.add_column("Col1", style="cyan")
    typed_table.add_column("Col2", justify="right")
    typed_table.add_row("val1", "val2")

    # Verify table can be printed (tests actual functionality)
    console.print(table)


def test_create_rich_table_no_header() -> None:
    """Test create_rich_table with show_header=False."""
    from platform_core.rich_logging import create_rich_table, get_rich_console, setup_rich_logging

    setup_rich_logging()
    console = get_rich_console()
    table = create_rich_table(show_header=False)
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("k1", "v1")

    # Verify table can be printed
    console.print(table)


def test_create_rich_panel_basic() -> None:
    """Test create_rich_panel creates a panel."""
    from platform_core.rich_logging import create_rich_panel, get_rich_console, setup_rich_logging

    setup_rich_logging()
    console = get_rich_console()
    panel = create_rich_panel("[bold]Content[/bold]", title="Title")

    # Verify panel can be printed (tests actual functionality)
    console.print(panel)


def test_create_rich_panel_no_title() -> None:
    """Test create_rich_panel without title."""
    from platform_core.rich_logging import create_rich_panel, get_rich_console, setup_rich_logging

    setup_rich_logging()
    console = get_rich_console()
    panel = create_rich_panel("Simple content")

    # Verify panel can be printed
    console.print(panel)


def test_create_rich_progress_context_manager() -> None:
    """Test create_rich_progress as context manager."""
    from platform_core.rich_logging import (
        RichProgressProtocol,
        create_rich_progress,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()
    progress = create_rich_progress(console)

    # Verify Protocol interface by using all methods
    typed_progress: RichProgressProtocol = progress
    task_id_captured = -1

    with typed_progress as p:
        task_id = p.add_task("Testing...", total=10.0)
        task_id_captured = task_id
        p.update(task_id, description="Updated")
        p.advance(task_id, advance=5.0)

    # Task ID should be a valid integer (0 for first task)
    assert task_id_captured == 0


def test_create_rich_spinner_progress_context_manager() -> None:
    """Test create_rich_spinner_progress as context manager."""
    from platform_core.rich_logging import (
        RichProgressProtocol,
        create_rich_spinner_progress,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()
    progress = create_rich_spinner_progress(console)

    # Verify Protocol interface
    typed_progress: RichProgressProtocol = progress
    task_id_captured = -1

    with typed_progress as p:
        task_id = p.add_task("Spinner test...")
        task_id_captured = task_id
        p.advance(task_id)

    # Task ID should be a valid integer (0 for first task)
    assert task_id_captured == 0


def test_rich_console_print_table() -> None:
    """Test rich console can print a table."""
    from platform_core.rich_logging import create_rich_table, get_rich_console, setup_rich_logging

    setup_rich_logging()
    console = get_rich_console()
    table = create_rich_table(title="Print Test")
    table.add_column("A")
    table.add_row("value")

    # Should not raise
    console.print(table)


def test_rich_console_print_panel() -> None:
    """Test rich console can print a panel."""
    from platform_core.rich_logging import create_rich_panel, get_rich_console, setup_rich_logging

    setup_rich_logging()
    console = get_rich_console()
    panel = create_rich_panel("Panel content")

    # Should not raise
    console.print(panel)
