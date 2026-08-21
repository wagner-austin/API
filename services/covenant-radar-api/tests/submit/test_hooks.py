"""Tests for submit hooks module.

Tests the dependency injection hooks for console, project root, and registry.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from scripts.submit._hooks import (
    _default_console_factory,
    _default_project_root,
    _default_registry_factory,
    _RichConsoleAdapter,
    get_console,
    get_project_root,
    get_registry,
)

from .conftest import get_captured_console


class TestConsoleHook:
    """Tests for console hook functionality."""

    def test_get_console_returns_protocol(self) -> None:
        """Test that get_console returns a ConsoleProtocol."""
        console = get_console()
        console.write("protocol test")
        captured = get_captured_console()
        assert len(captured.messages) == 1

    def test_console_write_captures_messages(self) -> None:
        """Test that console write captures messages."""
        console = get_console()
        console.write("test message")

        captured = get_captured_console()
        assert len(captured.messages) == 1
        assert captured.messages[0] == "test message"


class TestRichConsoleAdapter:
    """Tests for RichConsoleAdapter."""

    def test_rich_console_adapter_write(self) -> None:
        """Test RichConsoleAdapter write method."""
        from platform_core.rich_logging import setup_rich_logging

        # Set up rich logging before using the adapter
        setup_rich_logging()

        adapter = _RichConsoleAdapter()
        # Call write method - it will call Rich console
        # We just verify no exception is raised
        adapter.write("adapter test")


class TestDefaultConsoleFactory:
    """Tests for default console factory."""

    def test_default_console_factory_returns_writable_console(self) -> None:
        """Test that factory returns a console that can write."""
        from platform_core.rich_logging import setup_rich_logging

        # Set up rich logging before using the console factory
        setup_rich_logging()

        console = _default_console_factory()
        # Should be able to call write without exception
        console.write("factory test")
        # Verify it has the write method from the protocol
        write_method = console.write
        assert callable(write_method)


class TestProjectRootHook:
    """Tests for project root hook functionality."""

    def test_get_project_root_returns_path(self) -> None:
        """Test that get_project_root returns a Path."""
        root = get_project_root()
        assert root.exists()
        assert root.is_dir()

    def test_default_project_root_returns_valid_path(self) -> None:
        """Test that default project root hook returns a valid path."""
        root = _default_project_root()
        assert root.exists()
        assert root.is_dir()
        # Should be the covenant-radar-api directory
        assert root.name == "covenant-radar-api"


class TestRegistryHook:
    """Tests for registry hook functionality."""

    def test_get_registry_returns_registry_with_get_method(self) -> None:
        """Test that get_registry returns a registry with get method."""
        registry = get_registry()
        # Verify it has the get method
        get_method = registry.get
        assert callable(get_method)
        # Verify it can retrieve a backend
        backend = registry.get("lightgbm")
        assert backend.backend_name() == "lightgbm"

    def test_default_registry_factory_returns_registry_with_backends(self) -> None:
        """Test that default registry factory returns registry with backends."""
        registry = _default_registry_factory()
        # Verify it has the get method
        get_method = registry.get
        assert callable(get_method)
        # Default registry should have backends like lightgbm
        backend = registry.get("lightgbm")
        assert backend.backend_name() == "lightgbm"
