"""Tests for discover_datasets __init__ module."""

from __future__ import annotations

import runpy
import tempfile
from collections.abc import Generator

import pytest
from platform_core.logging import RichConsoleProtocol, RichRenderableProtocol
from scripts.discover_datasets import _test_hooks

from scripts import discover_datasets

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


_test_console: FakeConsole | None = None


def _test_console_factory() -> RichConsoleProtocol:
    """Factory that returns the test console."""
    global _test_console
    if _test_console is None:
        _test_console = FakeConsole()
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
# Tests
# =============================================================================


class TestDiscoverDatasetsInit:
    """Tests for discover_datasets package init."""

    def test_exports_main(self) -> None:
        """Test that main is exported from package."""
        main_fn = discover_datasets.main
        assert main_fn.__module__ == "scripts.discover_datasets.main"

    def test_all_exports(self) -> None:
        """Test __all__ contains expected exports."""
        assert "main" in discover_datasets.__all__


class TestDiscoverDatasetsMainModule:
    """Tests for discover_datasets __main__ module."""

    def test_dunder_main_import(self) -> None:
        """__main__ module can be imported."""
        from scripts.discover_datasets import __main__ as dunder_main

        # The import triggers line 3 coverage
        # Verify module name is the expected value
        assert dunder_main.__name__ == "scripts.discover_datasets.__main__"

    def test_main_module_executes(self, _reset_hooks: None) -> None:
        """__main__ module's if __name__ == '__main__' block is covered.

        Uses runpy.run_module to execute the module in a way that triggers
        the __name__ == '__main__' guard, which allows coverage to track
        line 6 execution.
        """
        import sys
        from types import ModuleType

        # Clear module from sys.modules to avoid runpy warning about
        # module already being imported
        modules_to_clear = [k for k in sys.modules if k.startswith("scripts.discover_datasets")]
        saved_modules: dict[str, ModuleType] = {}
        for mod in modules_to_clear:
            saved_modules[mod] = sys.modules.pop(mod)

        with tempfile.TemporaryDirectory() as tmpdir:
            original_argv = sys.argv
            sys.argv = [
                "scripts.discover_datasets",
                "--external-dir",
                str(tmpdir),
            ]
            try:
                with pytest.raises(SystemExit) as exc_info:
                    runpy.run_module(
                        "scripts.discover_datasets", run_name="__main__", alter_sys=True
                    )
                assert exc_info.value.code == 0
            finally:
                sys.argv = original_argv
                # Restore modules
                sys.modules.update(saved_modules)
