"""The guard entry point, driven through real code with injected hooks.

Hook binding uses an explicitly typed context manager rather than a pytest
fixture: ``pytest.fixture`` is an overloaded callable, so decorating with it
produces an expression containing ``Any`` and fails strict type checking.
"""

from __future__ import annotations

import runpy
import sys
import tempfile
from pathlib import Path
from types import TracebackType

import pytest
from scripts import _test_hooks
from scripts.guard import main

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FAKE_MONOREPO = Path("/monorepo")


class _StubbedOrchestrator:
    """Stand in for the shared orchestrator and record what it was asked to check.

    Attributes:
        calls: One ``(monorepo_root, project_root)`` pair per invocation.
        result: Exit code every invocation returns.
    """

    def __init__(self, result: int) -> None:
        self.calls: list[tuple[Path, Path]] = []
        self.result = result
        self._original_find: _test_hooks.FindMonorepoRootProto = _test_hooks.find_monorepo_root
        self._original_load: _test_hooks.LoadOrchestratorProto = _test_hooks.load_orchestrator

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Record one orchestrator invocation.

        Args:
            monorepo_root: Repository root passed by the guard.
            project_root: Project directory passed by the guard.

        Returns:
            The configured exit code.
        """
        self.calls.append((monorepo_root, project_root))
        return self.result

    def _find(self, start: Path) -> Path:
        """Stand in for the filesystem walk.

        Args:
            start: Ignored; the stub always reports the same root.

        Returns:
            A fixed fake monorepo root.
        """
        return _FAKE_MONOREPO

    def _load(self, monorepo_root: Path) -> _test_hooks.RunForProjectProto:
        """Stand in for the dynamic import.

        Args:
            monorepo_root: Ignored; the stub always returns itself.

        Returns:
            This stub, acting as the orchestrator.
        """
        return self

    def __enter__(self) -> _StubbedOrchestrator:
        """Install both guard hooks.

        Returns:
            This stub.
        """
        self._original_find = _test_hooks.find_monorepo_root
        self._original_load = _test_hooks.load_orchestrator
        _test_hooks.find_monorepo_root = self._find
        _test_hooks.load_orchestrator = self._load
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore both guard hooks.

        Args:
            exc_type: Exception class raised in the block, if any.
            exc: Exception raised in the block, if any.
            traceback: Traceback of the raised exception, if any.
        """
        _test_hooks.find_monorepo_root = self._original_find
        _test_hooks.load_orchestrator = self._original_load


def test_checks_this_project_by_default() -> None:
    with _StubbedOrchestrator(0) as stub:
        assert main([]) == 0
    assert stub.calls == [(_FAKE_MONOREPO, _PROJECT_ROOT)]


def test_propagates_a_nonzero_orchestrator_result() -> None:
    with _StubbedOrchestrator(3) as stub:
        assert main([]) == 3
    assert len(stub.calls) == 1


def test_root_override_redirects_the_check(tmp_path: Path) -> None:
    with _StubbedOrchestrator(0) as stub:
        assert main(["--root", str(tmp_path)]) == 0
    assert stub.calls == [(_FAKE_MONOREPO, tmp_path.resolve())]


def test_unknown_arguments_are_skipped() -> None:
    with _StubbedOrchestrator(0) as stub:
        assert main(["--verbose", "extra"]) == 0
    assert stub.calls == [(_FAKE_MONOREPO, _PROJECT_ROOT)]


def test_trailing_root_without_a_value_is_skipped() -> None:
    with _StubbedOrchestrator(0) as stub:
        assert main(["--root"]) == 0
    assert stub.calls == [(_FAKE_MONOREPO, _PROJECT_ROOT)]


def test_find_monorepo_root_locates_the_real_repository() -> None:
    found = _test_hooks.find_monorepo_root(_PROJECT_ROOT)
    assert (found / "libs" / "monorepo_guards").is_dir()
    assert (found / "clients" / "RustedWarfareBot" / "pyproject.toml").is_file()


def test_find_monorepo_root_raises_when_no_libs_directory_exists() -> None:
    # Deliberately not pytest's tmp_path: the suite's basetemp is .pytest_tmp
    # inside this project, so walking upward from it finds the monorepo's real
    # libs/ and the search succeeds. The system temp directory has no such
    # ancestor.
    with tempfile.TemporaryDirectory() as outside_repo, pytest.raises(RuntimeError) as caught:
        _test_hooks.find_monorepo_root(Path(outside_repo))
    assert str(caught.value) == "monorepo root with 'libs' directory not found"


def test_load_orchestrator_returns_a_working_rule_runner(tmp_path: Path) -> None:
    monorepo_root = _test_hooks.find_monorepo_root(_PROJECT_ROOT)
    run_for_project = _test_hooks.load_orchestrator(monorepo_root)
    assert run_for_project(monorepo_root=monorepo_root, project_root=tmp_path) == 0


def test_module_entry_point_exits_with_the_guard_result() -> None:
    # `make lint` invokes this as `python -m scripts.guard`, so the __main__
    # block is a real execution path and is covered by executing it, not by
    # excluding it from coverage.
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.guard")
    sys.argv = ["guard"]
    try:
        with _StubbedOrchestrator(7), pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.guard", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.guard"] = already_imported
    assert caught.value.code == 7
