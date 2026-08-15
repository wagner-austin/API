"""Tests for the scripts.guard entrypoint.

Fakes are installed by rebinding symbols on ``scripts._test_hooks`` and
restoring them afterwards, so tests never scan the real monorepo and never
patch module attributes outside the hooks module.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.guard import _find_monorepo_root, main

from scripts import _test_hooks


class _FakeLoader:
    """Loads a fake orchestrator that reports a fixed exit code."""

    def __init__(self, exit_code: int) -> None:
        """Record the exit code the fake orchestrator will report.

        Args:
            exit_code: Code the fake ``run_for_project`` returns.
        """
        self._exit_code = exit_code

    def __call__(self, monorepo_root: Path) -> _test_hooks.RunForProjectProtocol:
        """Return the fake orchestrator.

        Args:
            monorepo_root: Ignored; present to match the real signature.

        Returns:
            A ``run_for_project`` that reports the recorded exit code.
        """
        exit_code = self._exit_code

        def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
            return exit_code

        return _run_for_project


class _FakeIsDir:
    """Reports every path as a directory, so the root search stops at once."""

    def __call__(self, path: Path) -> bool:
        """Report the path as a directory.

        Args:
            path: Ignored; present to match the real signature.

        Returns:
            Always True.
        """
        return True


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore every guard hook to its real implementation after each test."""
    original_is_dir = _test_hooks.is_dir
    original_get_script_path = _test_hooks.get_script_path
    original_load_orchestrator = _test_hooks.load_orchestrator
    original_script_path = _test_hooks._SCRIPT_PATH
    yield
    _test_hooks.is_dir = original_is_dir
    _test_hooks.get_script_path = original_get_script_path
    _test_hooks.load_orchestrator = original_load_orchestrator
    _test_hooks._SCRIPT_PATH = original_script_path


def _install_fakes(exit_code: int = 0) -> None:
    """Install fakes for the hooks the guard entrypoint reaches.

    Args:
        exit_code: Code the fake orchestrator reports.
    """
    _test_hooks.is_dir = _FakeIsDir()
    _test_hooks.load_orchestrator = _FakeLoader(exit_code)


# ── _find_monorepo_root ────────────────────────────────────────────


def test_find_monorepo_root_returns_directory_containing_libs(tmp_path: Path) -> None:
    """The search stops at the first ancestor holding a 'libs' directory."""
    (tmp_path / "libs").mkdir()
    nested = tmp_path / "services" / "project"
    nested.mkdir(parents=True)

    assert _find_monorepo_root(nested) == tmp_path


def test_find_monorepo_root_raises_when_libs_is_absent(tmp_path: Path) -> None:
    """Reaching the filesystem root without finding 'libs' is an error."""
    with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
        _find_monorepo_root(tmp_path)


def test_find_monorepo_root_uses_the_is_dir_hook(tmp_path: Path) -> None:
    """The search reaches the filesystem through the hook, not Path directly."""
    _test_hooks.is_dir = _FakeIsDir()
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)

    assert _find_monorepo_root(nested) == nested


# ── main ───────────────────────────────────────────────────────────


def test_main_reports_the_orchestrator_exit_code(tmp_path: Path) -> None:
    """main returns whatever the orchestrator reported."""
    _install_fakes()

    assert main(["--root", str(tmp_path)]) == 0


def test_main_prints_exit_code_with_long_verbose_flag(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The --verbose flag prints the exit code."""
    _install_fakes()

    rc = main(["--root", str(tmp_path), "--verbose"])

    assert rc == 0
    assert capsys.readouterr().out.endswith("guard_exit_code code=0\n")


def test_main_prints_exit_code_with_short_verbose_flag(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The -v flag prints the exit code."""
    _install_fakes()

    rc = main(["--root", str(tmp_path), "-v"])

    assert rc == 0
    assert capsys.readouterr().out.endswith("guard_exit_code code=0\n")


def test_main_prints_nonzero_exit_code_when_verbose(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A failing guard run reports its nonzero code."""
    _install_fakes(exit_code=7)

    rc = main(["--root", str(tmp_path), "--verbose"])

    assert rc == 7
    assert capsys.readouterr().out.endswith("guard_exit_code code=7\n")


def test_main_ignores_unrecognised_arguments(tmp_path: Path) -> None:
    """Unknown flags are skipped rather than rejected."""
    _install_fakes()

    assert main(["--root", str(tmp_path), "--unknown-flag"]) == 0


def test_main_falls_back_to_project_root_without_root_override() -> None:
    """Without --root the guard targets the project the script lives in."""
    _install_fakes()

    assert main([]) == 0


def test_main_reads_process_arguments_when_argv_is_none(tmp_path: Path) -> None:
    """Passing None reads sys.argv."""
    _install_fakes()
    original_argv = sys.argv
    sys.argv = ["guard", "--root", str(tmp_path)]
    try:
        assert main(None) == 0
    finally:
        sys.argv = original_argv


def test_guard_module_runs_as_main(tmp_path: Path) -> None:
    """The module raises SystemExit with the guard's exit code when run."""
    original_argv = sys.argv
    sys.argv = ["guard", "--root", str(tmp_path)]
    try:
        if "scripts.guard" in sys.modules:
            del sys.modules["scripts.guard"]
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(
                str(Path(__file__).resolve().parents[2] / "scripts" / "guard.py"),
                run_name="__main__",
            )
        assert exc.value.code == 0
    finally:
        sys.argv = original_argv


# ── hook implementations ───────────────────────────────────────────


def test_real_is_dir_distinguishes_files_from_directories(tmp_path: Path) -> None:
    """The real is_dir hook reports directories and only directories."""
    a_file = tmp_path / "file.txt"
    a_file.write_text("contents", encoding="utf-8")

    assert _test_hooks.is_dir(tmp_path)
    assert not _test_hooks.is_dir(a_file)


def test_real_get_script_path_returns_the_recorded_path(tmp_path: Path) -> None:
    """set_script_path records the path get_script_path returns."""
    recorded = tmp_path / "guard.py"
    _test_hooks.set_script_path(recorded)

    assert _test_hooks.get_script_path() == recorded


def test_real_get_script_path_raises_when_unset() -> None:
    """Reading the script path before it is set is an error."""
    _test_hooks._SCRIPT_PATH = None

    with pytest.raises(RuntimeError, match="Script path not set"):
        _test_hooks.get_script_path()


def test_real_load_orchestrator_imports_the_monorepo_orchestrator() -> None:
    """The real loader returns the orchestrator's run_for_project."""
    project_root = Path(__file__).resolve().parents[2]
    monorepo_root = _find_monorepo_root(project_root)

    run_for_project = _test_hooks.load_orchestrator(monorepo_root)

    assert callable(run_for_project)
