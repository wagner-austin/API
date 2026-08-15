"""Tests for scripts.guard entrypoint.

Fakes are installed by rebinding symbols on ``scripts._test_hooks`` and restoring
them afterwards, so tests never scan the real monorepo.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.guard import _find_monorepo_root, main

from scripts import _test_hooks


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


def _install_fake_guard_hooks(tmp_path: Path, exit_code: int = 0) -> None:
    """Install fakes for the hooks the guard entrypoint reaches.

    Args:
        tmp_path: Unused; kept so call sites read as scoped to a temp tree.
        exit_code: Code the fake orchestrator reports.
    """
    _test_hooks.is_dir = _FakeIsDir()
    _test_hooks.load_orchestrator = _FakeLoader(exit_code)


@pytest.fixture(autouse=True)
def _restore_guard_hooks() -> Generator[None, None, None]:
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


def test_guard_entrypoint_runs_as_main(tmp_path: Path) -> None:
    """Guard module can be run as __main__."""
    _install_fake_guard_hooks(tmp_path)

    orig_argv = sys.argv
    sys.argv = ["guard", "--root", str(tmp_path)]

    if "scripts.guard" in sys.modules:
        del sys.modules["scripts.guard"]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(
            str(Path(__file__).resolve().parents[0].parent / "scripts" / "guard.py"),
            run_name="__main__",
        )
    code = exc.value.code if isinstance(exc.value.code, int) else 0
    assert code == 0

    sys.argv = orig_argv


def test_main_with_verbose_flag(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Guard main runs with verbose flag."""
    _install_fake_guard_hooks(tmp_path)
    rc = main(["--root", str(tmp_path), "--verbose"])
    captured = capsys.readouterr()
    assert captured.out.endswith(f"guard_exit_code code={rc}\n")
    assert rc == 0


def test_main_with_root_override(tmp_path: Path) -> None:
    """Guard main runs with root override."""
    _install_fake_guard_hooks(tmp_path)
    rc = main(["--root", str(tmp_path)])
    assert rc == 0


def test_main_with_short_verbose_flag(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Guard main runs with short verbose flag."""
    _install_fake_guard_hooks(tmp_path)
    rc = main(["--root", str(tmp_path), "-v"])
    captured = capsys.readouterr()
    assert captured.out.endswith(f"guard_exit_code code={rc}\n")
    assert rc == 0


def test_main_with_unknown_arg(tmp_path: Path) -> None:
    """Guard main ignores unknown arguments."""
    _install_fake_guard_hooks(tmp_path)
    rc = main(["--root", str(tmp_path), "--unknown-flag"])
    assert rc == 0


def test_find_monorepo_root_raises_when_not_found(tmp_path: Path) -> None:
    """_find_monorepo_root raises RuntimeError when root not found."""
    start = Path(tmp_path.anchor) / "tankpit-guard-missing-root" / "nested"
    with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
        _find_monorepo_root(start)


def test_find_monorepo_root_finds_libs_dir(tmp_path: Path) -> None:
    """_find_monorepo_root finds directory with libs folder."""
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()
    nested = tmp_path / "services" / "project"
    nested.mkdir(parents=True)
    result = _find_monorepo_root(nested)
    assert result == tmp_path


def test_real_load_orchestrator_imports_the_monorepo_orchestrator() -> None:
    """The real loader returns the orchestrator's run_for_project."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    monorepo_root = _find_monorepo_root(project_root)
    run_for_project = _test_hooks.load_orchestrator(monorepo_root)
    assert callable(run_for_project)


def test_verbose_prints_nonzero_exit_code(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Verbose flag prints nonzero exit code."""

    _install_fake_guard_hooks(tmp_path, exit_code=7)

    rc = main(["--root", str(tmp_path), "--verbose"])

    assert rc == 7
    assert capsys.readouterr().out.endswith("guard_exit_code code=7\n")


def test_main_fails_on_contract_rule_violation(tmp_path: Path) -> None:
    """Contract-rule violations flip a passing guard run to rc=1."""
    _install_fake_guard_hooks(tmp_path)
    facts_dir = tmp_path / "src" / "tankpit_bot" / "facts"
    facts_dir.mkdir(parents=True)
    (facts_dir / "mutators.py").write_text(
        "def apply_observation(*, value: int) -> int:\n    return value\n",
        encoding="utf-8",
    )
    rc = main(["--root", str(tmp_path)])
    assert rc == 1


def test_main_keeps_orchestrator_rc_over_contract_rc(tmp_path: Path) -> None:
    """A nonzero orchestrator rc is not overwritten by contract violations."""

    _install_fake_guard_hooks(tmp_path, exit_code=5)
    facts_dir = tmp_path / "src" / "tankpit_bot" / "facts"
    facts_dir.mkdir(parents=True)
    (facts_dir / "mutators.py").write_text(
        "def apply_observation(*, value: int) -> int:\n    return value\n",
        encoding="utf-8",
    )
    rc = main(["--root", str(tmp_path)])
    assert rc == 5


def _write_bad_physics_claim(project_root: Path) -> None:
    """Plant a wiki claim block that contradicts the real physics package.

    Args:
        project_root: Fake project root to receive the wiki page.
    """
    pages_dir = project_root / "wiki" / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "economy.md").write_text(
        "# Economy\n\n```json claims\n"
        '{"claims": [{"id": "walk-cost",'
        ' "code": "tankpit_bot.physics.costs:WALK_COST_PER_TILE",'
        ' "value": 2}]}\n'
        "```\n",
        encoding="utf-8",
    )


def test_main_fails_on_physics_claim_violation(tmp_path: Path) -> None:
    """Physics-claim violations flip a passing guard run to rc=1."""
    _install_fake_guard_hooks(tmp_path)
    _write_bad_physics_claim(tmp_path)
    rc = main(["--root", str(tmp_path)])
    assert rc == 1


def test_main_keeps_orchestrator_rc_over_physics_rc(tmp_path: Path) -> None:
    """A nonzero orchestrator rc is not overwritten by physics violations."""

    _install_fake_guard_hooks(tmp_path, exit_code=5)
    _write_bad_physics_claim(tmp_path)
    rc = main(["--root", str(tmp_path)])
    assert rc == 5


def test_guard_detects_violations(tmp_path: Path) -> None:
    """Test guard.main detects violations and returns non-zero exit code."""
    root = tmp_path
    src = root / "src"
    bad = src / "bad.py"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        "from typing import Any\n"
        "x: Any = 1  # type: ignore\n"
        "from typing import cast\n"
        "y = cast(int, 1)\n"
        "import contextlib\n"
        "with contextlib.suppress(Exception):\n"
        "    pass\n"
        "try:\n"
        "    1/0\n"
        "except Exception as exc:\n"
        "    raise RuntimeError('fail') from exc\n",
        encoding="utf-8",
    )
    rc = main(["--root", str(root)])
    assert rc != 0


def _write_bad_wiki(project_root: Path) -> None:
    """Plant a wiki page that violates the structure rule.

    The page carries no frontmatter at all, which ``wiki/SCHEMA.md``
    requires on every content page.

    Args:
        project_root: Fake project root to receive the wiki page.
    """
    pages_dir = project_root / "wiki" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "bare.md").write_text("# Bare page\n\nNo frontmatter.\n", encoding="utf-8")


def test_main_fails_on_wiki_structure_violation(tmp_path: Path) -> None:
    """Wiki-structure violations flip a passing guard run to rc=1."""
    _install_fake_guard_hooks(tmp_path)
    _write_bad_wiki(tmp_path)
    rc = main(["--root", str(tmp_path)])
    assert rc == 1


def test_main_keeps_orchestrator_rc_over_wiki_rc(tmp_path: Path) -> None:
    """A nonzero orchestrator rc is not overwritten by wiki violations."""

    _install_fake_guard_hooks(tmp_path, exit_code=5)
    _write_bad_wiki(tmp_path)
    rc = main(["--root", str(tmp_path)])
    assert rc == 5


# ── guard hook implementations ─────────────────────────────────────


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


def test_find_monorepo_root_returns_directory_containing_libs(tmp_path: Path) -> None:
    """The search stops at the first ancestor holding a 'libs' directory."""
    (tmp_path / "libs").mkdir()
    nested = tmp_path / "services" / "project"
    nested.mkdir(parents=True)

    assert _find_monorepo_root(nested) == tmp_path


def test_real_get_script_path_raises_when_unset() -> None:
    """Reading the script path before it is set is an error."""
    _test_hooks._SCRIPT_PATH = None

    with pytest.raises(RuntimeError, match="Script path not set"):
        _test_hooks.get_script_path()
