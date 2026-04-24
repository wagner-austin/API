"""Tests for guard check behavior."""

from __future__ import annotations

from pathlib import Path

from scripts.guard import _find_monorepo_root, _load_orchestrator, main


def test_find_monorepo_root() -> None:
    """Should find the root containing libs/ directory."""
    project_root = Path(__file__).resolve().parents[1]
    root = _find_monorepo_root(project_root)
    assert (root / "libs").is_dir()


def test_find_monorepo_root_not_found() -> None:
    """Should raise if no libs/ directory is found."""
    import pytest
    from scripts import guard

    original = guard._is_dir

    def _always_false(p: Path) -> bool:
        _ = p
        return False

    guard._is_dir = _always_false
    with pytest.raises(RuntimeError, match="monorepo root"):
        _find_monorepo_root(Path("/"))
    guard._is_dir = original


def test_load_orchestrator() -> None:
    """Should load the orchestrator from monorepo libs."""
    project_root = Path(__file__).resolve().parents[1]
    root = _find_monorepo_root(project_root)
    runner = _load_orchestrator(root)
    assert callable(runner)


def test_main_returns_exit_code() -> None:
    """main() should return a valid exit code."""
    code = main(["--verbose"])
    assert code in (0, 2)


def test_main_with_root_override() -> None:
    """main() with --root flag should work."""
    project_root = str(Path(__file__).resolve().parents[1])
    code = main(["--root", project_root, "--verbose"])
    assert code in (0, 2)


def test_main_with_unknown_arg() -> None:
    """main() ignores unknown arguments."""
    code = main(["--unknown-flag", "some-value"])
    assert code in (0, 2)
