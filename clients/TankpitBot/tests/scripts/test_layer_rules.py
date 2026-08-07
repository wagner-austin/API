"""Tests for the base-layer guard rule.

The rule states that the packages lifted clear of the remaining
import component stay clear. These pin both import spellings -- the
``from tankpit_bot import state`` form is the one a naive checker
misses, and missing it is how three cycles stayed hidden through two
passes of the layering work.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.layer_rules import BASE_LAYER, PACKAGE_ROOT, evaluate, run_layer_rules


def _package(root: Path, name: str, module: str, source: str) -> Path:
    """Write one module inside a package of a fake tree.

    Args:
        root: Fake ``src/tankpit_bot`` root.
        name: Package directory name.
        module: Module filename.
        source: Module source text.

    Returns:
        Path to the created module.
    """
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / module
    path.write_text(source, encoding="utf-8")
    return path


def _scaffold(root: Path) -> Path:
    """Create an empty directory for every declared base-layer package.

    ``evaluate`` reports a declaration whose package is gone, so a fake
    tree must carry them all or every test drowns in stale-declaration
    noise.

    Args:
        root: Fake ``src/tankpit_bot`` root.

    Returns:
        The same root, for chaining.
    """
    for name in BASE_LAYER:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


class TestEvaluate:
    """Tests for the per-package import check."""

    def test_leaf_importing_nothing_passes(self, tmp_path: Path) -> None:
        """A leaf with no tankpit_bot imports is clean."""
        _scaffold(tmp_path)
        _package(tmp_path, "types", "m.py", "import json\n")
        assert evaluate(tmp_path) == []

    def test_leaf_importing_a_package_is_reported(self, tmp_path: Path) -> None:
        """A leaf may import nothing from this codebase."""
        _scaffold(tmp_path)
        _package(tmp_path, "types", "m.py", "from tankpit_bot.state import X\n")
        _package(tmp_path, "state", "s.py", "")
        violations = evaluate(tmp_path)
        assert len(violations) == 1
        assert "imports 'state'" in violations[0]
        assert "may import nothing" in violations[0]

    def test_alias_import_form_is_caught(self, tmp_path: Path) -> None:
        """``from tankpit_bot import state`` counts as importing state.

        A checker that only follows ``from tankpit_bot.state import X``
        under-reports; this spelling is what hid three cycles.
        """
        _scaffold(tmp_path)
        _package(tmp_path, "types", "m.py", "from tankpit_bot import state\n")
        _package(tmp_path, "state", "s.py", "")
        violations = evaluate(tmp_path)
        assert len(violations) == 1
        assert "imports 'state'" in violations[0]

    def test_plain_import_form_is_caught(self, tmp_path: Path) -> None:
        """``import tankpit_bot.state`` counts too."""
        _scaffold(tmp_path)
        _package(tmp_path, "types", "m.py", "import tankpit_bot.state\n")
        _package(tmp_path, "state", "s.py", "")
        assert len(evaluate(tmp_path)) == 1

    def test_relative_import_is_ignored(self, tmp_path: Path) -> None:
        """A relative import names no package to check."""
        _scaffold(tmp_path)
        _package(tmp_path, "types", "m.py", "from . import sibling\n")
        assert evaluate(tmp_path) == []

    def test_allowed_dependency_passes(self, tmp_path: Path) -> None:
        """A package may import exactly what it declares."""
        _scaffold(tmp_path)
        _package(tmp_path, "facts", "m.py", "from tankpit_bot.contracts import C\n")
        _package(tmp_path, "contracts", "c.py", "")
        assert evaluate(tmp_path) == []

    def test_disallowed_dependency_names_the_allowance(self, tmp_path: Path) -> None:
        """The message says what the package MAY import."""
        _scaffold(tmp_path)
        _package(tmp_path, "facts", "m.py", "from tankpit_bot.state import X\n")
        _package(tmp_path, "state", "s.py", "")
        violations = evaluate(tmp_path)
        assert len(violations) == 1
        assert "may import contracts" in violations[0]

    def test_self_import_is_not_a_violation(self, tmp_path: Path) -> None:
        """A package importing its own submodule is fine."""
        _scaffold(tmp_path)
        _package(tmp_path, "types", "m.py", "from tankpit_bot.types.other import X\n")
        assert evaluate(tmp_path) == []

    def test_top_level_module_import_is_ignored(self, tmp_path: Path) -> None:
        """Only packages are checked; a top-level module is not one."""
        _scaffold(tmp_path)
        _package(tmp_path, "types", "m.py", "from tankpit_bot.terrain import X\n")
        (tmp_path / "terrain.py").write_text("", encoding="utf-8")
        assert evaluate(tmp_path) == []

    def test_missing_declared_package_is_reported(self, tmp_path: Path) -> None:
        """A declaration for a package that no longer exists is stale."""
        violations = evaluate(tmp_path)
        assert len(violations) == len(BASE_LAYER)
        assert all("does not exist" in v for v in violations)

    def test_pycache_is_skipped(self, tmp_path: Path) -> None:
        """Compiled artifacts are never scanned."""
        _scaffold(tmp_path)
        _package(tmp_path, "types", "m.py", "")
        cache = tmp_path / "types" / "__pycache__"
        cache.mkdir()
        (cache / "stale.py").write_text("from tankpit_bot.state import X\n", encoding="utf-8")
        _package(tmp_path, "state", "s.py", "")
        assert evaluate(tmp_path) == []


class TestRunLayerRules:
    """Tests for the guard entry point."""

    def test_absent_package_root_passes(self, tmp_path: Path) -> None:
        """A tree without src/tankpit_bot has nothing to check."""
        assert run_layer_rules(tmp_path) == 0

    def test_violation_is_counted_and_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Each violation prints one tagged line and counts once."""
        root = tmp_path / PACKAGE_ROOT
        for name in BASE_LAYER:
            _package(root, name, "__init__.py", "")
        _package(root, "wire", "bad.py", "from tankpit_bot.state import X\n")
        _package(root, "state", "s.py", "")
        assert run_layer_rules(tmp_path) == 1
        out = capsys.readouterr().out
        assert out.startswith("layer_violation ")
        assert "imports 'state'" in out

    def test_clean_tree_is_silent(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A conforming tree prints nothing."""
        root = tmp_path / PACKAGE_ROOT
        for name in BASE_LAYER:
            _package(root, name, "__init__.py", "")
        assert run_layer_rules(tmp_path) == 0
        assert capsys.readouterr().out == ""


class TestRealRepository:
    """The rule must hold against the real tree."""

    def test_base_layer_is_clean(self) -> None:
        """Every declared base-layer package respects its allowance."""
        repo_root = Path(__file__).resolve().parents[2]
        assert (repo_root / PACKAGE_ROOT).is_dir()
        assert evaluate(repo_root / PACKAGE_ROOT) == []
