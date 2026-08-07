"""Tests for the shim / re-export guard rule.

The rule exists because the no-shims standard was the one coding
standard with no enforcing artifact, and an unenforced rule rots — the
same way the 400-600 line ceiling went from a 40-file backlog to 77
while it was documented-but-unchecked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.shim_rules import (
    evaluate,
    find_legacy_markers,
    find_reexports,
    run_shim_rules,
)


def _module(root: Path, relative: str, source: str) -> Path:
    """Write one module inside a fake project tree.

    Args:
        root: Fake project root.
        relative: Path relative to the root, POSIX separated.
        source: Module source text.

    Returns:
        Path to the created module.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


class TestFindLegacyMarkers:
    """Tests for the legacy-vocabulary scan."""

    @pytest.mark.parametrize(
        "line",
        [
            "# Backward-compatible alias",
            "TEAM = TEAM  # back compat",
            '"""Deprecated helper."""',
            "# legacy function name",
            "# kept for signature compatibility",
            "# kept for API compatibility",
            "# retained for compatibility with the old caller",
        ],
    )
    def test_shim_vocabulary_is_flagged(self, line: str) -> None:
        """Each phrase that announces a shim is reported."""
        assert find_legacy_markers(line) != []

    @pytest.mark.parametrize(
        "line",
        [
            "# Backward contamination guard for radar isolation windows.",
            "compat_score = 3",
            "# walks the path backwards to find the first sighting",
        ],
    )
    def test_innocent_prose_is_not_flagged(self, line: str) -> None:
        """Words that merely resemble the vocabulary are left alone.

        ``validate/archive.py`` really does have a "backward
        contamination guard"; flagging it would force an exclusion,
        and exclusions are what this project refuses.
        """
        assert find_legacy_markers(line) == []

    def test_the_reported_line_number_is_one_based(self) -> None:
        """A marker on the second line reports line 2."""
        assert find_legacy_markers("clean\n# legacy\n") == [(2, "legacy")]


class TestFindReexports:
    """Tests for the module-level alias scan."""

    def test_self_named_alias_is_flagged(self) -> None:
        """``X = X`` exists only to re-export an imported name."""
        source = "from other import TEAM_NAMES\n\nTEAM_NAMES = TEAM_NAMES\n"
        hits = find_reexports(source)
        assert len(hits) == 1
        assert "re-exports its own name" in hits[0][1]

    def test_renamed_reexport_in_all_is_flagged(self) -> None:
        """``NEW = IMPORTED`` plus ``__all__`` membership is a re-export."""
        source = 'from other import RADAR_COST\n\nFUEL_COST = RADAR_COST\n__all__ = ["FUEL_COST"]\n'
        hits = find_reexports(source)
        assert len(hits) == 1
        assert "import 'RADAR_COST' where it is used" in hits[0][1]

    def test_private_rename_outside_all_is_not_flagged(self) -> None:
        """An unexported local binding is a naming choice, not a re-export.

        The rule targets the export surface: re-publishing someone
        else's symbol under a new name. A module-private binding is
        caught by review, not by this rule, because flagging it would
        also flag every legitimate local constant.
        """
        source = 'from other import TICK\n\n_local = TICK\n__all__ = ["something_else"]\n'
        assert find_reexports(source) == []

    def test_assignment_from_a_local_name_is_not_flagged(self) -> None:
        """Aliasing a name defined in this module is not a re-export."""
        source = 'def build() -> int:\n    return 1\n\nmake = build\n__all__ = ["make"]\n'
        assert find_reexports(source) == []

    def test_non_name_assignments_are_ignored(self) -> None:
        """Only plain ``NAME = NAME`` bindings are candidates."""
        source = 'from other import TICK\n\nVALUES = [TICK]\n__all__ = ["VALUES"]\n'
        assert find_reexports(source) == []


class TestEvaluate:
    """Tests for the whole-tree scan."""

    def test_clean_tree_has_no_violations(self, tmp_path: Path) -> None:
        """A project with neither markers nor re-exports passes."""
        _module(tmp_path, "src/pkg/clean.py", "VALUE = 1\n")
        assert evaluate(tmp_path) == []

    def test_absent_roots_are_skipped(self, tmp_path: Path) -> None:
        """A tree without src or scripts has nothing to scan."""
        assert evaluate(tmp_path) == []

    def test_pycache_is_never_scanned(self, tmp_path: Path) -> None:
        """Compiled artifacts cannot introduce violations."""
        _module(tmp_path, "src/pkg/__pycache__/stale.py", "# legacy\n")
        assert evaluate(tmp_path) == []

    def test_scripts_are_scanned_too(self, tmp_path: Path) -> None:
        """The rule covers scripts, not just the package."""
        _module(tmp_path, "scripts/tool.py", "# deprecated\n")
        violations = evaluate(tmp_path)
        assert len(violations) == 1
        assert violations[0].endswith("scripts/tool.py:1 legacy marker 'deprecated'")

    def test_test_hooks_modules_keep_their_di_aliases(self, tmp_path: Path) -> None:
        """The ``_test_hooks`` binding IS the injection seam.

        Binding an imported implementation to a patchable module
        attribute is how this codebase does DI, so the alias is the
        mechanism rather than a re-export. The exemption is the module
        kind, not a list of blessed symbols.
        """
        source = (
            "from real import gather_intel as _real\n\n"
            'gather_intel = _real\n__all__ = ["gather_intel"]\n'
        )
        _module(tmp_path, "src/pkg/_test_hooks.py", source)
        _module(tmp_path, "src/pkg/_test_hooks/inner.py", source)
        assert evaluate(tmp_path) == []

    def test_ordinary_module_does_not_get_the_seam_exemption(self, tmp_path: Path) -> None:
        """The same alias outside a hooks module is still a re-export."""
        source = (
            "from real import gather_intel as _real\n\n"
            'gather_intel = _real\n__all__ = ["gather_intel"]\n'
        )
        _module(tmp_path, "src/pkg/ordinary.py", source)
        assert len(evaluate(tmp_path)) == 1


class TestRunShimRules:
    """Tests for the guard entry point."""

    def test_violation_is_counted_and_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Each violation prints one tagged line and counts once."""
        _module(tmp_path, "src/pkg/bad.py", "# legacy\n")
        assert run_shim_rules(tmp_path) == 1
        out = capsys.readouterr().out
        assert out.startswith("shim_violation ")
        assert "legacy marker" in out

    def test_clean_tree_is_silent(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A conforming tree prints nothing."""
        _module(tmp_path, "src/pkg/clean.py", "VALUE = 1\n")
        assert run_shim_rules(tmp_path) == 0
        assert capsys.readouterr().out == ""


class TestRealRepository:
    """The rule must hold against the real tree."""

    def test_repository_is_free_of_shims(self) -> None:
        """No back-compat marker and no re-export survives in src or scripts."""
        repo_root = Path(__file__).resolve().parents[2]
        assert evaluate(repo_root) == []
