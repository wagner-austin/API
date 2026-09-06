"""Tests for binding physics claims against a given tree.

Separated from the per-kind checker tests by role: these are about
WHICH TREE the rule reads -- source versus import, working tree versus
a published revision -- rather than about what any one claim kind
verifies. The two grew past the 600-line ceiling together.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

import pytest
from scripts.physics_claims import (
    _exported_names,
    _import_claim_module,
    _module_level_names,
    _symbol_is_exported,
    run_physics_claim_rules,
)

from tests.scripts._physics_claim_fixtures import (
    FIXTURE_MODULE,
    FIXTURE_PACKAGE,
    _claims_page,
    _write_page,
)

#: The fixture packages are rooted at the package directory, not under
#: src/ -- they are test doubles, not shipped modules.
_FIXTURE_ROOT = Path(__file__).resolve().parents[2]


class TestReadingExportsFromSource:
    """``__all__`` is read, not imported, so the rule can describe a tree."""

    def test_a_plain_list_is_read(self) -> None:
        assert _exported_names('__all__ = ["a", "b"]\n') == ["a", "b"]

    def test_an_annotated_empty_tuple_is_read(self) -> None:
        """`ledger.outcome` spells it this way, and reading only the bare
        form reported it as undeclared when it declares an empty one."""
        assert _exported_names("__all__: tuple[str, ...] = ()\n") == []

    def test_a_module_without_all_reads_as_none(self) -> None:
        assert _exported_names("x = 1\n") is None

    def test_a_computed_all_reads_as_none_rather_than_a_guess(self) -> None:
        """Guessing at a name a parser cannot see would bind a claim to a
        symbol that may not exist."""
        assert _exported_names("__all__ = sorted(_REGISTRY)\n") is None

    def test_a_non_string_entry_reads_as_none(self) -> None:
        assert _exported_names("__all__ = [1]\n") is None


class TestReadingModuleLevelNames:
    """The source-level reading of ``hasattr``, which it replaces."""

    def test_every_binding_form_is_found(self) -> None:
        source = (
            "import os\n"
            "from x import y as z\n"
            "A = 1\n"
            "B: int = 2\n"
            "def f() -> None: ...\n"
            "class C: ...\n"
        )

        assert _module_level_names(source) == frozenset({"os", "z", "A", "B", "f", "C"})

    def test_a_name_bound_only_inside_a_function_is_not_module_level(self) -> None:
        assert "inner" not in _module_level_names("def f() -> None:\n    inner = 1\n")


class TestBindingAgainstTheCommittedTree:
    """The gate must describe the revision that is published, not the one
    that happens to be checked out with edits in it.

    A claim committed ahead of the code it names is invisible to a rule
    that reads the wiki from a tree and the symbols from the installed
    package: the pair it compares exists in no revision. These two tests
    pin both directions of that.
    """

    @staticmethod
    def _extract(revision: str, into: Path) -> Path:
        """Materialise one revision of this package, read-only.

        `git archive` is used rather than `git worktree add`, which needs
        the index and fails outright when another process holds it.

        Args:
            revision: Any commit-ish.
            into: Directory to extract beneath.

        Returns:
            The extracted package root.
        """
        package = Path(__file__).resolve().parents[2]
        repo_root = package.parents[1]
        relative = package.relative_to(repo_root).as_posix()
        archive = subprocess.run(
            ["git", "archive", revision, relative],
            cwd=repo_root,
            capture_output=True,
            check=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(into, filter="data")
        return into / relative

    def test_the_committed_tree_binds_green(self, tmp_path: Path) -> None:
        """HEAD must be self-consistent, which is the property CI checks
        and a working-tree run cannot see."""
        extracted = self._extract("HEAD", tmp_path)
        assert (extracted / "wiki" / "pages").is_dir()
        assert (extracted / "src").is_dir()

        assert run_physics_claim_rules(extracted) == 0

    def test_a_claim_ahead_of_its_code_is_caught(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The positive control, and the reason the rule reads source.

        This is the shape three sessions measured at `7206cc00` on
        2026-09-06: a page committed naming a symbol its module did not
        yet carry. It is BUILT rather than extracted from that revision
        because CI checks this package out shallow -- `tankpitbot.yml`
        takes actions/checkout's default depth of 1 -- so a historical
        commit-ish resolves on a developer's clone and errors on the
        runner. A control that runs only where the history happens to
        be is not a control.

        Before the rule read source this scored 0: the page came from
        the tree under check and the symbol from the editable install,
        a pair no commit contains.
        """
        claims = (
            f'{{"claims": ['
            f'{{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},'
            f'{{"id": "double", "code": "{FIXTURE_MODULE}:double",'
            ' "formula": "2 * value", "probes": [{"args": [2], "expect": 4}]},'
            f'{{"id": "ahead", "code": "{FIXTURE_MODULE}:NOT_YET_WRITTEN", "value": 1}}'
            "]}"
        )
        _write_page(tmp_path, "client-commands.md", _claims_page(claims))

        count = run_physics_claim_rules(
            tmp_path,
            package_name=FIXTURE_MODULE,
            source_root=_FIXTURE_ROOT,
        )
        out = capsys.readouterr().out

        assert count == 1
        assert (
            "physics_claim_violation client-commands.md#ahead: "
            f"'NOT_YET_WRITTEN' not found in {FIXTURE_MODULE}" in out
        )


class TestTheTwoTreesDisagreeing:
    """Reading source and importing can now answer differently.

    Before the rule read source, both halves came from the installed
    package and could not disagree. Pointed at another revision they
    can, and each way round is a different violation.
    """

    @staticmethod
    def _tree(root: Path, body: str) -> Path:
        """Write a one-module package into a synthetic source tree.

        Args:
            root: Directory to build the tree under.
            body: Source of the package's ``facts`` module.

        Returns:
            The source root the package is rooted at.
        """
        package = root / "pkg"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("__all__: tuple[str, ...] = ()\n", encoding="utf-8")
        (package / "facts.py").write_text(body, encoding="utf-8")
        return root

    def test_a_module_this_tree_has_but_python_cannot_import_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The ordinary case for a checked-out revision: the tree holds
        the module, the interpreter has no such package installed."""
        source_root = self._tree(tmp_path / "src", '__all__ = ["X"]\nX = 1\n')
        _write_page(
            tmp_path,
            "bad.md",
            _claims_page('{"claims": [{"id": "x", "code": "pkg.facts:X", "value": 1}]}'),
        )

        count = run_physics_claim_rules(tmp_path, package_name="pkg.facts", source_root=source_root)

        out = capsys.readouterr().out
        assert count == 1
        assert "physics_claim_violation bad.md#x: module 'pkg.facts' does not import" in out

    def test_a_symbol_this_tree_has_and_the_installed_one_lacks_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Checking an OLDER revision against a newer install: the claim
        binds in the tree under check and the live module has since lost
        the symbol. Source alone would call this green."""
        shadow = tmp_path / "src" / FIXTURE_PACKAGE.replace(".", "/")
        shadow.mkdir(parents=True)
        (shadow / "__init__.py").write_text("__all__: tuple[str, ...] = ()\n", encoding="utf-8")
        (shadow / "facts.py").write_text('__all__ = ["GONE"]\nGONE = 1\n', encoding="utf-8")
        _write_page(
            tmp_path,
            "bad.md",
            _claims_page(
                f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:GONE", "value": 1}}]}}'
            ),
        )

        count = run_physics_claim_rules(
            tmp_path, package_name=FIXTURE_MODULE, source_root=tmp_path / "src"
        )

        out = capsys.readouterr().out
        assert count >= 1
        assert f"physics_claim_violation bad.md#x: 'GONE' not found in {FIXTURE_MODULE}" in out


class TestTheHelpersDirectly:
    """Two arms the rule reaches only through a tree it is not given."""

    def test_importing_a_module_that_does_not_exist_is_reported(self) -> None:
        module, violations = _import_claim_module("no.such.module", "page#id")

        assert module is None
        assert violations == ["page#id: module 'no.such.module' does not import"]

    def test_a_symbol_in_a_module_absent_from_the_tree_is_not_exported(
        self, tmp_path: Path
    ) -> None:
        assert _symbol_is_exported("absent.module", "NAME", tmp_path) is False

    def test_a_subscripted_annotation_target_binds_no_module_level_name(self) -> None:
        """`_REGISTRY["k"]: int = 1` annotates a subscript, not a name;
        reading `.target.id` off it unconditionally would raise."""
        assert _module_level_names('_R = {}\n_R["k"]: int = 1\n') == frozenset({"_R"})
