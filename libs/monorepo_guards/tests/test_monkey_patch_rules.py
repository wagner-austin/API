"""Tests for the monkey-patch ban guard rule.

The rule bans ad-hoc module attribute mutation in tests while allowing the
sanctioned forms: hooks-module targets, sys/os, resetting to None, and
save-and-restore. Each of those allowances is exercised here, because a rule
that is too strict pushes authors toward misleading aliases and a rule that is
too loose lets real cross-test pollution through.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.monkey_patch_rules import MonkeyPatchBanRule


def _write_test_file(tmp_path: Path, content: str, name: str = "test_example.py") -> Path:
    """Write content to a file inside a tests directory.

    Args:
        tmp_path: Pytest temporary directory.
        content: Python source to write.
        name: File name to use.

    Returns:
        Path to the written file.
    """
    test_dir = tmp_path / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / name
    test_file.write_text(content, encoding="utf-8")
    return test_file


def _run(paths: list[Path]) -> list[str]:
    """Run the rule and return the rendered violation lines.

    Args:
        paths: Files to check.

    Returns:
        The `line` field of each violation.
    """
    rule = MonkeyPatchBanRule()
    return [v.line for v in rule.run(paths)]


class TestBannedForms:
    """Ad-hoc module attribute mutation is reported."""

    def test_direct_attribute_assignment_is_reported(self, tmp_path: Path) -> None:
        """`module.attr = fake` with no restore is a violation."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n    othermod.thing = 1\n",
        )

        violations = _run([test_file])

        assert len(violations) == 1
        assert "othermod.thing" in violations[0]

    def test_setattr_with_literal_name_is_reported(self, tmp_path: Path) -> None:
        """`setattr(module, "attr", fake)` with no restore is a violation."""
        test_file = _write_test_file(
            tmp_path,
            'import othermod\n\n\ndef test_x() -> None:\n    setattr(othermod, "thing", 1)\n',
        )

        violations = _run([test_file])

        assert len(violations) == 1
        assert "setattr(othermod, 'thing', ...)" in violations[0]

    def test_setattr_with_variable_name_is_reported(self, tmp_path: Path) -> None:
        """A non-literal attribute name is still reported."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n    name = 'thing'\n"
            "    setattr(othermod, name, 1)\n",
        )

        violations = _run([test_file])

        assert len(violations) == 1


class TestAllowedForms:
    """The sanctioned patterns are not reported."""

    def test_hooks_module_alias_is_allowed(self, tmp_path: Path) -> None:
        """Assigning to a *_hooks alias is dependency injection, not patching."""
        test_file = _write_test_file(
            tmp_path,
            "from pkg import _test_hooks as worker_hooks\n\n\n"
            "def test_x() -> None:\n    worker_hooks.factory = 1\n",
        )

        assert _run([test_file]) == []

    def test_sys_and_os_are_allowed(self, tmp_path: Path) -> None:
        """sys/os mutation is a standard test idiom."""
        test_file = _write_test_file(
            tmp_path,
            "import sys\nimport os\n\n\ndef test_x() -> None:\n"
            "    sys.argv = []\n    os.environ = {}\n",
        )

        assert _run([test_file]) == []

    def test_assigning_none_is_allowed(self, tmp_path: Path) -> None:
        """Resetting an attribute to None is state cleanup."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n    othermod.thing = None\n",
        )

        assert _run([test_file]) == []

    def test_save_and_restore_in_one_function_is_allowed(self, tmp_path: Path) -> None:
        """A save then restore in the same scope keeps the test isolated."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n"
            "    original = othermod.thing\n"
            "    othermod.thing = 1\n"
            "    othermod.thing = original\n",
        )

        assert _run([test_file]) == []

    def test_getattr_setattr_save_and_restore_is_allowed(self, tmp_path: Path) -> None:
        """The getattr/setattr spelling of save-and-restore is recognised."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n"
            '    original = getattr(othermod, "thing")\n'
            '    setattr(othermod, "thing", 1)\n'
            '    setattr(othermod, "thing", original)\n',
        )

        assert _run([test_file]) == []

    def test_self_attribute_assignment_is_allowed(self, tmp_path: Path) -> None:
        """Assigning to self is not module mutation."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\nclass TestThing:\n"
            "    def test_x(self) -> None:\n        self.thing = othermod\n",
        )

        assert _run([test_file]) == []

    def test_unimported_name_is_not_treated_as_a_module(self, tmp_path: Path) -> None:
        """Only names bound by an import are candidate module targets."""
        test_file = _write_test_file(
            tmp_path,
            "def test_x() -> None:\n    local = object()\n    local.thing = 1\n",
        )

        assert _run([test_file]) == []


class TestConftestAwareRestores:
    """A restore living in conftest.py protects sibling test modules."""

    def test_restore_in_conftest_exempts_a_sibling_module(self, tmp_path: Path) -> None:
        """pytest applies a conftest fixture to every module beside it.

        Analysing each file alone reported correctly isolated tests as
        monkey-patching, which pushed authors toward misleading renames.
        """
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "conftest.py").write_text(
            "import othermod\n\n\ndef _reset() -> None:\n"
            "    original = othermod.thing\n"
            "    yield\n"
            "    othermod.thing = original\n",
            encoding="utf-8",
        )
        test_file = tests_dir / "test_example.py"
        test_file.write_text(
            "import othermod\n\n\ndef test_x() -> None:\n    othermod.thing = 1\n",
            encoding="utf-8",
        )

        assert _run([test_file]) == []

    def test_unrelated_conftest_restore_does_not_exempt(self, tmp_path: Path) -> None:
        """A conftest that restores a different attribute grants no exemption."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "conftest.py").write_text(
            "import othermod\n\n\ndef _reset() -> None:\n"
            "    original = othermod.unrelated\n"
            "    yield\n"
            "    othermod.unrelated = original\n",
            encoding="utf-8",
        )
        test_file = tests_dir / "test_example.py"
        test_file.write_text(
            "import othermod\n\n\ndef test_x() -> None:\n    othermod.thing = 1\n",
            encoding="utf-8",
        )

        assert len(_run([test_file])) == 1


class TestUnsupportedForms:
    """Shapes the rule cannot reason about are passed over, not misreported."""

    def test_non_setattr_call_is_ignored(self, tmp_path: Path) -> None:
        """An ordinary call is not a patch."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n    print(othermod)\n",
        )

        assert _run([test_file]) == []

    def test_setattr_with_too_few_arguments_is_ignored(self, tmp_path: Path) -> None:
        """A two-argument setattr cannot be a patch."""
        test_file = _write_test_file(
            tmp_path,
            'import othermod\n\n\ndef test_x() -> None:\n    setattr(othermod, "thing")\n',
        )

        assert _run([test_file]) == []

    def test_setattr_on_a_non_name_target_is_ignored(self, tmp_path: Path) -> None:
        """A computed target is not a module reference the rule can resolve."""
        test_file = _write_test_file(
            tmp_path,
            'import othermod\n\n\ndef test_x() -> None:\n    setattr(othermod.inner, "thing", 1)\n',
        )

        assert _run([test_file]) == []

    def test_setattr_to_none_is_allowed(self, tmp_path: Path) -> None:
        """Resetting via setattr is state cleanup, like a None assignment."""
        test_file = _write_test_file(
            tmp_path,
            'import othermod\n\n\ndef test_x() -> None:\n    setattr(othermod, "thing", None)\n',
        )

        assert _run([test_file]) == []

    def test_setattr_with_unsupported_name_expression_is_ignored(self, tmp_path: Path) -> None:
        """An attribute name that is neither a literal nor a plain name."""
        test_file = _write_test_file(
            tmp_path,
            'import othermod\n\n\ndef test_x() -> None:\n    setattr(othermod, "a" + "b", 1)\n',
        )

        assert _run([test_file]) == []

    def test_chained_attribute_assignment_is_ignored(self, tmp_path: Path) -> None:
        """`module.inner.attr = x` does not name a module directly."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n    othermod.inner.thing = 1\n",
        )

        assert _run([test_file]) == []

    def test_plain_local_assignment_is_ignored(self, tmp_path: Path) -> None:
        """An assignment with no attribute target is not a patch."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n    value = 1\n    del value\n",
        )

        assert _run([test_file]) == []


class TestPartialSaveRestore:
    """A save without a matching restore grants no exemption."""

    def test_save_of_a_different_attribute_does_not_exempt(self, tmp_path: Path) -> None:
        """Restoring one attribute does not license patching another."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n"
            "    original = othermod.other\n"
            "    othermod.thing = 1\n"
            "    othermod.other = original\n",
        )

        assert len(_run([test_file])) == 1

    def test_restore_from_an_unsaved_local_does_not_exempt(self, tmp_path: Path) -> None:
        """Assigning an arbitrary local back is not a restore."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n"
            "    othermod.thing = 1\n"
            "    othermod.thing = something_else\n",
        )

        assert len(_run([test_file])) == 2

    def test_setattr_restore_of_a_different_attribute_does_not_exempt(self, tmp_path: Path) -> None:
        """A setattr restore must match the saved attribute to count."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n"
            '    original = getattr(othermod, "other")\n'
            '    setattr(othermod, "thing", 1)\n'
            '    setattr(othermod, "other", original)\n',
        )

        assert len(_run([test_file])) == 1


class TestUnmatchedSaveRestore:
    """Save/restore bookkeeping only exempts an exact module+attribute match."""

    def test_getattr_save_with_computed_name_is_not_recorded(self, tmp_path: Path) -> None:
        """A getattr save whose attribute name is computed cannot be matched."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n"
            '    original = getattr(othermod, "a" + "b")\n'
            "    othermod.thing = 1\n"
            "    othermod.thing = original\n",
        )

        assert len(_run([test_file])) == 2

    def test_restore_onto_a_different_module_is_not_recorded(self, tmp_path: Path) -> None:
        """Writing the saved value onto another module is not a restore."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\nimport secondmod\n\n\ndef test_x() -> None:\n"
            "    original = othermod.thing\n"
            "    othermod.thing = 1\n"
            "    secondmod.thing = original\n",
        )

        assert len(_run([test_file])) == 2

    def test_setattr_restore_with_computed_name_is_not_recorded(self, tmp_path: Path) -> None:
        """A setattr restore with a computed attribute name cannot be matched."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\n\n\ndef test_x() -> None:\n"
            '    original = getattr(othermod, "thing")\n'
            '    setattr(othermod, "thing", 1)\n'
            '    setattr(othermod, "a" + "b", original)\n',
        )

        assert len(_run([test_file])) == 1

    def test_setattr_restore_onto_a_different_module_is_not_recorded(self, tmp_path: Path) -> None:
        """A setattr restore must target the module the value came from."""
        test_file = _write_test_file(
            tmp_path,
            "import othermod\nimport secondmod\n\n\ndef test_x() -> None:\n"
            '    original = getattr(othermod, "thing")\n'
            '    setattr(othermod, "thing", 1)\n'
            '    setattr(secondmod, "thing", original)\n',
        )

        assert len(_run([test_file])) == 2

    def test_setattr_on_a_non_module_name_is_ignored(self, tmp_path: Path) -> None:
        """setattr against a local object is not module mutation."""
        test_file = _write_test_file(
            tmp_path,
            'def test_x() -> None:\n    local = object()\n    setattr(local, "thing", 1)\n',
        )

        assert _run([test_file]) == []


class TestResetContainerIsolation:
    """An autouse fixture calling `X.reset()` isolates every attribute of X.

    Libraries expose their injection seam from testing.py as a container class
    (`hooks = HooksContainer`) rather than a `_test_hooks` module, so the name
    misses the suffix allowlist. Isolation comes from a conftest autouse
    fixture calling reset(), which the save-and-restore matcher cannot see.
    """

    def _write_pair(self, tmp_path: Path, conftest: str, module: str) -> Path:
        """Write a conftest and a sibling test module.

        Args:
            tmp_path: Pytest temporary directory.
            conftest: Source for tests/conftest.py.
            module: Source for tests/test_example.py.

        Returns:
            Path to the written test module.
        """
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "conftest.py").write_text(conftest, encoding="utf-8")
        test_file = tests_dir / "test_example.py"
        test_file.write_text(module, encoding="utf-8")
        return test_file

    def test_autouse_reset_in_conftest_exempts_a_sibling_module(self, tmp_path: Path) -> None:
        """The real platform_workers shape: reset in conftest, patch in module."""
        test_file = self._write_pair(
            tmp_path,
            "import pytest\n\nfrom pkg.testing import hooks\n\n\n"
            "@pytest.fixture(autouse=True)\ndef reset_hooks() -> None:\n"
            "    hooks.reset()\n    yield\n    hooks.reset()\n",
            "from pkg.testing import hooks\n\n\ndef test_x() -> None:\n    hooks.loader = 1\n",
        )

        assert _run([test_file]) == []

    def test_autouse_reset_exempts_the_setattr_spelling(self, tmp_path: Path) -> None:
        """A reset container covers setattr patches as well as assignments."""
        test_file = self._write_pair(
            tmp_path,
            "import pytest\n\nfrom pkg.testing import hooks\n\n\n"
            "@pytest.fixture(autouse=True)\ndef reset_hooks() -> None:\n    hooks.reset()\n",
            "from pkg.testing import hooks\n\n\ndef test_x() -> None:\n"
            '    setattr(hooks, "loader", 1)\n',
        )

        assert _run([test_file]) == []

    def test_autouse_reset_inline_in_the_test_module_exempts(self, tmp_path: Path) -> None:
        """The fixture need not live in a conftest to isolate the module."""
        test_file = _write_test_file(
            tmp_path,
            "import pytest\n\nfrom pkg.testing import hooks\n\n\n"
            "@pytest.fixture(autouse=True)\ndef reset_hooks() -> None:\n    hooks.reset()\n\n\n"
            "def test_x() -> None:\n    hooks.loader = 1\n",
        )

        assert _run([test_file]) == []

    def test_reset_without_autouse_does_not_exempt(self, tmp_path: Path) -> None:
        """A fixture a test must request by name proves nothing about others."""
        test_file = self._write_pair(
            tmp_path,
            "import pytest\n\nfrom pkg.testing import hooks\n\n\n"
            "@pytest.fixture()\ndef reset_hooks() -> None:\n    hooks.reset()\n",
            "from pkg.testing import hooks\n\n\ndef test_x() -> None:\n    hooks.loader = 1\n",
        )

        assert len(_run([test_file])) == 1

    def test_autouse_false_does_not_exempt(self, tmp_path: Path) -> None:
        """An explicit autouse=False is not autouse."""
        test_file = self._write_pair(
            tmp_path,
            "import pytest\n\nfrom pkg.testing import hooks\n\n\n"
            "@pytest.fixture(autouse=False)\ndef reset_hooks() -> None:\n    hooks.reset()\n",
            "from pkg.testing import hooks\n\n\ndef test_x() -> None:\n    hooks.loader = 1\n",
        )

        assert len(_run([test_file])) == 1

    def test_undecorated_fixture_does_not_exempt(self, tmp_path: Path) -> None:
        """A bare decorator carries no autouse keyword to inspect."""
        test_file = self._write_pair(
            tmp_path,
            "import pytest\n\nfrom pkg.testing import hooks\n\n\n"
            "@pytest.fixture\ndef reset_hooks() -> None:\n    hooks.reset()\n",
            "from pkg.testing import hooks\n\n\ndef test_x() -> None:\n    hooks.loader = 1\n",
        )

        assert len(_run([test_file])) == 1

    def test_other_decorator_keyword_does_not_exempt(self, tmp_path: Path) -> None:
        """A fixture parameterised on something other than autouse."""
        test_file = self._write_pair(
            tmp_path,
            "import pytest\n\nfrom pkg.testing import hooks\n\n\n"
            '@pytest.fixture(scope="function")\ndef reset_hooks() -> None:\n    hooks.reset()\n',
            "from pkg.testing import hooks\n\n\ndef test_x() -> None:\n    hooks.loader = 1\n",
        )

        assert len(_run([test_file])) == 1

    def test_reset_of_a_different_container_does_not_exempt(self, tmp_path: Path) -> None:
        """Resetting one container does not license patching another."""
        test_file = self._write_pair(
            tmp_path,
            "import pytest\n\nfrom pkg.testing import other\n\n\n"
            "@pytest.fixture(autouse=True)\ndef reset_hooks() -> None:\n    other.reset()\n",
            "from pkg.testing import hooks\n\n\ndef test_x() -> None:\n    hooks.loader = 1\n",
        )

        assert len(_run([test_file])) == 1

    def test_autouse_fixture_calling_another_method_does_not_exempt(self, tmp_path: Path) -> None:
        """Only reset() is known to clear every attribute."""
        test_file = self._write_pair(
            tmp_path,
            "import pytest\n\nfrom pkg.testing import hooks\n\n\n"
            "@pytest.fixture(autouse=True)\ndef reset_hooks() -> None:\n    hooks.configure()\n",
            "from pkg.testing import hooks\n\n\ndef test_x() -> None:\n    hooks.loader = 1\n",
        )

        assert len(_run([test_file])) == 1

    def test_bare_reset_call_does_not_exempt(self, tmp_path: Path) -> None:
        """A plain `reset()` names no container to exempt."""
        test_file = self._write_pair(
            tmp_path,
            "import pytest\n\nfrom pkg.testing import hooks, reset\n\n\n"
            "@pytest.fixture(autouse=True)\ndef reset_hooks() -> None:\n    reset()\n",
            "from pkg.testing import hooks\n\n\ndef test_x() -> None:\n    hooks.loader = 1\n",
        )

        assert len(_run([test_file])) == 1

    def test_chained_reset_call_does_not_exempt(self, tmp_path: Path) -> None:
        """`pkg.hooks.reset()` does not resolve to a bound name the rule tracks."""
        test_file = self._write_pair(
            tmp_path,
            "import pytest\n\nfrom pkg import testing\n\n\n"
            "@pytest.fixture(autouse=True)\ndef reset_hooks() -> None:\n"
            "    testing.hooks.reset()\n",
            "from pkg.testing import hooks\n\n\ndef test_x() -> None:\n    hooks.loader = 1\n",
        )

        assert len(_run([test_file])) == 1


class TestFileSelection:
    """Only files under a tests directory are checked."""

    def test_non_test_file_is_skipped(self, tmp_path: Path) -> None:
        """Production modules are outside this rule's scope."""
        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        source = src_dir / "thing.py"
        source.write_text(
            "import othermod\n\n\ndef go() -> None:\n    othermod.thing = 1\n",
            encoding="utf-8",
        )

        assert _run([source]) == []
