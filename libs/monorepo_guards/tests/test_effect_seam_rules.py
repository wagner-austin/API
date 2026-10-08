"""Tests for the effect-seam-twin guard rule."""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.effect_seam_rules import EffectSeamTwinRule
from tests._effect_support import config_for, files_of, write

HOOKS = """
import subprocess
from shutil import rmtree

def _default_run(argv):
    return subprocess.run(argv, timeout=5)

def _default_now():
    return 1

remove_tree = rmtree
"""


def _run(root: Path) -> list[tuple[int, str, str]]:
    """Run the rule over a written package.

    Args:
        root: The package root.

    Returns:
        ``(line, kind, text)`` per violation.
    """
    violations = EffectSeamTwinRule(config_for(root)).run(files_of(root))
    return [(v.line_no, v.kind, v.line) for v in violations]


class TestEffectSeamTwinRule:
    def test_the_rule_is_named_as_mcps_names_it(self, tmp_path: Path) -> None:
        assert EffectSeamTwinRule(config_for(tmp_path)).name == "effect-seam-twin"

    def test_a_seam_no_test_runs_is_named_with_its_chain(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/_test_hooks.py", HOOKS)
        assert _run(tmp_path) == [
            (
                5,
                "effect-seam-twin-missing",
                "src/pkg/_test_hooks.py:_default_run (process: _default_run -> subprocess.run)"
                " no test runs its real implementation",
            ),
            (
                11,
                "effect-seam-twin-missing",
                "src/pkg/_test_hooks.py:remove_tree (file swap: remove_tree -> shutil.rmtree)"
                " no test runs its real implementation",
            ),
        ]

    def test_real_tests_that_only_succeed_are_named(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/_test_hooks.py", HOOKS)
        write(
            tmp_path,
            "tests/test_run.py",
            """
            from pkg._test_hooks import _default_run

            def test_true_succeeds():
                assert _default_run(["true"]).returncode == 0

            def test_echo_succeeds():
                assert _default_run(["echo"]).stdout == b""
            """,
        )
        violations = _run(tmp_path)
        assert violations[0] == (
            5,
            "effect-seam-twin-no-failure",
            "src/pkg/_test_hooks.py:_default_run (process: _default_run -> subprocess.run)"
            " its real tests tests/test_run.py::test_true_succeeds,"
            " tests/test_run.py::test_echo_succeeds exercise no process failure",
        )
        assert len(violations) == 2

    def test_a_failure_of_another_kind_does_not_satisfy_the_seam(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/_test_hooks.py", HOOKS)
        write(
            tmp_path,
            "tests/test_run.py",
            """
            import pytest
            from pkg._test_hooks import _default_run

            def test_the_run_beside_a_refused_removal():
                assert _default_run(["true"]).returncode == 0
                with pytest.raises(PermissionError):
                    remove_locked()
            """,
        )
        assert [kind for _, kind, _ in _run(tmp_path)] == [
            "effect-seam-twin-no-failure",
            "effect-seam-twin-missing",
        ]

    def test_one_real_test_through_a_failure_satisfies_the_seam(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/_test_hooks.py", HOOKS)
        write(
            tmp_path,
            "tests/test_run.py",
            """
            import subprocess
            import pytest
            from pkg._test_hooks import _default_run

            def test_true_succeeds():
                assert _default_run(["true"]).returncode == 0

            def test_a_hung_child_times_out():
                with pytest.raises(subprocess.TimeoutExpired):
                    _default_run(["sleep", "60"])
            """,
        )
        assert [kind for _, kind, _ in _run(tmp_path)] == ["effect-seam-twin-missing"]

    def test_a_package_without_hooks_is_clean(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/runner.py", "import subprocess\nsubprocess.run([])\n")
        assert _run(tmp_path) == []
