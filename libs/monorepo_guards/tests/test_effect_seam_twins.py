"""Tests for finding each effect seam's real tests and the failures they name."""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.effect_failures import PROCESS
from monorepo_guards.effect_seam_twins import RealTest, real_tests
from monorepo_guards.effect_seams import effect_seams, index_package
from tests._effect_support import files_of, write

PROCESS_FAILURE = frozenset({PROCESS})
NO_FAILURE: frozenset[str] = frozenset()

HOOKS = """
import subprocess
import os

def _default_run(argv):
    return subprocess.run(argv, timeout=5)

def _default_kill(pid):
    os.kill(pid, 9)

def default_hooks():
    return Hooks(run=_default_run, spawn=lambda argv: subprocess.Popen(argv))

run: RunProto = _default_run
"""


def _real(root: Path) -> dict[tuple[str, str], list[RealTest]]:
    """Run the twin finder over a written package.

    Args:
        root: The package root.

    Returns:
        Real tests per seam.
    """
    files = files_of(root)
    index = index_package(files, root)
    return real_tests(files, root, index, effect_seams(index))


class TestRealTests:
    def test_named_module_object_and_factory_calls_count(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/_test_hooks.py", HOOKS)
        write(
            tmp_path,
            "tests/test_real.py",
            """
            import signal
            from pkg import _test_hooks
            from pkg._test_hooks import _default_kill, default_hooks

            def test_named_kill_of_a_dead_pid():
                _default_kill(99999)
                assert signal.SIGKILL

            def test_module_object_run_success():
                assert _test_hooks._default_run(["true"]).returncode == 0

            class TestFactory:
                def test_spawn_through_the_bundle(self):
                    default_hooks().spawn(["x"]).kill()

                def helper(self):
                    return 1

            class Other:
                def test_ignored(self):
                    _default_kill(1)
            """,
        )
        found = _real(tmp_path)
        assert found[("pkg._test_hooks", "_default_kill")] == [
            RealTest("tests/test_real.py::test_named_kill_of_a_dead_pid", PROCESS_FAILURE)
        ]
        assert found[("pkg._test_hooks", "_default_run")] == [
            RealTest("tests/test_real.py::test_module_object_run_success", NO_FAILURE)
        ]
        assert found[("pkg._test_hooks", "default_hooks.spawn")] == [
            RealTest(
                "tests/test_real.py::TestFactory.test_spawn_through_the_bundle", PROCESS_FAILURE
            )
        ]

    def test_the_rebindable_hook_is_not_the_implementation(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/_test_hooks.py", HOOKS)
        write(
            tmp_path,
            "tests/test_hook.py",
            """
            from pkg import _test_hooks

            def test_through_the_hook():
                assert _test_hooks.run(["false"]).returncode == 1
            """,
        )
        found = _real(tmp_path)
        assert found[("pkg._test_hooks", "_default_run")] == []

    def test_a_bound_factory_result_and_a_handed_over_bundle(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/_test_hooks.py", HOOKS)
        write(
            tmp_path,
            "tests/test_bundle.py",
            """
            from pkg._test_hooks import default_hooks

            def test_assigned_bundle_field():
                real = default_hooks()
                other = make()
                built = build()()
                fake = replace(real, run=real.run)
                assert fake.run(["false"]).returncode == 2
                assert other.run
                assert built

            def test_whole_bundle_handed_over():
                set_hooks(default_hooks())
                set_hooks(unknown_factory())
                items[0].run
                assert run_code() == 0
            """,
        )
        found = _real(tmp_path)
        assert found[("pkg._test_hooks", "_default_run")] == [
            RealTest("tests/test_bundle.py::test_assigned_bundle_field", PROCESS_FAILURE),
            RealTest("tests/test_bundle.py::test_whole_bundle_handed_over", NO_FAILURE),
        ]

    def test_helpers_and_fixtures_reach_but_only_helpers_carry_failure(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path, "src/pkg/_test_hooks.py", HOOKS)
        write(
            tmp_path,
            "tests/conftest.py",
            """
            import pytest
            from pkg import _test_hooks

            @pytest.fixture
            def real_run():
                return _test_hooks._default_run(["true"])

            @pytest.fixture()
            def restored():
                _test_hooks.run = _test_hooks._default_run
                assert TimeoutError
                return 1

            @pytest.fixture
            def chained(real_run, chained):
                return real_run
            """,
        )
        write(
            tmp_path,
            "tests/sub/test_reach.py",
            """
            from pkg import _test_hooks

            def _dial():
                return _test_hooks._default_run(["false"]).returncode == 7

            def _again():
                return _dial() and _again()

            def test_through_a_helper():
                assert _again()

            def test_through_a_fixture(chained):
                assert chained.returncode == 0

            def test_restored_hooks_are_not_a_run(restored):
                assert restored == 1
            """,
        )
        found = _real(tmp_path)
        assert found[("pkg._test_hooks", "_default_run")] == [
            RealTest("tests/sub/test_reach.py::test_through_a_helper", PROCESS_FAILURE),
            RealTest("tests/sub/test_reach.py::test_through_a_fixture", NO_FAILURE),
        ]

    def test_a_local_fixture_shadows_a_conftest_one(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/_test_hooks.py", HOOKS)
        write(
            tmp_path,
            "tests/conftest.py",
            """
            import functools
            from pkg import _test_hooks
            import pytest

            @pytest.fixture
            def runner():
                return _test_hooks._default_run(["x"])

            @functools.cache
            def cached():
                return 1
            """,
        )
        write(
            tmp_path,
            "tests/other/conftest.py",
            "import pytest\n\n@pytest.fixture\ndef runner():\n    return 1\n",
        )
        write(
            tmp_path,
            "tests/test_shadow.py",
            """
            import pytest

            @pytest.fixture
            def runner():
                return 2

            def test_uses_the_local_one(runner):
                assert runner == 2
            """,
        )
        found = _real(tmp_path)
        assert found[("pkg._test_hooks", "_default_run")] == []
