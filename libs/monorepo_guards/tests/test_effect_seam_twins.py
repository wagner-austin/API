"""Tests for finding each effect seam's real tests and their failure evidence."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from monorepo_guards.effect_seam_twins import RealTest, failure_evidence, real_tests
from monorepo_guards.effect_seams import effect_seams, index_package
from tests._effect_support import files_of, write

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


def _fails(source: str) -> bool:
    """Read failure evidence from one function's source.

    Args:
        source: A function definition.

    Returns:
        Whether it names a failure.
    """
    return failure_evidence(ast.parse(textwrap.dedent(source)))


class TestFailureEvidence:
    def test_failure_names_attributes_and_keys(self) -> None:
        assert _fails("with raises(subprocess.TimeoutExpired):\n    pass\n")
        assert _fails("assert result.timed_out\n")
        assert _fails('assert result["killed"] is True\n')
        assert _fails("raise ConnectError\n")
        assert not _fails("timed_out = False\n")
        assert not _fails('assert result["stdout"] == "x"\n')
        assert not _fails("value = items[0]\n")

    def test_kill_calls_and_nonzero_exit_text(self) -> None:
        assert _fails("child.terminate()\n")
        assert _fails('script = "import sys; sys.exit(3)"\n')
        assert _fails('script = "process.exit(1)"\n')
        assert _fails('script = "exit 2"\n')
        assert not _fails('script = "sys.exit(0)"\n')
        assert not _fails("make()()\n")

    def test_exit_code_comparisons(self) -> None:
        assert _fails("assert result.returncode == 3\n")
        assert _fails("assert 3 == result.returncode\n")
        assert _fails('assert result["returncode"] != 0\n')
        assert _fails("assert result.exit_code > 0\n")
        assert _fails("assert 0 < result.code\n")
        assert not _fails("assert result.returncode == 0\n")
        assert not _fails("assert result.returncode != 1\n")
        assert not _fails("assert result.returncode > 1\n")
        assert not _fails("assert 0 > result.code\n")
        assert not _fails("assert result.returncode == expected\n")
        assert not _fails("assert result.returncode == True\n")
        assert not _fails("assert size == 3\n")

    def test_status_comparisons(self) -> None:
        assert _fails("assert response.status_code == 503\n")
        assert _fails("assert 404 == response.status\n")
        assert not _fails("assert response.status_code == 200\n")
        assert not _fails("assert response.status == expected\n")


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
            RealTest("tests/test_real.py::test_named_kill_of_a_dead_pid", True)
        ]
        assert found[("pkg._test_hooks", "_default_run")] == [
            RealTest("tests/test_real.py::test_module_object_run_success", False)
        ]
        assert found[("pkg._test_hooks", "default_hooks.spawn")] == [
            RealTest("tests/test_real.py::TestFactory.test_spawn_through_the_bundle", True)
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
            RealTest("tests/test_bundle.py::test_assigned_bundle_field", True),
            RealTest("tests/test_bundle.py::test_whole_bundle_handed_over", False),
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
            RealTest("tests/sub/test_reach.py::test_through_a_helper", True),
            RealTest("tests/sub/test_reach.py::test_through_a_fixture", False),
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
