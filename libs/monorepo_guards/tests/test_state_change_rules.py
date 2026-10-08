"""Tests for the state-change-verified guard rule."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from monorepo_guards.state_change_rules import (
    STATE_CHANGE_VERBS,
    StateChangeVerifiedRule,
    leading_word,
    missing_safeguards,
)
from tests._effect_support import config_for, files_of, write

HOOKS = """
import subprocess

def _default_run(argv):
    return subprocess.run(argv, timeout=5)

def _default_now():
    return 1

run: RunProto = _default_run
now: NowProto = _default_now
"""


def _run(root: Path) -> list[str]:
    """Run the rule over a written package.

    Args:
        root: The package root.

    Returns:
        Each violation's text.
    """
    violations = StateChangeVerifiedRule(config_for(root)).run(files_of(root))
    assert all(v.kind == "state-change-unverified" for v in violations)
    return [v.line for v in violations]


def _missing(source: str) -> list[str]:
    """Read what the first function of a module lacks.

    Args:
        source: A module whose first statement is the state change.

    Returns:
        What it lacks.
    """
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise TypeError("the first statement must be a function")
    helpers: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    return missing_safeguards(function, helpers)


class TestNames:
    def test_rollout_is_not_a_verb(self) -> None:
        assert "rollout" not in STATE_CHANGE_VERBS
        assert "promote" in STATE_CHANGE_VERBS

    def test_the_leading_word_of_snake_and_camel_names(self) -> None:
        assert leading_word("install_local") == "install"
        assert leading_word("_swap_launcher") == "swap"
        assert leading_word("restartHost") == "restart"
        assert leading_word("record_deploy") == "record"
        assert leading_word("promote") == "promote"


class TestMissingSafeguards:
    def test_verify_and_a_guarded_restore_satisfy_it(self) -> None:
        source = """
        def promote(new):
            swap_in(new)
            if not verify_launcher():
                restore_previous()
                raise RuntimeError("launcher missing")
        """
        assert _missing(source) == []

    def test_an_unguarded_restore_does_not_count(self) -> None:
        source = """
        def promote(new):
            swap_in(new)
            path.exists()
            rollback()
        """
        assert _missing(source) == ["restore"]

    def test_handlers_and_finally_guard_and_helpers_are_followed(self) -> None:
        source = """
        def install(new):
            try:
                put(new)
            except OSError as error:
                _revert(new)
                raise RuntimeError("install failed") from error
            finally:
                done()
            _probe()

        def _probe():
            return _probe()
        """
        assert _missing(source) == []
        source_finally = """
        def install(new):
            try:
                put(new)
            finally:
                roll_back()
            hooks()()
        """
        assert _missing(source_finally) == ["verify"]

    def test_nothing_at_all(self) -> None:
        assert _missing("def deploy():\n    go()\n") == ["verify", "restore"]

    def test_an_else_branch_guards_too(self) -> None:
        source = """
        def upgrade():
            if check_ok():
                pass
            else:
                restore()
        """
        assert _missing(source) == []


class TestStateChangeVerifiedRule:
    def test_the_rule_is_named_as_mcps_names_it(self, tmp_path: Path) -> None:
        assert StateChangeVerifiedRule(config_for(tmp_path)).name == "state-change-verified"

    def test_effects_direct_through_hooks_and_through_delegates(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/_test_hooks.py", HOOKS)
        write(
            tmp_path,
            "src/pkg/remote.py",
            """
            from pkg import _test_hooks

            def run_script(host):
                return _attempt(host)

            def _attempt(host):
                return _test_hooks.run(["ssh", host])
            """,
        )
        write(
            tmp_path,
            "src/pkg/ops.py",
            """
            import shutil
            from pkg import _test_hooks, remote
            from pkg.remote import run_script

            def install_missing(node):
                remote.run_script(node)

            def deploy(node):
                run_script(node)

            def swap_launcher(a, b):
                shutil.move(a, b)

            class Service:
                def restartWorker(self):
                    self._hooks.run(["x"])

            def promote(build):
                default_hooks().run(build)

            def upgrade(build):
                get_hooks().run(build)

            def reinstall(build):
                make_hooks().run(build)

            def uninstall():
                return _test_hooks.now()

            def install_signal_handler():
                signal.signal(2, handler)

            def redeploy(node):
                registry[node].run()
                factory()()
                remote.missing(node)
                os.path.join(node)
                redeploy(node)

            def restart(node):
                _describe(node)
                remote.run_script(node)
                if not check_running(node):
                    restore_previous(node)

            def record_deploy(node):
                remote.run_script(node)

            def _describe(node):
                return str(node)
            """,
        )
        assert _run(tmp_path) == [
            "src/pkg/ops.py:6 install_missing has no verify and no restore",
            "src/pkg/ops.py:9 deploy has no verify and no restore",
            "src/pkg/ops.py:12 swap_launcher has no verify and no restore",
            "src/pkg/ops.py:16 restartWorker has no verify and no restore",
            "src/pkg/ops.py:19 promote has no verify and no restore",
            "src/pkg/ops.py:22 upgrade has no verify and no restore",
        ]

    def test_tests_are_not_read(self, tmp_path: Path) -> None:
        write(
            tmp_path,
            "tests/test_ops.py",
            "import shutil\n\ndef swap(a, b):\n    shutil.move(a, b)\n",
        )
        assert _run(tmp_path) == []
