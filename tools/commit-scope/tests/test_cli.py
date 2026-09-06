"""The command a hook runs: wiring, exit codes, and what it asks git."""

from __future__ import annotations

import runpy
import sys

import pytest
from platform_core.error_codes_tooling import CommitScopeErrorCode
from platform_core.errors import AppError

from commit_scope import _test_hooks
from commit_scope.cli.check import EXIT_ERROR, EXIT_OK, EXIT_REFUSED, entrypoint, gather, main
from commit_scope.config import REPO_ROOT_ARGUMENTS, SCOPE_VARIABLE, STAGED_PATHS_ARGUMENTS
from tests.conftest import FakeEmit, FakeEnv, FakeGit


def _bind(staged: str, declaration: dict[str, str]) -> tuple[FakeGit, FakeEnv, FakeEmit]:
    """Bind all three seams to fakes.

    Args:
        staged: What git reports as the index listing.
        declaration: Environment variables that are set.

    Returns:
        The three fakes, for assertion.
    """
    git = FakeGit({REPO_ROOT_ARGUMENTS: "/repo\n", STAGED_PATHS_ARGUMENTS: staged})
    env = FakeEnv(declaration)
    emit = FakeEmit()
    _test_hooks.run_git = git
    _test_hooks.env = env
    _test_hooks.emit = emit
    return git, env, emit


class TestGather:
    """What the command asks, and in which order."""

    def test_proves_the_work_tree_before_reading_the_index(self) -> None:
        """A non-repository must fail naming the repository, not the index."""
        git, _env, _emit = _bind("a.py\n", {})
        gather()
        assert git.calls == [REPO_ROOT_ARGUMENTS, STAGED_PATHS_ARGUMENTS]

    def test_reads_the_declaration_from_the_documented_variable(self) -> None:
        """The name is config, and a test must pin it."""
        _git, env, _emit = _bind("a.py\n", {})
        gather()
        assert env.names == [SCOPE_VARIABLE]

    def test_returns_the_decoded_question(self) -> None:
        """Both halves arrive decoded rather than raw."""
        _bind("a.py\nb.py\n", {SCOPE_VARIABLE: "a.py"})
        assert gather() == {"staged": ("a.py", "b.py"), "scope": ("a.py",)}


class TestMain:
    """Exit codes, which are this command's whole interface."""

    def test_undeclared_commit_proceeds_and_prints_the_receipt(self) -> None:
        """Narrowing, not closing: it shows and allows."""
        _git, _env, emit = _bind("a.py\nb.py\n", {})
        assert main() == EXIT_OK
        assert "staging receipt (2 path(s))" in emit.text

    def test_declared_and_clean_proceeds(self) -> None:
        """The ordinary case must be quiet and fast."""
        _git, _env, emit = _bind("a.py\n", {SCOPE_VARIABLE: "a.py"})
        assert main() == EXIT_OK
        assert "staged scope OK" in emit.text

    def test_the_sweep_is_refused(self) -> None:
        """The incident this package exists for, end to end through the CLI."""
        _git, _env, emit = _bind("mine.py\ntheirs.py\n", {SCOPE_VARIABLE: "mine.py"})
        assert main() == EXIT_REFUSED
        assert "COMMIT BLOCKED" in emit.text
        assert "    theirs.py" in emit.text

    def test_a_broken_declaration_propagates_rather_than_passing(self) -> None:
        """An unmatchable entry must not silently protect nothing."""
        _bind("a.py\n", {SCOPE_VARIABLE: "/etc/passwd"})
        with pytest.raises(AppError) as caught:
            main()
        assert caught.value.code is CommitScopeErrorCode.SCOPE_ENTRY_NOT_RELATIVE


class TestEntrypoint:
    """The process boundary: a failure is not a refusal."""

    def test_passes_through_the_ok_status(self) -> None:
        """A clean commit reaches the shell as success."""
        _bind("a.py\n", {SCOPE_VARIABLE: "a.py"})
        assert entrypoint() == EXIT_OK

    def test_passes_through_the_refusal_status(self) -> None:
        """A refusal stays 1 so git stops the commit."""
        _bind("mine.py\ntheirs.py\n", {SCOPE_VARIABLE: "mine.py"})
        assert entrypoint() == EXIT_REFUSED

    def test_renders_a_failure_with_its_code_and_a_distinct_status(self) -> None:
        """A broken environment must be distinguishable from a refused commit.

        Returning 1 for both would teach the author that this check is noisy,
        and a check people route around protects nothing.
        """
        _git, _env, emit = _bind("a.py\n", {SCOPE_VARIABLE: "../elsewhere"})
        assert entrypoint() == EXIT_ERROR
        assert CommitScopeErrorCode.SCOPE_ENTRY_ESCAPES_REPO.value in emit.text
        assert "climbs out of the repository" in emit.text

    def test_running_as_a_module_exits_with_the_check_status(self) -> None:
        """``python -m commit_scope.cli.check`` is how a hook may invoke it.

        The console script is one entry and the module is the other; a
        ``__main__`` block that raised instead of exiting would break the
        hook while every direct call to :func:`entrypoint` still passed.
        """
        _bind("mine.py\ntheirs.py\n", {SCOPE_VARIABLE: "mine.py"})
        sys.modules.pop("commit_scope.cli.check", None)
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("commit_scope.cli.check", run_name="__main__")
        assert caught.value.code == EXIT_REFUSED

    def test_reports_a_git_failure_under_its_own_code(self) -> None:
        """An unreadable index must never be read as an empty one."""
        emit = FakeEmit()
        _test_hooks.emit = emit
        _test_hooks.env = FakeEnv({})

        def _failing_git(arguments: tuple[str, ...]) -> str:
            """Fail the way a broken index does.

            Args:
                arguments: Ignored; every call fails.

            Returns:
                Never returns.

            Raises:
                AppError: Always, with the index code.
            """
            raise AppError(
                code=CommitScopeErrorCode.GIT_INDEX_UNREADABLE,
                message="git diff --cached failed with status 128",
            )

        _test_hooks.run_git = _failing_git
        assert entrypoint() == EXIT_ERROR
        assert CommitScopeErrorCode.GIT_INDEX_UNREADABLE.value in emit.text
