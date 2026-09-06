"""The production hook implementations, exercised for real.

The fakes elsewhere assert what this package ASKS FOR. Nothing there would
notice if the real git runner treated a failure as an empty index, or if the
emitter buffered its output forever -- so these run the real implementations
against a real repository and real standard output.

EVERY GIT CALL HERE IS SCOPED WITH ``-C``, NOT WITH ``chdir``. The working
directory is process-global, and ``-n auto`` runs several tests per worker
process, so a test that changed directory would be reaching into its
neighbours. ``-C`` targets a repository per invocation and leaves nothing
behind.

AND NOTHING HERE STAGES INTO THE SHARED INDEX. Each test builds its own
throwaway repository. Testing this package by staging into the real index
would be committing the exact defect it was written to prevent, in a tree
several sessions are editing right now.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.config import config_test_hooks
from platform_core.error_codes_tooling import CommitScopeErrorCode
from platform_core.errors import AppError

from commit_scope import _test_hooks
from commit_scope.config import REPO_ROOT_ARGUMENTS, STAGED_PATHS_ARGUMENTS
from commit_scope.contracts import decode_staged_paths


def _init_repo(root: pathlib.Path) -> None:
    """Create a real, isolated git repository.

    Args:
        root: An empty directory to initialise.
    """
    setup: tuple[tuple[str, ...], ...] = (
        ("init", "-q"),
        ("config", "user.email", "test@example.invalid"),
        ("config", "user.name", "test"),
    )
    for arguments in setup:
        # ``encoding`` is not cosmetic: without it ``subprocess.run`` is typed
        # ``CompletedProcess[Any]``, and this package forbids an Any-typed
        # expression anywhere -- including one whose result is discarded.
        subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            encoding="utf-8",
        )


def _at(root: pathlib.Path, arguments: tuple[str, ...]) -> tuple[str, ...]:
    """Scope one argument tuple to a repository.

    Args:
        root: The repository.
        arguments: The arguments the production code would use.

    Returns:
        The same arguments, preceded by ``-C <root>``.
    """
    return ("-C", str(root), *arguments)


def test_the_real_runner_reads_a_real_repository(tmp_path: pathlib.Path) -> None:
    """The production git seam answers the two questions the CLI asks.

    Exercises the binding against real git rather than asserting it is
    callable. A runner that compiled but could not read an index would
    otherwise be discovered at somebody's commit.
    """
    _init_repo(tmp_path)

    assert _test_hooks.run_git(_at(tmp_path, REPO_ROOT_ARGUMENTS)).strip()
    assert decode_staged_paths(_test_hooks.run_git(_at(tmp_path, STAGED_PATHS_ARGUMENTS))) == ()

    (tmp_path / "staged.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(
        ("git", "-C", str(tmp_path), "add", "staged.py"),
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    assert decode_staged_paths(_test_hooks.run_git(_at(tmp_path, STAGED_PATHS_ARGUMENTS))) == (
        "staged.py",
    )


def test_a_directory_that_is_not_a_repository_names_the_repository(
    tmp_path: pathlib.Path,
) -> None:
    """The failure must send the reader to the right place.

    An index error here would point at staging, which is not the problem.
    """
    with pytest.raises(AppError) as caught:
        _test_hooks.run_git(_at(tmp_path, REPO_ROOT_ARGUMENTS))
    assert caught.value.code is CommitScopeErrorCode.GIT_REPO_ROOT_UNRESOLVED


def test_a_failing_index_query_raises_rather_than_reporting_nothing_staged(
    tmp_path: pathlib.Path,
) -> None:
    """The dangerous failure mode, asserted directly.

    A hook that read a failed git as an empty index would print "0 path(s)"
    and wave every commit through -- worse than the defect this package
    removes, because it would look like protection.
    """
    _init_repo(tmp_path)
    with pytest.raises(AppError) as caught:
        _test_hooks.run_git(_at(tmp_path, ("diff", "--cached", "--this-flag-does-not-exist")))
    assert caught.value.code is CommitScopeErrorCode.GIT_INDEX_UNREADABLE


def test_the_real_environment_reader_normalises_blank_to_unset() -> None:
    """It delegates to the monorepo's one permitted environment reader.

    Rebinding that reader's own hook rather than setting a real variable is
    what keeps this package from growing a second ``os.environ`` access, and
    it exercises the delegation rather than assuming it.

    A blank declaration reading as ``""`` instead of None would decode to an
    empty scope, which this package treats as UNDECLARED -- so the bug would
    be invisible rather than loud.
    """
    config_test_hooks.get_env = {"SET": "present", "BLANK": "   "}.get
    assert _test_hooks.env("SET") == "present"
    assert _test_hooks.env("BLANK") is None
    assert _test_hooks.env("ABSENT") is None


def test_the_real_emitter_writes_and_flushes(capsys: pytest.CaptureFixture[str]) -> None:
    """Git shows a hook's output as it arrives.

    A buffered refusal is one the author has not read when their editor
    opens, so the flush is behaviour rather than tidiness.
    """
    _test_hooks.emit("first")
    _test_hooks.emit("second")
    assert capsys.readouterr().out == "first\nsecond\n"
