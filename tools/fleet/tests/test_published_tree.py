"""The published session-audit a hub session verb runs (MCPs board task f4cd489f).

Two halves. The scripted half pins the four commands, their order and
deadlines, and the refusal each failure produces, with nothing run after
it. The real half builds a small repository with a remote-tracking
``origin/main``, runs the real ``git`` and ``tar`` and a real generator
through the default command hook, and reads the extraction back off the
disk: the argv are only right if git and tar accept them.
"""

from __future__ import annotations

import pathlib

import pytest

from fleet.core import _test_hooks, published_tree
from tests._published_tree_fixtures import (
    COMMIT,
    extraction_calls,
    extraction_replies,
    pin_scratch,
    plant_extraction,
)
from tests.conftest import FakeRun, failed, ok


def refusal_of(outcome: published_tree.PublishedTree | str) -> str:
    """Narrow an extraction's outcome to the refusal a test expects.

    Args:
        outcome: What :func:`~fleet.core.published_tree.extract_published_tree`
            returned.

    Returns:
        The refusal detail.

    Raises:
        AssertionError: When the extraction succeeded instead.
    """
    if not isinstance(outcome, str):
        raise AssertionError(f"expected a refusal, got the extraction {outcome!r}")
    return outcome


class TestScriptedExtraction:
    def test_four_steps_in_order_each_with_the_step_deadline(self, tmp_path: pathlib.Path) -> None:
        mcps = tmp_path / "mcps"
        scratch = tmp_path / "scratch"
        tree = plant_extraction(scratch)
        runner = FakeRun(extraction_replies())
        _test_hooks.run = runner

        assert published_tree.extract_published_tree(mcps) == tree
        assert runner.calls == extraction_calls(mcps, scratch)
        assert runner.timeouts == [published_tree.TREE_STEP_TIMEOUT_SECONDS] * 4 == [120] * 4
        # Nothing about the extraction touches the environment: only the
        # verb that runs afterwards is given PYTHONPATH.
        assert runner.unset_env == [(), (), (), ()]
        assert runner.set_env == [(), (), (), ()]

    @pytest.mark.parametrize(
        "reply",
        [
            failed(128, "fatal: Needed a single revision"),
            ok("refs/remotes/origin/main\n"),
        ],
    )
    def test_a_ref_that_does_not_resolve_refuses_and_nothing_else_runs(
        self, tmp_path: pathlib.Path, reply: _test_hooks.CommandResult
    ) -> None:
        mcps = tmp_path / "mcps"
        pin_scratch(tmp_path / "scratch")
        runner = FakeRun([reply])
        _test_hooks.run = runner

        refusal = refusal_of(published_tree.extract_published_tree(mcps))

        assert refusal.startswith(
            f"SESSION_TREE_REF_UNRESOLVED: refs/remotes/origin/main in {mcps} did not resolve "
            f"to a commit (exit {reply['returncode']}"
        )
        assert refusal.endswith("; nothing ran")
        assert len(runner.calls) == 1

    @pytest.mark.parametrize(
        ("failing_step", "code", "step_words"),
        [
            (1, "SESSION_TREE_ARCHIVE_FAILED", f"git archive {COMMIT}"),
            (2, "SESSION_TREE_EXTRACT_FAILED", "tar -x "),
            (3, "SESSION_TREE_GENERATE_FAILED", "generate_contract_hash.py"),
        ],
    )
    def test_a_failed_step_refuses_by_its_own_code_and_stops_there(
        self, tmp_path: pathlib.Path, failing_step: int, code: str, step_words: str
    ) -> None:
        mcps = tmp_path / "mcps"
        pin_scratch(tmp_path / "scratch")
        replies = extraction_replies()[:failing_step]
        runner = FakeRun([*replies, failed(2, "  the step's own complaint  ")])
        _test_hooks.run = runner

        refusal = refusal_of(published_tree.extract_published_tree(mcps))

        assert refusal.startswith(f"{code}: ")
        assert step_words in refusal
        assert refusal.endswith(" exited 2: the step's own complaint")
        assert len(runner.calls) == failing_step + 1

    def test_an_extraction_missing_what_a_verb_imports_refuses_naming_each_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        mcps = tmp_path / "mcps"
        scratch = tmp_path / "scratch"
        pin_scratch(scratch)
        _test_hooks.run = FakeRun(extraction_replies())

        refusal = published_tree.extract_published_tree(mcps)

        destination = scratch / "fleet-session-audit" / COMMIT
        assert refusal == (
            f"SESSION_TREE_INCOMPLETE: the extraction of {COMMIT} at {destination} lacks "
            "packages/session-audit/src/session_audit/cli.py, "
            "mcp-shared-py/src/mcp_shared_py/contract_hash.py, "
            "mcp-shared/src/source-registry/supervision.json; nothing ran"
        )


#: A generator shaped like mcp-shared-py's: it resolves the package beside
#: itself and writes the one module git does not carry.
GENERATOR = (
    "import pathlib\n"
    "package = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'mcp_shared_py'\n"
    "(package / 'contract_hash.py').write_text('HASH = \"x\"\\n', encoding='utf-8')\n"
)

#: The files the committed tree carries, and one the archive must leave out.
COMMITTED: dict[str, str] = {
    "packages/session-audit/src/session_audit/cli.py": "PUBLISHED = True\n",
    "packages/session-audit/pyproject.toml": "[tool.poetry]\n",
    "mcp-shared-py/src/mcp_shared_py/__init__.py": "",
    "mcp-shared-py/scripts/generate_contract_hash.py": GENERATOR,
    "mcp-shared/src/source-registry/supervision.json": "{}\n",
}


def git(repo: pathlib.Path, *arguments: str) -> str:
    """Run one real git command in the repository and return its stdout.

    Args:
        repo: The repository.
        arguments: The git arguments.

    Returns:
        Its standard output, stripped.
    """
    result = _test_hooks.run(
        (
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=fleet-test",
            "-c",
            "user.email=fleet-test@example.invalid",
            "-c",
            "core.autocrlf=false",
            *arguments,
        ),
        timeout_seconds=60,
    )
    assert result["returncode"] == 0, result["stderr"]
    return result["stdout"].strip()


def published_repository(repo: pathlib.Path) -> str:
    """Commit :data:`COMMITTED`, point ``origin/main`` at it, then dirty the tree.

    Args:
        repo: Where to build the repository.

    Returns:
        The commit ``refs/remotes/origin/main`` names.
    """
    repo.mkdir()
    git(repo, "init", "--quiet")
    for relative, body in COMMITTED.items():
        path = repo / pathlib.PurePosixPath(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "-m", "published")
    commit = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/remotes/origin/main", commit)
    # The working tree moves on without publishing: an uncommitted edit that
    # the verb must never see.
    (repo / "packages/session-audit/src/session_audit/cli.py").write_text(
        "PUBLISHED = False\n", encoding="utf-8"
    )
    return commit


class TestRealExtraction:
    def test_the_published_commit_is_extracted_generated_and_the_working_tree_ignored(
        self, tmp_path: pathlib.Path
    ) -> None:
        repo = tmp_path / "mcps"
        commit = published_repository(repo)
        scratch = tmp_path / "scratch"
        pin_scratch(scratch)

        tree = published_tree.extract_published_tree(repo)

        if isinstance(tree, str):
            raise AssertionError(f"expected an extraction, got the refusal {tree!r}")
        assert tree["commit"] == commit
        destination = scratch / "fleet-session-audit" / commit
        cli = destination / "packages/session-audit/src/session_audit/cli.py"
        assert cli.read_text(encoding="utf-8") == "PUBLISHED = True\n"
        generated = destination / "mcp-shared-py/src/mcp_shared_py/contract_hash.py"
        assert generated.read_text(encoding="utf-8") == 'HASH = "x"\n'
        assert tree["registry_dir"] == str(destination / "mcp-shared/src/source-registry")
        # Only the named trees are archived; the package's own project file
        # is not code a verb imports and stays out.
        assert not (destination / "packages/session-audit/pyproject.toml").exists()

    def test_a_repository_with_no_published_ref_refuses_for_real(
        self, tmp_path: pathlib.Path
    ) -> None:
        repo = tmp_path / "mcps"
        repo.mkdir()
        git(repo, "init", "--quiet")
        pin_scratch(tmp_path / "scratch")

        refusal = refusal_of(published_tree.extract_published_tree(repo))

        assert refusal.startswith(
            f"SESSION_TREE_REF_UNRESOLVED: refs/remotes/origin/main in {repo} did not resolve "
            "to a commit (exit 128"
        )
        assert not (tmp_path / "scratch" / "fleet-session-audit").exists()
