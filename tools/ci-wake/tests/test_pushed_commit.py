"""The commit a push lands and its workflows, asked of git (MCPs board task 24547f50).

Scripted answers pin each command and each refusal; one case runs the
production runner against a real repository built in ``tmp_path``, with an
annotated tag over a commit carrying MCPs's two workflows, because the peel
is the defect this exists for and a scripted git cannot show what real git
answers for a tag object.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.error_codes_tooling import CiWakeErrorCode
from platform_core.errors import AppError

from ci_wake import _test_hooks
from ci_wake.pushed_commit import GIT_TIMEOUT_SECONDS, peel_commit, read_workflows
from ci_wake.workflow_triggers import EVERY_REF, PushTrigger, Workflow
from tests.conftest import (
    MCPS_CHECK,
    MCPS_CHECK_LIVE,
    MCPS_WORKFLOWS,
    OTHER_SHA,
    SHA,
    TAG_OBJECT,
    FakeCompleted,
    FakeGh,
    git_replies,
)

MAIN_ONLY = PushTrigger(branches=("main",), branches_ignore=None, tags=None, tags_ignore=None)


class TestPeel:
    def test_a_tag_object_peels_to_the_commit_it_names(self) -> None:
        gh = FakeGh(git_replies(TAG_OBJECT, SHA, {}))
        _test_hooks.run_process = gh

        assert peel_commit(TAG_OBJECT) == SHA
        assert gh.calls == [("git", "rev-parse", "--verify", f"{TAG_OBJECT}^{{commit}}")]

    def test_an_object_naming_no_commit_is_refused_with_gits_words(self) -> None:
        _test_hooks.run_process = FakeGh(
            {
                ("git", "rev-parse", "--verify", f"{SHA}^{{commit}}"): FakeCompleted(
                    stderr="fatal: Needed a single revision\n", returncode=128
                )
            }
        )

        with pytest.raises(AppError) as refused:
            peel_commit(SHA)

        assert refused.value.code is CiWakeErrorCode.PUSHED_COMMIT_UNREADABLE
        assert refused.value.message == (
            f"git rev-parse --verify {SHA}^{{commit}} exited 128 while peeling the pushed "
            f"object {SHA} to a commit: fatal: Needed a single revision"
        )

    def test_an_answer_that_is_not_a_sha_is_refused(self) -> None:
        _test_hooks.run_process = FakeGh(
            {
                ("git", "rev-parse", "--verify", f"{SHA}^{{commit}}"): FakeCompleted(
                    stdout="6a0e66a\n"
                )
            }
        )

        with pytest.raises(AppError) as refused:
            peel_commit(SHA)

        assert refused.value.code is CiWakeErrorCode.ENROLMENT_FIELD_MALFORMED


class TestReadWorkflows:
    def test_reads_each_yaml_file_directly_in_the_directory(self) -> None:
        replies = git_replies(SHA, SHA, {"check.yml": MCPS_CHECK, "deploy.yaml": "on: push\n"})
        listing = replies[("git", "ls-tree", "-z", SHA, "--", ".github/workflows/")].stdout
        # A subdirectory and a README beside them: GitHub reads neither.
        replies[("git", "ls-tree", "-z", SHA, "--", ".github/workflows/")] = FakeCompleted(
            stdout=listing
            + f"040000 tree {OTHER_SHA}\t.github/workflows/shared.yml\0"
            + f"100644 blob {OTHER_SHA}\t.github/workflows/README.md\0"
        )
        gh = FakeGh(replies)
        _test_hooks.run_process = gh

        assert read_workflows(SHA) == (
            Workflow("check.yml", MAIN_ONLY),
            Workflow("deploy.yaml", EVERY_REF),
        )
        assert [call[1] for call in gh.calls] == ["ls-tree", "cat-file", "cat-file"]

    def test_a_commit_without_the_directory_has_no_workflows(self) -> None:
        _test_hooks.run_process = FakeGh(git_replies(SHA, SHA, {}))

        assert read_workflows(SHA) == ()

    def test_a_listing_git_cannot_make_is_refused(self) -> None:
        _test_hooks.run_process = FakeGh(
            {
                ("git", "ls-tree", "-z", SHA, "--", ".github/workflows/"): FakeCompleted(
                    stderr=f"fatal: not a tree object: {SHA}\n", returncode=128
                )
            }
        )

        with pytest.raises(AppError) as refused:
            read_workflows(SHA)

        assert refused.value.code is CiWakeErrorCode.PUSHED_COMMIT_UNREADABLE
        assert refused.value.message == (
            f"git ls-tree -z {SHA} -- .github/workflows/ exited 128 while listing "
            f".github/workflows in {SHA}: fatal: not a tree object: {SHA}"
        )

    def test_a_blob_git_cannot_read_is_refused(self) -> None:
        replies = git_replies(SHA, SHA, {"check.yml": MCPS_CHECK})
        blob = f"{1:040x}"
        replies[("git", "cat-file", "blob", blob)] = FakeCompleted(
            stderr="fatal: bad file\n", returncode=128
        )
        _test_hooks.run_process = FakeGh(replies)

        with pytest.raises(AppError) as refused:
            read_workflows(SHA)

        assert refused.value.message == (
            f"git cat-file blob {blob} exited 128 while reading .github/workflows/check.yml "
            f"in {SHA}: fatal: bad file"
        )


def _git(repo: pathlib.Path, *args: str) -> str:
    """Run git in the test repository, failing the test on any refusal."""
    completed = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        capture_output=True,
        encoding="utf-8",
        check=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    return completed.stdout.strip()


def test_real_git_peels_an_annotated_tag_and_reads_the_workflows_under_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production runner, real git, the shape of the 2026-10-11 push: an
    annotated tag over a commit holding MCPs's two workflows, one of them
    carrying a UTF-8 character outside cp1252's defined range in a comment."""
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "check.yml").write_bytes(("# ā\n" + MCPS_CHECK).encode())
    (workflows / "check-live.yml").write_bytes(MCPS_CHECK_LIVE.encode())
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "workflows")
    _git(tmp_path, "tag", "-a", "archive/dx-mod", "-m", "archived")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    tag_object = _git(tmp_path, "rev-parse", "archive/dx-mod")
    assert tag_object != commit
    monkeypatch.chdir(tmp_path)

    assert peel_commit(tag_object) == commit
    assert read_workflows(commit) == tuple(
        Workflow(name, MAIN_ONLY) for name in sorted(MCPS_WORKFLOWS)
    )
