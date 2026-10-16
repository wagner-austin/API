"""Reading a checkout's repository, HEAD and tracked changes, against real git (MCPs 9f1367d5)."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, ErrorCode

from fleet.core.checkout import Checkout, read_checkout, repository_name
from tests._git_fixtures import committed_repository, git


def _checkout(tmp_path: pathlib.Path, remote: str) -> tuple[pathlib.Path, str]:
    """Build a one-commit checkout whose origin is ``remote``.

    Args:
        tmp_path: The test's directory.
        remote: The origin url to declare.

    Returns:
        The checkout and its HEAD commit.
    """
    repo = tmp_path / "checkout"
    commit = committed_repository(
        repo, {"tools/hpc-wake/run.py": "print('tick')\n"}, "refs/heads/main"
    )
    git(repo, "checkout", "--quiet", "main")
    git(repo, "remote", "add", "origin", remote)
    return repo, commit


class TestRepositoryName:
    @pytest.mark.parametrize(
        ("remote", "expected"),
        [
            ("https://github.com/wagner-austin/API.git", "API"),
            ("https://github.com/wagner-austin/MCPs", "MCPs"),
            ("git@github.com:wagner-austin/MCPs.git", "MCPs"),
            ("git@github.com:MCPs.git", "MCPs"),
            ("C:\\Users\\Test\\PROJECTS\\API\n", "API"),
            ("/srv/git/API.git/", "API"),
            ("https://github.com/", "github.com"),
            ("/", ""),
        ],
    )
    def test_it_names_the_last_segment_without_dot_git(self, remote: str, expected: str) -> None:
        assert repository_name(remote) == expected


class TestReadCheckout:
    def test_a_clean_checkout_reads_its_repository_and_head(self, tmp_path: pathlib.Path) -> None:
        repo, commit = _checkout(tmp_path, "https://github.com/wagner-austin/API.git")

        assert read_checkout(repo / "tools" / "hpc-wake") == Checkout(
            repo="API", head_sha=commit, dirty=False
        )

    def test_an_edited_tracked_file_reads_dirty(self, tmp_path: pathlib.Path) -> None:
        repo, commit = _checkout(tmp_path, "https://github.com/wagner-austin/API.git")
        (repo / "tools" / "hpc-wake" / "run.py").write_text("print('edited')\n", encoding="utf-8")

        assert read_checkout(repo) == Checkout(repo="API", head_sha=commit, dirty=True)

    def test_an_untracked_file_leaves_it_clean(self, tmp_path: pathlib.Path) -> None:
        # Every bridge keeps its runs/ and cursors untracked beside its code.
        repo, _ = _checkout(tmp_path, "https://github.com/wagner-austin/API.git")
        (repo / "tools" / "hpc-wake" / "cycle.log").write_text("== tick\n", encoding="utf-8")

        assert read_checkout(repo)["dirty"] is False

    def test_a_directory_outside_any_checkout_is_refused(self, tmp_path: pathlib.Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()

        with pytest.raises(AppError) as caught:
            read_checkout(outside)

        assert caught.value.code is ErrorCode.CONFIG_ERROR
        assert caught.value.message.startswith(
            f"CHECKOUT_UNREADABLE: HEAD in {outside} did not resolve to a commit"
        )

    def test_a_checkout_without_an_origin_is_refused(self, tmp_path: pathlib.Path) -> None:
        repo = tmp_path / "checkout"
        committed_repository(repo, {"run.py": "print('tick')\n"}, "refs/heads/main")

        with pytest.raises(AppError) as caught:
            read_checkout(repo)

        assert caught.value.code is ErrorCode.CONFIG_ERROR
        assert caught.value.message.startswith(
            f"CHECKOUT_UNREADABLE: git -C {repo} remote get-url origin exited 2: "
        )

    def test_an_origin_naming_no_repository_is_refused(self, tmp_path: pathlib.Path) -> None:
        repo, _ = _checkout(tmp_path, "/")

        with pytest.raises(AppError) as caught:
            read_checkout(repo)

        assert caught.value.code is ErrorCode.CONFIG_ERROR
        assert caught.value.message == (
            f"CHECKOUT_UNREADABLE: the origin remote of {repo} is '/', which names no repository"
        )
