"""``align --repo`` moves a PEER checkout, on its own default branch, with real git.

MCPs board task a0d34aa8. A session host holds main checkouts of
repositories with no mover of their own (ziggurat's corvis-stick was 2
commits behind on 2026-10-20), so the one mover is named the checkout. Each
case builds a bare origin and a clone of it under the test's directory and
runs the command through the registry from this package's directory, so the
checkout moved is the one named, never the one the command runs in.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from platform_core.errors import AppError

from maketools.cli import dispatch

#: Who the fixture commits are by: git refuses a commit with no identity.
IDENTITY: tuple[str, ...] = ("-c", "user.email=t@example.invalid", "-c", "user.name=t")


def git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` and return its stdout, stripped.

    Args:
        cwd: The directory.
        *args: The git subcommand and its arguments.

    Returns:
        Standard output.
    """
    done = subprocess.run(
        ("git", *IDENTITY, *args), cwd=cwd, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def commit_file(seed: Path, name: str, text: str) -> str:
    """Commit one file in the seed clone and push it to its branch.

    Args:
        seed: The clone that writes origin.
        name: The file.
        text: Its contents.

    Returns:
        The new commit's full sha.
    """
    (seed / name).write_text(text, encoding="utf-8", newline="\n")
    git(seed, "add", name)
    git(seed, "commit", "-q", "-m", name)
    git(seed, "push", "-q", "origin", "HEAD")
    return git(seed, "rev-parse", "HEAD")


def origin_and_peer(tmp_path: Path, branch: str) -> tuple[Path, Path, str]:
    """A bare origin whose default branch is ``branch``, and a peer clone of it.

    Args:
        tmp_path: Where both go.
        branch: The default branch.

    Returns:
        The seed clone that writes origin, the peer, and the peer's HEAD.
    """
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", branch, str(origin))
    git(tmp_path, "clone", "-q", str(origin), "seed")
    seed = tmp_path / "seed"
    git(seed, "checkout", "-q", "-b", branch)
    first = commit_file(seed, "a.txt", "one\n")
    git(tmp_path, "clone", "-q", str(origin), "peer")
    return seed, tmp_path / "peer", first


def test_a_peer_whose_default_branch_is_master_is_moved_to_a_pushed_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seed, peer, first = origin_and_peer(tmp_path, "master")
    second = commit_file(seed, "b.txt", "two\n")

    assert dispatch(["align", "--repo", str(peer), "--commit", second, "--drop"]) == 0

    assert git(peer, "rev-parse", "HEAD") == second
    assert capsys.readouterr().out.splitlines() == [
        f"align: {first[:12]} -> {second[:12]}, 0 local commit(s) dropped"
    ]


def test_a_peer_on_main_is_moved_forward_and_its_untracked_files_carried(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seed, peer, first = origin_and_peer(tmp_path, "main")
    second = commit_file(seed, "b.txt", "two\n")
    third = commit_file(seed, "c.txt", "three\n")
    (peer / "notes.txt").write_text("mine\n", encoding="utf-8")

    assert dispatch(["align", "--repo", str(peer), "--commit", third, "--drop"]) == 0

    assert git(peer, "rev-parse", "HEAD") == third
    assert (peer / "notes.txt").read_text(encoding="utf-8") == "mine\n"
    assert capsys.readouterr().out.splitlines() == [
        f"align: {first[:12]} -> {third[:12]}, 0 local commit(s) dropped",
        "  carried untracked notes.txt",
    ]
    assert second != third


def test_a_commit_the_default_branch_does_not_carry_is_refused_and_nothing_moves(
    tmp_path: Path,
) -> None:
    seed, peer, first = origin_and_peer(tmp_path, "master")
    git(seed, "checkout", "-q", "-b", "side")
    side = commit_file(seed, "s.txt", "side\n")

    with pytest.raises(AppError, match=rf"{side[:12]} is not on origin/master.*push it first"):
        dispatch(["align", "--repo", str(peer), "--commit", side, "--drop"])

    assert git(peer, "rev-parse", "HEAD") == first
