"""The align command's flags, and the report it prints through the registry."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.errors import AppError

from maketools.align_checkout import DEFAULT_BRANCH_ARGV
from maketools.align_command import AlignArguments, parse_align_arguments
from maketools.cli import dispatch
from tests.conftest import World, ok

TARGET = "b" * 40


@pytest.fixture()
def cwd(tmp_path: Path) -> Generator[Path, None, None]:
    before = Path.cwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(before)


def test_the_flags_are_read_in_any_order_and_an_empty_drop_is_allowed() -> None:
    assert parse_align_arguments(["--repo", ".", "--commit", TARGET]) == AlignArguments(
        repo=".", commit=TARGET, drop=[]
    )
    assert parse_align_arguments(["--commit", TARGET, "--repo", "../stick", "--drop"]) == (
        AlignArguments(repo="../stick", commit=TARGET, drop=[])
    )
    assert parse_align_arguments(["--drop", "--repo", ".", "--commit", TARGET]) == (
        AlignArguments(repo=".", commit=TARGET, drop=[])
    )
    assert parse_align_arguments(
        ["--repo", ".", "--drop", "1234,5678", "--commit", TARGET]
    ) == AlignArguments(repo=".", commit=TARGET, drop=["1234", "5678"])


def test_a_missing_repo_or_commit_or_a_stray_argument_is_a_usage_error() -> None:
    with pytest.raises(AppError, match=r"align needs --repo PATH and --commit SHA"):
        parse_align_arguments(["--repo", ".", "--drop", "1234"])
    with pytest.raises(AppError, match=r"align needs --repo PATH and --commit SHA"):
        parse_align_arguments(["--commit", TARGET])
    with pytest.raises(
        AppError, match=r"align takes --repo PATH --commit SHA \[--drop SHA,SHA\], got \['x'\]"
    ):
        parse_align_arguments(["x"])
    with pytest.raises(AppError, match=r"got \['--commit'\]"):
        parse_align_arguments(["--commit"])
    with pytest.raises(AppError, match=r"got \['--repo'\]"):
        parse_align_arguments(["--repo"])


def test_the_registry_runs_the_align_on_its_default_branch_and_prints_its_report(
    world: World, cwd: Path
) -> None:
    """The branch is the checkout's own (slime's master); which checkout is
    moved is test_align_peer_real's, since only real git can show it."""
    for argv, answer in (
        (("git", "fetch", "origin"), ok("")),
        (DEFAULT_BRANCH_ARGV, ok("origin/master\n")),
        (("git", "rev-parse", "--verify", f"{TARGET}^{{commit}}"), ok(TARGET)),
        (("git", "merge-base", "--is-ancestor", TARGET, "origin/master"), ok("")),
        (("git", "status", "--porcelain"), ok("")),
        (("git", "ls-tree", "-r", "--name-only", TARGET), ok("")),
        (("git", "rev-parse", "HEAD"), ok(TARGET)),
    ):
        world.capturing_answers[argv] = answer
    assert dispatch(["align", "--repo", "../slime", "--commit", TARGET, "--drop"]) == 0
    assert world.lines == [f"align: already at {TARGET[:12]}; nothing moved"]
    assert world.captured[:2] == [("git", "fetch", "origin"), DEFAULT_BRANCH_ARGV]
