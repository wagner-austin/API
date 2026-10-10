"""The align command's flags, and the report it prints through the registry."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.errors import AppError

from maketools.align_command import parse_align_arguments
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
    assert parse_align_arguments(["--commit", TARGET]) == (TARGET, [])
    assert parse_align_arguments(["--commit", TARGET, "--drop"]) == (TARGET, [])
    assert parse_align_arguments(["--drop", "--commit", TARGET]) == (TARGET, [])
    assert parse_align_arguments(["--drop", "1234,5678", "--commit", TARGET]) == (
        TARGET,
        ["1234", "5678"],
    )


def test_a_missing_commit_or_a_stray_argument_is_a_usage_error() -> None:
    with pytest.raises(AppError, match=r"align needs --commit SHA"):
        parse_align_arguments(["--drop", "1234"])
    with pytest.raises(AppError, match=r"align takes --commit SHA \[--drop SHA,SHA\], got \['x'\]"):
        parse_align_arguments(["x"])
    with pytest.raises(AppError, match=r"got \['--commit'\]"):
        parse_align_arguments(["--commit"])


def test_the_registry_runs_the_align_in_the_cwd_and_prints_its_report(
    world: World, cwd: Path
) -> None:
    for argv, answer in (
        (("git", "fetch", "origin"), ok("")),
        (("git", "rev-parse", "--verify", f"{TARGET}^{{commit}}"), ok(TARGET)),
        (("git", "merge-base", "--is-ancestor", TARGET, "origin/main"), ok("")),
        (("git", "status", "--porcelain"), ok("")),
        (("git", "ls-tree", "-r", "--name-only", TARGET), ok("")),
        (("git", "rev-parse", "HEAD"), ok(TARGET)),
    ):
        world.capturing_answers[argv] = answer
    assert dispatch(["align", "--commit", TARGET, "--drop"]) == 0
    assert world.lines == [f"align: already at {TARGET[:12]}; nothing moved"]
    assert world.captured[0] == ("git", "fetch", "origin")
