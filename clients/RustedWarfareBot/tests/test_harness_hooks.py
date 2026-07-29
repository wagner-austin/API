"""The harness's real hook implementations, against a real filesystem.

The fakes elsewhere prove the harness's control flow. These prove the things
the fakes stand in for actually behave the way the fakes pretend they do --
which is the half of dependency injection that is easy to leave untested and
that fails silently when it is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rw_bot.harness import _test_hooks


def test_a_directory_and_its_parents_are_created(tmp_path: Path) -> None:
    _test_hooks.make_dirs(tmp_path / "a" / "b" / "c")
    assert (tmp_path / "a" / "b" / "c").is_dir()


def test_creating_a_directory_that_exists_is_not_an_error(tmp_path: Path) -> None:
    _test_hooks.make_dirs(tmp_path / "a")
    _test_hooks.make_dirs(tmp_path / "a")
    assert (tmp_path / "a").is_dir()


def test_lines_are_written_and_read_back_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "card.txt"
    _test_hooks.write_text_lines(target, ("### tank-s1", "verdict survived"))
    assert _test_hooks.read_text_lines(target) == ("### tank-s1", "verdict survived")


def test_writing_replaces_rather_than_appends(tmp_path: Path) -> None:
    """A replayed match must not leave the previous attempt's lines behind."""
    target = tmp_path / "card.txt"
    _test_hooks.write_text_lines(target, ("first", "attempt"))
    _test_hooks.write_text_lines(target, ("second",))
    assert _test_hooks.read_text_lines(target) == ("second",)


def test_an_empty_file_reads_as_no_lines(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    _test_hooks.write_text_lines(target, ())
    assert _test_hooks.read_text_lines(target) == ()


def test_names_are_listed_sorted_and_without_their_parent(tmp_path: Path) -> None:
    for name in ("jvm64", "assets", "saves"):
        (tmp_path / name).mkdir()
    (tmp_path / "game-lib.jar").write_text("", encoding="utf-8")
    assert _test_hooks.list_names(tmp_path) == ("assets", "game-lib.jar", "jvm64", "saves")


def test_a_file_is_copied_into_a_directory(tmp_path: Path) -> None:
    source = tmp_path / "game-lib.jar"
    source.write_text("jar bytes", encoding="utf-8")
    destination = tmp_path / "clone"
    destination.mkdir()
    _test_hooks.copy_entry(source, destination)
    assert (destination / "game-lib.jar").read_text(encoding="utf-8") == "jar bytes"


def test_a_tree_is_copied_whole(tmp_path: Path) -> None:
    """The clone needs the JVM, which is nested three deep."""
    (tmp_path / "jvm64" / "bin").mkdir(parents=True)
    (tmp_path / "jvm64" / "bin" / "java.exe").write_text("exe", encoding="utf-8")
    destination = tmp_path / "clone"
    destination.mkdir()
    _test_hooks.copy_entry(tmp_path / "jvm64", destination)
    assert (destination / "jvm64" / "bin" / "java.exe").read_text(encoding="utf-8") == "exe"


def test_copying_over_an_existing_tree_is_not_an_error(tmp_path: Path) -> None:
    """A clone left half made by an interrupted sweep is completed, not
    refused.
    """
    (tmp_path / "assets" / "units").mkdir(parents=True)
    (tmp_path / "assets" / "units" / "tank.ini").write_text("stats", encoding="utf-8")
    destination = tmp_path / "clone"
    (destination / "assets").mkdir(parents=True)
    _test_hooks.copy_entry(tmp_path / "assets", destination)
    _test_hooks.copy_entry(tmp_path / "assets", destination)
    assert (destination / "assets" / "units" / "tank.ini").read_text(encoding="utf-8") == "stats"


def test_a_path_that_was_never_created_does_not_exist(tmp_path: Path) -> None:
    assert not _test_hooks.path_exists(tmp_path / "absent")
    assert _test_hooks.path_exists(tmp_path)


def test_a_child_process_output_is_captured_in_order() -> None:
    status, lines = _test_hooks.run_capture(
        [sys.executable, "-c", "print('first'); print('second')"]
    )
    assert status == 0
    assert lines == ("first", "second")


def test_both_streams_are_captured_because_the_transcript_spans_them() -> None:
    """The launcher writes progress to one stream and the planner writes its
    scorecard to the other, and a sweep needs one transcript of the match.
    """
    status, lines = _test_hooks.run_capture(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('launcher\\n'); print('verdict survived')",
        ]
    )
    assert status == 0
    assert set(lines) == {"launcher", "verdict survived"}


def test_a_failing_child_reports_its_status_rather_than_raising() -> None:
    """The exit status is data: a match that fails is filed as a failure, and
    raising here would take down the whole batch instead.
    """
    status, _ = _test_hooks.run_capture([sys.executable, "-c", "raise SystemExit(3)"])
    assert status == 3


def test_a_program_that_does_not_exist_raises() -> None:
    with pytest.raises(OSError):
        _test_hooks.run_capture(["no-such-program-anywhere", "--version"])


def test_the_process_arguments_are_read_after_the_program_name() -> None:
    original = sys.argv
    sys.argv = ["sweep", "jobs.txt", "demo"]
    try:
        assert _test_hooks.read_argv() == ["jobs.txt", "demo"]
    finally:
        sys.argv = original


def test_a_line_is_written_with_its_terminator(capsys: pytest.CaptureFixture[str]) -> None:
    _test_hooks.write_line("[sweep] done")
    assert capsys.readouterr().out == "[sweep] done\n"
