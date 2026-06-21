"""Tests for the bot-runs CLI.

CLI commands operate on the default index path; tests inject a fake
filesystem so each command sees its own controlled corpus.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import (
    AppendTextProtocol,
    PathExistsProtocol,
    ReadTextProtocol,
)
from tankpit_bot.diagnostics import bot_runs_cli
from tankpit_bot.diagnostics.runs_index import (
    DEFAULT_INDEX_PATH,
    HEADER_LINE,
    BotRunIndexRowDict,
    encode_row,
)


class _FakeFileSystem:
    """Save-and-restore fake mirroring the runs_index test helper."""

    def __init__(self) -> None:
        """Initialise with no virtual files registered."""
        self._files: dict[str, str] = {}

    def write(self, path: Path, content: str) -> None:
        """Register a virtual file's contents."""
        self._files[str(path)] = content

    def path_exists(self, path: Path) -> bool:
        """Return True when ``path`` was registered via :meth:`write`."""
        return str(path) in self._files

    def read_text(self, path: Path) -> str:
        """Return the contents of ``path``."""
        return self._files[str(path)]

    def append_text(self, path: Path, content: str) -> None:
        """Append ``content`` to the virtual file at ``path``."""
        existing = self._files.get(str(path), "")
        self._files[str(path)] = existing + content


def _install_fake_filesystem() -> tuple[
    _FakeFileSystem, PathExistsProtocol, ReadTextProtocol, AppendTextProtocol
]:
    """Swap the real script hooks for a fake; return originals for restore."""
    fake = _FakeFileSystem()
    original_path_exists: PathExistsProtocol = _test_hooks.path_exists
    original_read_text: ReadTextProtocol = _test_hooks.read_text
    original_append_text: AppendTextProtocol = _test_hooks.append_text
    _test_hooks.path_exists = fake.path_exists
    _test_hooks.read_text = fake.read_text
    _test_hooks.append_text = fake.append_text
    return (fake, original_path_exists, original_read_text, original_append_text)


def _row(stamp: str, *, exit_reason: str = "completed") -> BotRunIndexRowDict:
    """Build a deterministic row varying only the discriminators."""
    return BotRunIndexRowDict(
        stamp=stamp,
        duration_s=155,
        exit_reason=exit_reason,
        ticks=1543,
        stalls=2,
        shots_fired=16,
        kills=3,
        kills_per_min=1.16,
    )


class _CliTestBase:
    """Base setup/teardown for CLI tests: install + restore fake FS."""

    def setup_method(self) -> None:
        """Install the fake filesystem."""
        (
            self._fake,
            self._original_path_exists,
            self._original_read_text,
            self._original_append_text,
        ) = _install_fake_filesystem()

    def teardown_method(self) -> None:
        """Restore the real ``_test_hooks`` bindings."""
        _test_hooks.path_exists = self._original_path_exists
        _test_hooks.read_text = self._original_read_text
        _test_hooks.append_text = self._original_append_text

    def _populate(self, rows: list[BotRunIndexRowDict]) -> None:
        """Seed the default index with ``rows`` and the canonical header."""
        text = HEADER_LINE + "".join(encode_row(row) for row in rows)
        self._fake.write(DEFAULT_INDEX_PATH, text)


class TestRunList(_CliTestBase):
    """Tests for ``bot-runs list``."""

    def test_prints_header_and_every_row(self, capsys: pytest.CaptureFixture[str]) -> None:
        """List output is header + every row in file order."""
        self._populate([_row("20260620-150138"), _row("20260620-160000")])
        assert bot_runs_cli.run_list() == 0
        out = capsys.readouterr().out
        lines = out.rstrip("\n").splitlines()
        assert lines[0] == "\t".join(
            [
                "stamp",
                "duration_s",
                "exit_reason",
                "ticks",
                "stalls",
                "shots_fired",
                "kills",
                "kills_per_min",
            ]
        )
        assert "20260620-150138" in lines[1]
        assert "20260620-160000" in lines[2]

    def test_prints_no_runs_recorded_when_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        """An empty index prints a friendly stub line and still exits 0."""
        # Header-only file (no data rows yet).
        self._fake.write(DEFAULT_INDEX_PATH, HEADER_LINE)
        assert bot_runs_cli.run_list() == 0
        out = capsys.readouterr().out
        assert "(no runs recorded)" in out

    def test_runs_against_missing_index_file_succeeds(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No file at all -> still prints header and the stub."""
        assert bot_runs_cli.run_list() == 0
        out = capsys.readouterr().out
        assert "(no runs recorded)" in out


class TestRunFind(_CliTestBase):
    """Tests for ``bot-runs find <pattern>``."""

    def test_matches_substring_on_stamp(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Stamp substring matches are case-insensitive."""
        self._populate(
            [
                _row("20260620-150138"),
                _row("20260620-160000"),
                _row("20260101-090000"),
            ]
        )
        assert bot_runs_cli.run_find("0620") == 0
        out = capsys.readouterr().out
        assert "20260620-150138" in out
        assert "20260620-160000" in out
        assert "20260101-090000" not in out

    def test_matches_substring_on_exit_reason(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Exit-reason substring matches identify interrupted runs."""
        self._populate(
            [
                _row("a", exit_reason="completed"),
                _row("b", exit_reason="interrupted"),
                _row("c", exit_reason="stop_file"),
            ]
        )
        assert bot_runs_cli.run_find("interrupt") == 0
        out = capsys.readouterr().out
        assert "interrupted" in out
        assert "completed" not in out.split("interrupted")[0].splitlines()[-1]

    def test_returns_one_when_no_match(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No matches -> exit code 1 with an actionable stderr line."""
        self._populate([_row("a")])
        assert bot_runs_cli.run_find("nonexistent") == 1
        err = capsys.readouterr().err
        assert "no rows matched 'nonexistent'" in err


class TestRunShow(_CliTestBase):
    """Tests for ``bot-runs show <stamp>``."""

    def test_prints_long_form(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Show output is one ``label: value`` line per column."""
        self._populate([_row("20260620-150138")])
        assert bot_runs_cli.run_show("20260620-150138") == 0
        out = capsys.readouterr().out
        assert "stamp:          20260620-150138" in out
        assert "duration_s:     155" in out
        assert "kills_per_min:  1.16" in out

    def test_returns_one_when_stamp_missing(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Missing stamp -> exit code 1 with an actionable stderr line."""
        self._populate([_row("a")])
        assert bot_runs_cli.run_show("nope") == 1
        err = capsys.readouterr().err
        assert "no row with stamp 'nope'" in err


class TestRunDispatcher(_CliTestBase):
    """Tests for the ``run(argv)`` argv dispatcher."""

    def test_empty_argv_prints_usage_and_returns_one(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Empty argv -> usage block to stderr and exit code 1."""
        assert bot_runs_cli.run([]) == 1
        err = capsys.readouterr().err
        assert "usage: bot-runs" in err

    def test_list_with_extra_argument_is_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Subcommand arity is checked."""
        assert bot_runs_cli.run(["list", "unexpected"]) == 1
        err = capsys.readouterr().err
        assert "usage: bot-runs" in err

    def test_find_without_pattern_is_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``find`` requires exactly one argument."""
        assert bot_runs_cli.run(["find"]) == 1
        err = capsys.readouterr().err
        assert "usage: bot-runs" in err

    def test_show_without_stamp_is_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``show`` requires exactly one argument."""
        assert bot_runs_cli.run(["show"]) == 1
        err = capsys.readouterr().err
        assert "usage: bot-runs" in err

    def test_unknown_command_is_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Unknown subcommand -> usage block, exit code 1."""
        assert bot_runs_cli.run(["garbage"]) == 1
        err = capsys.readouterr().err
        assert "usage: bot-runs" in err

    def test_list_dispatches_to_run_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``run(["list"])`` returns ``run_list``'s exit code."""
        self._populate([])
        assert bot_runs_cli.run(["list"]) == 0
        out = capsys.readouterr().out
        assert "(no runs recorded)" in out

    def test_find_dispatches_to_run_find(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``run(["find", needle])`` returns ``run_find``'s exit code."""
        self._populate([_row("20260620-150138")])
        assert bot_runs_cli.run(["find", "150138"]) == 0
        out = capsys.readouterr().out
        assert "20260620-150138" in out

    def test_show_dispatches_to_run_show(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``run(["show", stamp])`` returns ``run_show``'s exit code."""
        self._populate([_row("20260620-150138")])
        assert bot_runs_cli.run(["show", "20260620-150138"]) == 0
        out = capsys.readouterr().out
        assert "stamp:          20260620-150138" in out


class TestMain(_CliTestBase):
    """Tests for ``main()`` and the ``__main__`` runpy entrypoint."""

    def test_main_exits_with_run_code(self) -> None:
        """``main()`` propagates ``run()``'s exit code via ``SystemExit``."""
        self._populate([])
        old_argv = sys.argv
        sys.argv = ["bot-runs", "list"]
        try:
            with pytest.raises(SystemExit) as exc:
                bot_runs_cli.main()
            assert exc.value.code == 0
        finally:
            sys.argv = old_argv

    def test_module_entrypoint_runs_main(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``python -m tankpit_bot.diagnostics.bot_runs_cli`` executes ``main``."""
        self._populate([_row("20260620-150138")])
        old_argv = sys.argv
        sys.argv = ["tankpit_bot.diagnostics.bot_runs_cli", "show", "20260620-150138"]
        try:
            sys.modules.pop("tankpit_bot.diagnostics.bot_runs_cli", None)
            with pytest.raises(SystemExit) as exc:
                runpy.run_module(
                    "tankpit_bot.diagnostics.bot_runs_cli",
                    run_name="__main__",
                )
            assert exc.value.code == 0
        finally:
            sys.argv = old_argv
        out = capsys.readouterr().out
        assert "stamp:          20260620-150138" in out
