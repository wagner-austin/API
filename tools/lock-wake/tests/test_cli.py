"""The CLI: one flag, one cycle, and the ``__main__`` form the pump uses."""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.mcp_testing import FakeHttpPost, announcing_poster

from lock_wake import _test_hooks
from lock_wake.cli import wake
from tests.conftest import (
    CONFIGURED_ENV,
    action_journal_path,
    check_journal_path,
    journal_line,
    offset_of,
    pin_env,
    stage_journal,
)


def local_argv(tmp_path: pathlib.Path, journal: pathlib.Path) -> list[str]:
    """The hub row's three journals, the fleet journal named by the caller."""
    return [
        "--journal",
        str(journal),
        "--check-journal",
        str(check_journal_path(tmp_path)),
        "--action-journal",
        str(action_journal_path(tmp_path)),
    ]


class TestMain:
    def test_runs_one_cycle_against_the_named_journal(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        content = (
            journal_line(ts="2026-09-09T19:28:00.0000000Z", kind="acquired")
            + journal_line(ts="2026-09-09T19:28:05.0000000Z", kind="released")
        ).encode("utf-8")
        journal = stage_journal(tmp_path, content)
        _test_hooks.http_post = announcing_poster()

        assert wake.main(local_argv(tmp_path, journal)) == 0
        assert offset_of(journal) == len(content)

    def test_a_missing_journal_flag_refuses(self) -> None:
        with pytest.raises(ValueError, match="--journal"):
            wake.main([])

    def test_a_missing_check_journal_flag_refuses(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--check-journal"):
            wake.main(["--journal", str(tmp_path / ".fleet-events.jsonl")])

    def test_a_missing_action_journal_flag_refuses(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--action-journal"):
            wake.main(
                [
                    "--journal",
                    str(tmp_path / ".fleet-events.jsonl"),
                    "--check-journal",
                    str(tmp_path / ".check-events.jsonl"),
                ]
            )

    @pytest.mark.parametrize("remote_flag", ["--remote-journal", "--remote-action-journal"])
    def test_a_remote_journal_beside_a_local_one_refuses(
        self, tmp_path: pathlib.Path, remote_flag: str
    ) -> None:
        with pytest.raises(
            ValueError,
            match=f"{remote_flag} reads one remote journal and takes no --journal or "
            "--check-journal or --action-journal",
        ):
            wake.main(
                [
                    remote_flag,
                    "diphtheria:/home/corvis/PROJECTS/MCPs/.fleet-events.jsonl",
                    *local_argv(tmp_path, tmp_path / ".fleet-events.jsonl"),
                    "--cursor-dir",
                    str(tmp_path),
                ]
            )

    def test_both_remote_journals_in_one_row_refuse(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="are one row each"):
            wake.main(
                [
                    "--remote-journal",
                    "diphtheria:/home/corvis/PROJECTS/MCPs/.fleet-events.jsonl",
                    "--remote-action-journal",
                    "diphtheria:/home/corvis/PROJECTS/MCPs/.action-events.jsonl",
                    "--cursor-dir",
                    str(tmp_path),
                ]
            )

    def test_a_remote_journal_without_a_cursor_dir_refuses(self) -> None:
        with pytest.raises(ValueError, match="--cursor-dir is required"):
            wake.main(["--remote-journal", "diphtheria:/home/corvis/j.jsonl"])

    def test_a_cursor_dir_for_local_journals_refuses(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(
            ValueError, match="--cursor-dir belongs to --remote-journal or --remote-action-journal"
        ):
            wake.main(
                [
                    *local_argv(tmp_path, tmp_path / ".fleet-events.jsonl"),
                    "--cursor-dir",
                    str(tmp_path),
                ]
            )


class TestInvocationForms:
    def test_the_console_entry_point_runs_and_exits_zero(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, b"")
        _test_hooks.http_post = FakeHttpPost([])
        saved_argv = list(sys.argv)
        sys.argv = ["lock-wake", *local_argv(tmp_path, journal)]
        try:
            with pytest.raises(SystemExit) as caught:
                wake.entrypoint()
        finally:
            sys.argv[:] = saved_argv
        assert caught.value.code == 0
        assert emitted == ["hub journals quiet; offsets 0, 0, 0"]

    def test_running_as_a_module_actually_runs(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        """The half that silently goes missing without an ``if __name__``
        block: ``python -m lock_wake.cli.wake`` would import, run nothing
        and exit 0 -- which from the pump's side reads as a quiet fleet."""
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, b"")
        _test_hooks.http_post = FakeHttpPost([])
        module_name = "lock_wake.cli.wake"
        saved_argv = list(sys.argv)
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["lock-wake", *local_argv(tmp_path, journal)]
        try:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv[:] = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module
        assert caught.value.code == 0
        assert emitted == ["hub journals quiet; offsets 0, 0, 0"]
