"""The CLI: one flag, one cycle, and the ``__main__`` form the pump uses."""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.mcp_testing import FakeHttpPost, posted_ok

from lock_wake import _test_hooks
from lock_wake.cli import wake
from lock_wake.position import position_path, read_offset
from tests.conftest import CONFIGURED_ENV, journal_line, pin_env, stage_journal


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
        _test_hooks.http_post = FakeHttpPost([posted_ok()])

        assert wake.main(["--journal", str(journal)]) == 0
        assert read_offset(position_path(journal)) == len(content)

    def test_a_missing_flag_refuses(self) -> None:
        with pytest.raises(ValueError, match="--journal"):
            wake.main([])


class TestInvocationForms:
    def test_the_console_entry_point_runs_and_exits_zero(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, b"")
        _test_hooks.http_post = FakeHttpPost([])
        saved_argv = list(sys.argv)
        sys.argv = ["lock-wake", "--journal", str(journal)]
        try:
            with pytest.raises(SystemExit) as caught:
                wake.entrypoint()
        finally:
            sys.argv[:] = saved_argv
        assert caught.value.code == 0
        assert emitted == ["journal quiet; offset 0"]

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
        sys.argv = ["lock-wake", "--journal", str(journal)]
        try:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv[:] = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module
        assert caught.value.code == 0
        assert emitted == ["journal quiet; offset 0"]
