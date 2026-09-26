"""The CLI: one flag, one cycle, and the ``__main__`` form the pump uses."""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.mcp_testing import FakeHttpPost, announcing_poster

from fleet_health_wake import _test_hooks
from fleet_health_wake.cli import wake
from tests.conftest import CONFIGURED_ENV, TRANSITIONS_LINE, offset_of, pin_env, stage_journal


class TestMain:
    def test_runs_one_cycle_against_the_named_journal(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        content = TRANSITIONS_LINE.encode("utf-8")
        journal = stage_journal(tmp_path, content)
        _test_hooks.http_post = announcing_poster()

        assert wake.main(["--journal", str(journal)]) == 0
        assert offset_of(journal) == len(content)
        assert len(emitted) == 1

    def test_a_missing_journal_flag_refuses(self) -> None:
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
        sys.argv = ["fleet-health-wake", "--journal", str(journal)]
        try:
            with pytest.raises(SystemExit) as caught:
                wake.entrypoint()
        finally:
            sys.argv[:] = saved_argv
        assert caught.value.code == 0
        assert emitted == ["health journal quiet; offset 0"]

    def test_running_as_a_module_actually_runs(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        """Without the ``if __name__`` block, ``python -m`` would import, run
        nothing and exit 0, which from the pump's side reads as a quiet fleet."""
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, b"")
        _test_hooks.http_post = FakeHttpPost([])
        module_name = "fleet_health_wake.cli.wake"
        saved_argv = list(sys.argv)
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["fleet-health-wake", "--journal", str(journal)]
        try:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv[:] = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module
        assert caught.value.code == 0
        assert emitted == ["health journal quiet; offset 0"]
