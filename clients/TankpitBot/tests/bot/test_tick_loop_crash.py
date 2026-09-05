"""The crash boundary: an unhandled tick exception still leaves artifacts.

[[bot-behavior-contract]] §1.3 promised ``exit_reason="crashed"`` from
the 2026-06-20 write-up, but no writer existed until 2026-07-31 — a
crashed session simply vanished from ``runs/bot/_index.tsv``. The
boundary finalizes the scorecard, ``latest.summary.txt``, and the
index row, then RE-RAISES so the process still fails loudly.

The defect is injected by rebinding ``tick_body._tick_once`` — the
same seam the lifecycle suite's browser-closed test uses — because
that is where any real mid-tick defect surfaces. (The previous
injection point, a chunk bus whose demand signal raised, left the
codebase with the in-page caster on 2026-09-05.)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot.bot import tick_body as tick_body_module
from tankpit_bot.bot.base import Bot
from tests.bot._tick_loop_fakes import _FakePage
from tests.conftest import FakeEnv, FakeFileSystem
from tests.fakes import FakeCDPSession


def _fail_tick_once_with_runtime_error(bot: Bot) -> None:
    """Stand-in tick body modelling an unhandled mid-tick defect.

    Args:
        bot: Ignored — the defect fires before any bot state is read.

    Raises:
        RuntimeError: Always.
    """
    del bot
    raise RuntimeError("tick wiring broke mid-tick")


class TestCrashedExitReason:
    """An unhandled tick exception produces ``exit_reason=crashed``."""

    def test_unhandled_tick_exception_finalizes_artifacts_and_reraises(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """The crash writes summary + index row, then propagates."""
        from tankpit_bot.bot.tick_loop import run_tick_loop
        from tankpit_bot.diagnostics.runs_index import DEFAULT_INDEX_PATH, decode_row
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        _ = fake_env
        configure_bot_runtime_logging("20260731-000002")
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._cdp = FakeCDPSession()

        saved_tick_once = tick_body_module._tick_once
        tick_body_module._tick_once = _fail_tick_once_with_runtime_error
        try:
            with pytest.raises(RuntimeError, match="tick wiring broke mid-tick"):
                run_tick_loop(
                    bot,
                    _FakePage(),
                    session_seconds=0,
                    stop_file_path=Path("C:/tmp/never_exists.sentinel"),
                )
        finally:
            tick_body_module._tick_once = saved_tick_once

        text = fake_fs.get_written_files()[str(DEFAULT_INDEX_PATH)]
        data_lines = [line for line in text.splitlines() if line and not line.startswith("stamp\t")]
        if len(data_lines) != 1:
            raise AssertionError(f"expected 1 index row, got {len(data_lines)}")
        row = decode_row(data_lines[0])
        assert row["exit_reason"] == "crashed"
        assert row["ticks"] == 0
        summary = fake_fs.get_written_files()["runs\\bot\\latest.summary.txt"]
        assert summary == (
            "TANKPIT SESSION SUMMARY\n"
            "========================================\n"
            "Ticks:    0\n"
            "Exit:     crashed\n"
            "Kills:    0\n"
            "Shots:    0 (0 hits, 0 misses, 0 rejected)\n"
            "Hit rate: n/a\n"
            "Blocked:  0\n"
            "========================================\n"
            "Fuel:     0\n"
            "Duals:    0\n"
            "Homings:  0\n"
            "Radars:   0\n"
            "========================================\n"
            "Mode:     UNSET/\n"
        )
