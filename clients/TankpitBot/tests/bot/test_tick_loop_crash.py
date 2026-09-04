"""The crash boundary: an unhandled tick exception still leaves artifacts.

[[bot-behavior-contract]] §1.3 promised ``exit_reason="crashed"`` from
the 2026-06-20 write-up, but no writer existed until 2026-07-31 — a
crashed session simply vanished from ``runs/bot/_index.tsv``. The
boundary finalizes the scorecard, ``latest.summary.txt``, and the
index row, then RE-RAISES so the process still fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot.bus.frame_bus import FrameSubscriber, FrameSubscriberProtocol
from tests.bot._tick_loop_fakes import _FakePage
from tests.conftest import FakeEnv, FakeFileSystem
from tests.fakes import FakeCDPSession


class _ExplodingFrameBus:
    """Protocol-complete frame bus whose demand signal raises.

    ``subscriber_count`` is read inside the tick's exception boundary
    (``_sync_live_view_demand``), so raising there models any
    unhandled mid-tick defect without reaching around the sanctioned
    constructor DI seam.

    The bot that uses this bus MUST be built with a cast URL. A bot
    without one has no caster, and ``_sync_live_view_demand`` returns
    before it ever reads demand -- so the injected defect never fires,
    the loop never ends, and the suite hangs instead of failing. That
    is not hypothetical: it cost a 30-minute run on 2026-09-04.
    """

    def publish(self, frame: bytes) -> None:
        """Drop the frame — no viewers in this scenario."""
        del frame

    def subscribe(self) -> FrameSubscriberProtocol:
        """Return a real (inert) subscriber."""
        return FrameSubscriber()

    def unsubscribe(self, subscriber: FrameSubscriberProtocol) -> None:
        """Close the handed-back subscriber."""
        subscriber.close()

    def subscriber_count(self) -> int:
        """Explode — the injected mid-tick defect."""
        raise RuntimeError("frame bus wiring broke mid-tick")

    def latest(self) -> bytes | None:
        """No cached frame."""
        return None


class TestCrashedExitReason:
    """An unhandled tick exception produces ``exit_reason=crashed``."""

    def test_unhandled_tick_exception_finalizes_artifacts_and_reraises(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """The crash writes summary + index row, then propagates."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import run_tick_loop
        from tankpit_bot.diagnostics.runs_index import DEFAULT_INDEX_PATH, decode_row
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        configure_bot_runtime_logging("20260731-000002")
        bot = Bot(
            "https://test.tankpit.com/",
            headless=True,
            frame_bus=_ExplodingFrameBus(),
            cast_url="http://127.0.0.1:27100/cast",
        )
        bot._cdp = FakeCDPSession()

        with pytest.raises(RuntimeError, match="frame bus wiring broke mid-tick"):
            run_tick_loop(
                bot,
                _FakePage(),
                session_seconds=0,
                stop_file_path=Path("C:/tmp/never_exists.sentinel"),
            )

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
