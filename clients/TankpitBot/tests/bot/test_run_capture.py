"""``Bot.run`` owning the Xvfb + ffmpeg pair for a streamed session.

Split from :mod:`tests.bot.test_run` at the 600-line ceiling; the cut
is by role — that module covers the session loop, this one covers the
capture lifecycle wrapped around it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tankpit_bot import _test_hooks
from tankpit_bot.browser import _test_hooks as browser_hooks
from tankpit_bot.browser._test_hooks import AutoscrollEnforcerProtocol
from tankpit_bot.sniffer.world_service import WorldService
from tests.conftest import FakeEnv, FakeFileSystem

_STOP = Path("__nonexistent_stop_file__")


def _stub_autoscroll_hook() -> AutoscrollEnforcerProtocol:
    """Replace the autoscroll enforcement with an inert stand-in.

    Returns:
        The original hook, for the caller's ``finally``.
    """
    from tankpit_bot._test_hooks import CDPSessionProtocol, PageWaitProtocol
    from tankpit_bot.types.message import CapturedMessage

    original = browser_hooks.ensure_autoscroll_off

    def _inert(
        page: PageWaitProtocol,
        cdp: CDPSessionProtocol,
        messages: list[CapturedMessage],
        ws: WorldService,
    ) -> None:
        del page, cdp, messages, ws

    browser_hooks.ensure_autoscroll_off = _inert
    return original


def test_the_real_child_environment_is_a_copy_not_a_live_view() -> None:
    """Mutating the returned mapping cannot reach back into this process.

    The copy is the contract: the launch path overlays ``DISPLAY`` on
    it, and an overlay on a live view would set ``DISPLAY`` for the
    whole process — exactly the mutation the design avoids.
    """
    env = _test_hooks._real_child_environment()
    env["TANKPIT_TEST_PROBE_KEY"] = "x"
    again = _test_hooks._real_child_environment()
    assert "TANKPIT_TEST_PROBE_KEY" not in again


class TestBotRunWithCapture:
    """The whole capture contract across one session."""

    def test_run_starts_capture_launches_on_the_display_and_stops_it(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
        tmp_path: Path,
    ) -> None:
        """Display first, kiosk launch with DISPLAY overlaid, encoder
        after game-ready, both helpers ended when ``run`` returns.

        The teardown holds on THIS path too — the session ends via
        KeyboardInterrupt, and a leaked Xvfb is the exact operator
        complaint the design answers.

        The helpers are REAL child processes (``sys.executable``
        sleepers spawned through the production spawner); only which
        program runs is substituted, so the teardown assertion is
        about processes the operating system actually ended.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.stream import _test_hooks as stream_hooks
        from tankpit_bot.stream.capture import ffmpeg_command, x11_socket_path, xvfb_command
        from tankpit_bot.stream.types import StreamConfigDict
        from tests.fakes.bot import FakeSyncPlaywrightContextManagerBot

        _ = (fake_env, fake_fs)
        config = StreamConfigDict(
            display=7,
            width=704,
            height=544,
            scale=2,
            fps=30,
            bitrate_kbps=1000,
            segment_seconds=2,
            hls_dir=str(tmp_path / "hls"),
        )

        commands: list[list[str]] = []
        processes: list[stream_hooks.CaptureProcessProtocol] = []

        def substituting_spawner(
            command: list[str], log_path: Path
        ) -> stream_hooks.CaptureProcessProtocol:
            commands.append(command)
            process = stream_hooks._real_spawn_capture_process(
                [sys.executable, "-c", "import time; time.sleep(60)"], log_path
            )
            processes.append(process)
            return process

        stream_hooks.spawn_capture_process = substituting_spawner
        saved_path_exists = _test_hooks.path_exists

        def socket_or_saved(path: Path) -> bool:
            if path == x11_socket_path(7):
                return True
            return saved_path_exists(path)

        _test_hooks.path_exists = socket_or_saved
        _test_hooks.child_environment = lambda: {"KEEP": "1"}
        playwright_cm = FakeSyncPlaywrightContextManagerBot(interrupt_after=15)
        _test_hooks.sync_playwright = lambda: playwright_cm
        original_autoscroll = _stub_autoscroll_hook()

        try:
            bot = Bot(
                "https://test.tankpit.com/",
                headless=False,
                stream_config=config,
            )
            bot.run(session_seconds=0, stop_file_path=_STOP)
        finally:
            browser_hooks.ensure_autoscroll_off = original_autoscroll
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait(10.0)

        assert commands == [xvfb_command(config), ffmpeg_command(config)]
        playwright = playwright_cm._playwright
        if playwright is None:
            raise AssertionError("the fake playwright was never started")
        browser_type = playwright._chromium
        assert browser_type.launch_envs == [{"KEEP": "1", "DISPLAY": ":7"}]
        launch_args = browser_type.launch_args[0]
        if launch_args is None:
            raise AssertionError("the launch was handed no args at all")
        # The config's scale travels to Chromium: the client's picture
        # size is entirely DPR, and geometry was chosen for factor 2.
        assert launch_args == ["--force-device-scale-factor=2"]
        # Ended by run()'s own teardown, not the reaper in the finally.
        for name, process in zip(("Xvfb", "ffmpeg"), processes, strict=True):
            if process.poll() is None:
                raise AssertionError(f"{name} stand-in still running after run()")
