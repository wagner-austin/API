"""Coverage tests for _test_hooks/runtime.py: _real_start_watchdog."""

from __future__ import annotations

import threading

from tankpit_bot._test_hooks.runtime import _real_start_watchdog


def test_real_start_watchdog_fires_callback() -> None:
    """_real_start_watchdog arms a daemon timer that calls the callback."""
    fired = threading.Event()

    def on_fire() -> None:
        fired.set()

    _real_start_watchdog(0.05, on_fire)

    assert fired.wait(timeout=2.0), "watchdog callback did not fire within 2s"
