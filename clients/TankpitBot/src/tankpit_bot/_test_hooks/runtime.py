"""Misc runtime hooks: argv, static-key discovery, replay dispatch.

Three small, otherwise-orphaned process-level hooks live here so they
do not require their own modules:

* ``get_argv`` -- ``sys.argv`` accessor (tests substitute a fixed list).
* ``find_best_static_byte`` -- optional override that lets tests inject
  the XOR static-byte discovery without invoking the real algorithm.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Protocol


def _real_get_current_time_ms() -> int:
    """Real implementation -- wall-clock milliseconds since the Unix epoch.

    Returns:
        Current Unix timestamp in milliseconds.
    """
    return int(time.time() * 1000)


#: Hookable clock. Production code that needs "now in ms" -- the
#: dispatcher's wire-timestamp stamping, the AI scan cooldown, the
#: bot's tick cadence -- imports this name and calls it. Tests that
#: drive multi-tick scenarios with controlled time replace this
#: attribute via save-and-restore so the scenario's clock is the
#: only ms-source in the system.
get_current_time_ms: Callable[[], int] = _real_get_current_time_ms


class FindBestStaticByteProtocol(Protocol):
    """Protocol for finding the best static key byte.

    Matches browser.find_best_static_byte signature.
    """

    def __call__(self, raw_first_bytes: list[int], magic_first_byte: int) -> tuple[int, int]:
        """Find the static key's first byte that maximizes known signature matches.

        Args:
            raw_first_bytes: First XOR-encoded bytes from binary messages.
            magic_first_byte: ASCII value of magic key's first character.

        Returns:
            Tuple of (best_static_byte, match_count).
        """
        ...


find_best_static_byte: FindBestStaticByteProtocol | None = None
"""Default is None - browser.py uses its own implementation when None."""


def _real_get_argv() -> list[str]:
    """Real implementation - returns sys.argv.

    Returns:
        The command line arguments.
    """
    return sys.argv


get_argv: Callable[[], list[str]] = _real_get_argv


class StartWatchdogProtocol(Protocol):
    """Protocol for arming a one-shot daemon watchdog timer."""

    def __call__(self, seconds: float, on_fire: Callable[[], None]) -> None:
        """Arm a daemon timer that calls ``on_fire`` after ``seconds``.

        Args:
            seconds: Delay before firing.
            on_fire: Zero-argument callback to invoke on expiry.
        """
        ...


def _real_start_watchdog(seconds: float, on_fire: Callable[[], None]) -> None:
    """Real implementation - daemon ``threading.Timer``.

    Daemon so a normally exiting process is never kept alive by an
    armed watchdog; the timer simply dies with the process.

    Args:
        seconds: Delay before firing.
        on_fire: Zero-argument callback to invoke on expiry.
    """
    timer = threading.Timer(seconds, on_fire)
    timer.daemon = True
    timer.start()


start_watchdog: StartWatchdogProtocol = _real_start_watchdog


class InstallSignalHandlersProtocol(Protocol):
    """Protocol for registering SIGINT / SIGTERM handlers on the bot CLI.

    Production binds ``signal.signal`` for both signals; tests inject
    a fake that records the registered callback and exercise it
    synchronously without touching process-wide signal state.
    """

    def __call__(self, on_interrupt: Callable[[], None]) -> None:
        """Register ``on_interrupt`` for SIGINT and SIGTERM.

        Args:
            on_interrupt: Zero-argument callback invoked when either
                signal fires.
        """
        ...


def _real_install_signal_handlers(on_interrupt: Callable[[], None]) -> None:
    """Real implementation -- bind SIGINT and SIGTERM to ``on_interrupt``.

    The signal-handler signature ``(signum, frame)`` is adapted to the
    zero-argument ``on_interrupt`` callback by ignoring both arguments.
    SIGINT is what Ctrl+C raises on every platform; SIGTERM is what
    process supervisors (systemd, Docker, ``kill PID``) send for a
    graceful shutdown request. Both go through the same handler so
    callers do not have to discriminate.

    Args:
        on_interrupt: Zero-argument callback to invoke when either
            signal fires.
    """
    import signal as _signal
    from types import FrameType

    def _handle(signum: int, frame: FrameType | None) -> None:
        _ = (signum, frame)
        on_interrupt()

    _signal.signal(_signal.SIGINT, _handle)
    _signal.signal(_signal.SIGTERM, _handle)


install_signal_handlers: InstallSignalHandlersProtocol = _real_install_signal_handlers


# The real implementation IS os._exit -- bound directly, no wrapper.
# It must bypass interpreter teardown: the watchdog fires precisely
# because normal teardown is hung.
force_exit: Callable[[int], None] = os._exit


__all__ = [
    "FindBestStaticByteProtocol",
    "InstallSignalHandlersProtocol",
    "StartWatchdogProtocol",
    "_real_get_current_time_ms",
    "find_best_static_byte",
    "force_exit",
    "get_argv",
    "get_current_time_ms",
    "install_signal_handlers",
    "start_watchdog",
]
