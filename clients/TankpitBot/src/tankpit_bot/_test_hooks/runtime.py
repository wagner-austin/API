"""Misc runtime hooks: argv, static-key discovery, replay dispatch.

Three small, otherwise-orphaned process-level hooks live here so they
do not require their own modules:

* ``get_argv`` -- ``sys.argv`` accessor (tests substitute a fixed list).
* ``find_best_static_byte`` -- optional override that lets tests inject
  the XOR static-byte discovery without invoking the real algorithm.
* ``process_received_message_hook`` -- replay-time entry point into the
  sniffer decoder pipeline.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from typing import Protocol


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


class ProcessReceivedMessageProtocol(Protocol):
    """Protocol for processing a received WebSocket message payload."""

    def __call__(self, payload: str) -> None:
        """Process a received message payload.

        Args:
            payload: Base64-encoded WebSocket frame payload.
        """
        ...


def _real_process_received_message(payload: str) -> None:
    """Real implementation - delegates to sniffer decoders.

    Args:
        payload: Base64-encoded WebSocket frame payload.
    """
    from tankpit_bot.sniffer.decoders import process_received_message

    process_received_message(payload)


process_received_message_hook: ProcessReceivedMessageProtocol = _real_process_received_message


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


# The real implementation IS os._exit -- bound directly, no wrapper.
# It must bypass interpreter teardown: the watchdog fires precisely
# because normal teardown is hung.
force_exit: Callable[[int], None] = os._exit


__all__ = [
    "FindBestStaticByteProtocol",
    "ProcessReceivedMessageProtocol",
    "StartWatchdogProtocol",
    "find_best_static_byte",
    "force_exit",
    "get_argv",
    "process_received_message_hook",
    "start_watchdog",
]
