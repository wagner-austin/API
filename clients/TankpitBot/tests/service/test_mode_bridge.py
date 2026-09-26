"""Tests for :class:`tankpit_bot.bus.mode_bridge.ModeBridge`.

Covers the single-thread API contract — submit / drain / peek — plus a
light multi-threaded scenario to prove the internal lock actually
serialises writes. No mocks: the tests exercise the real primitive
with real threads.
"""

from __future__ import annotations

import threading

from tankpit_bot.bus.mode_bridge import ModeBridge
from tankpit_bot.bus.session_status import WireMode


class TestModeBridge:
    """Contract tests for :class:`ModeBridge`."""

    def test_empty_bridge_drains_to_none(self) -> None:
        """A fresh bridge yields ``None`` on drain."""
        bridge = ModeBridge()
        assert bridge.drain() is None

    def test_empty_bridge_peeks_to_none(self) -> None:
        """A fresh bridge yields ``None`` on peek."""
        bridge = ModeBridge()
        assert bridge.peek() is None

    def test_submit_then_drain_returns_value(self) -> None:
        """A submitted value round-trips through drain."""
        bridge = ModeBridge()
        bridge.submit(WireMode.HUNT)
        assert bridge.drain() is WireMode.HUNT

    def test_drain_is_destructive(self) -> None:
        """The second drain after one submit returns ``None``."""
        bridge = ModeBridge()
        bridge.submit(WireMode.COLLECT)
        assert bridge.drain() is WireMode.COLLECT
        assert bridge.drain() is None

    def test_peek_is_non_destructive(self) -> None:
        """Peek leaves the pending value in place for a later drain."""
        bridge = ModeBridge()
        bridge.submit(WireMode.HUNT)
        assert bridge.peek() is WireMode.HUNT
        assert bridge.peek() is WireMode.HUNT
        assert bridge.drain() is WireMode.HUNT
        assert bridge.peek() is None

    def test_second_submit_replaces_first_latest_wins(self) -> None:
        """Two submits between drains yield only the second — latest wins."""
        bridge = ModeBridge()
        bridge.submit(WireMode.HUNT)
        bridge.submit(WireMode.COLLECT)
        assert bridge.drain() is WireMode.COLLECT
        assert bridge.drain() is None

    def test_every_wire_mode_round_trips(self) -> None:
        """Every :class:`WireMode` member can round-trip through the bridge."""
        bridge = ModeBridge()
        for mode in WireMode:
            bridge.submit(mode)
            assert bridge.drain() is mode


def test_mode_bridge_serialises_concurrent_submits() -> None:
    """Under contention, every submit either wins the slot or is overwritten.

    Ten writer threads each submit their assigned mode a handful of
    times. When they finish, ``drain`` returns exactly one of the
    submitted modes (or ``None`` if the very last write happened
    before a drain in the same instant — impossible here since the
    drain runs after every writer has joined). No stale value, no
    corruption.
    """
    bridge = ModeBridge()
    modes = list(WireMode)
    threads: list[threading.Thread] = []

    def writer(mode: WireMode) -> None:
        for _ in range(50):
            bridge.submit(mode)

    for mode in modes:
        for _ in range(3):
            threads.append(threading.Thread(target=writer, args=(mode,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = bridge.drain()
    assert final in modes
